"""Leak-free utilities for the N5 NEZ-PU-RankCal experiment."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F


KNOWN_CENTERS = ("hup", "lzu", "multicenter", "pediatric")


def normalize_channel(value: object) -> str:
    return str(value).strip().lower().replace(" ", "").replace("-", "_").replace(".", "_")


def reliability_from_nez_confidence(q: np.ndarray | float, *, eta: float, r_min: float) -> np.ndarray:
    if eta <= 0.0:
        raise ValueError("N5 eta must be positive")
    if not 0.0 <= r_min <= 1.0:
        raise ValueError("N5 r_min must be between 0 and 1")
    values = np.asarray(q, dtype=np.float64)
    if not np.isfinite(values).all():
        raise ValueError("N5 OOF NEZ confidence contains non-finite values")
    return np.clip(np.power(1.0 - np.clip(values, 0.0, 1.0), eta), r_min, 1.0)


def build_reliability_lookup(
    records: Sequence[Mapping[str, Any]],
    *,
    eta: float,
    r_min: float,
) -> tuple[dict[tuple[str, str], float], list[dict[str, Any]]]:
    lookup: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []
    for record in records:
        subject_id = str(record["subject_id"])
        channels = list(record["canonical_channels"])
        labels_ez = np.asarray(record["labels_ez"], dtype=np.float64)
        score_nez = np.asarray(record["score_nez"], dtype=np.float64)
        valid = np.asarray(record["channel_mask"], dtype=bool)
        center = str(record.get("center", "unknown")).strip().lower()
        for channel, label_ez, q_value, is_valid in zip(channels, labels_ez, score_nez, valid):
            if not is_valid or label_ez < 0.0:
                continue
            reliability = 1.0
            if label_ez > 0.5:
                reliability = float(reliability_from_nez_confidence(q_value, eta=eta, r_min=r_min))
            key = (subject_id, normalize_channel(channel))
            if key in lookup:
                raise ValueError(f"Duplicate N5 OOF channel key: {key}")
            lookup[key] = reliability
            rows.append({
                "subject_id": subject_id,
                "center": center,
                "channel_name": str(channel),
                "label_ez": float(label_ez),
                "label_nez": float(1.0 - label_ez),
                "oof_score_nez": float(q_value),
                "ez_label_reliability": reliability,
            })
    return lookup, rows


def reliability_tensors(
    batch: Mapping[str, Any],
    lookup: Mapping[tuple[str, str], float],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    labels = batch["labels_ez"]
    reliability = torch.ones_like(labels, device=device, dtype=dtype)
    matched = torch.zeros_like(labels, device=device, dtype=torch.bool)
    for patient_idx, subject_id in enumerate(batch["subject_id"]):
        for channel_idx, channel in enumerate(batch["canonical_channels"][patient_idx]):
            key = (str(subject_id), normalize_channel(channel))
            if key in lookup:
                reliability[patient_idx, channel_idx] = float(lookup[key])
                matched[patient_idx, channel_idx] = True
    return reliability, matched


def confidence_weighted_pu_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    reliability: torch.Tensor,
    *,
    lambda_u: float,
    gamma_pos: float,
    gamma_neg: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    valid = channel_mask & (labels_ez >= 0.0)
    nez = valid & (labels_ez <= 0.5)
    ez_u = valid & (labels_ez > 0.5)
    p_nez = torch.sigmoid(logits)
    positive_terms = torch.pow(1.0 - p_nez[nez], float(gamma_pos)) * F.binary_cross_entropy_with_logits(
        logits[nez], torch.ones_like(logits[nez]), reduction="none"
    ) if bool(nez.any()) else logits.new_zeros((0,))
    negative_terms = torch.pow(p_nez[ez_u], float(gamma_neg)) * F.binary_cross_entropy_with_logits(
        logits[ez_u], torch.zeros_like(logits[ez_u]), reduction="none"
    ) if bool(ez_u.any()) else logits.new_zeros((0,))
    r = reliability[ez_u].clamp(0.0, 1.0)
    loss_nez = positive_terms.mean() if positive_terms.numel() else logits.sum() * 0.0
    loss_ez_u = (r * negative_terms).sum() / r.sum().clamp_min(1e-6) if r.numel() else logits.sum() * 0.0
    total = loss_nez + float(lambda_u) * loss_ez_u
    return total, {
        "n5_pu_loss": float(total.detach().cpu()),
        "n5_nez_positive_loss": float(loss_nez.detach().cpu()),
        "n5_ez_unlabeled_loss": float(loss_ez_u.detach().cpu()),
        "n5_mean_ez_reliability": float(r.mean().detach().cpu()) if r.numel() else 0.0,
        "n5_n_nez_channels": float(nez.sum().detach().cpu()),
        "n5_n_ez_channels": float(ez_u.sum().detach().cpu()),
    }


def patient_reliability_rank_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    reliability: torch.Tensor,
    *,
    margin: float,
    hard_negatives_per_patient: int,
    reliable_ez_gate: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    patient_losses: list[torch.Tensor] = []
    n_pairs = 0
    n_patients = 0
    for patient_idx in range(logits.shape[0]):
        valid = channel_mask[patient_idx] & (labels_ez[patient_idx] >= 0.0)
        nez = valid & (labels_ez[patient_idx] <= 0.5)
        reliable_ez = valid & (labels_ez[patient_idx] > 0.5) & (
            reliability[patient_idx] >= float(reliable_ez_gate)
        )
        if not bool(nez.any()) or not bool(reliable_ez.any()):
            continue
        ez_indices = torch.where(reliable_ez)[0]
        k = min(max(1, int(hard_negatives_per_patient)), int(ez_indices.numel()))
        # Among reliable observed-EZ channels, retain the most NEZ-like semi-hard negatives.
        selected_local = torch.topk(logits[patient_idx, ez_indices], k=k, largest=True).indices
        selected_ez = ez_indices[selected_local]
        z_nez = logits[patient_idx, nez].reshape(-1, 1)
        z_ez = logits[patient_idx, selected_ez].reshape(1, -1)
        r_ez = reliability[patient_idx, selected_ez].reshape(1, -1)
        pair_terms = r_ez * F.softplus(float(margin) - z_nez + z_ez)
        denominator = r_ez.sum().clamp_min(1e-6) * max(1, int(z_nez.shape[0]))
        patient_losses.append(pair_terms.sum() / denominator)
        n_pairs += int(z_nez.shape[0] * z_ez.shape[1])
        n_patients += 1
    if not patient_losses:
        zero = logits.sum() * 0.0
        return zero, {"n5_rank_loss": 0.0, "n5_rank_pairs": 0.0, "n5_rank_patients": 0.0}
    loss = torch.stack(patient_losses).mean()
    return loss, {
        "n5_rank_loss": float(loss.detach().cpu()),
        "n5_rank_pairs": float(n_pairs),
        "n5_rank_patients": float(n_patients),
    }


@dataclass(frozen=True)
class HierarchicalCalibration:
    temperature: float
    center_biases: dict[str, float]
    l2_weight: float
    n_channels: int
    n_patients: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "temperature": self.temperature,
            "center_biases": dict(self.center_biases),
            "center_bias_unseen_default": 0.0,
            "l2_weight": self.l2_weight,
            "n_channels": self.n_channels,
            "n_patients": self.n_patients,
        }


def _records_to_calibration_arrays(records: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    logits: list[float] = []
    labels: list[float] = []
    centers: list[str] = []
    for record in records:
        score_nez = np.asarray(record["score_nez"], dtype=np.float64)
        label_nez = np.asarray(record["labels_nez"], dtype=np.float64)
        valid = np.asarray(record["channel_mask"], dtype=bool) & (label_nez >= 0.0)
        clipped = np.clip(score_nez[valid], 1e-6, 1.0 - 1e-6)
        logits.extend(np.log(clipped / (1.0 - clipped)).tolist())
        labels.extend(label_nez[valid].tolist())
        centers.extend([str(record.get("center", "unknown")).strip().lower()] * int(valid.sum()))
    if not logits:
        raise ValueError("Cannot fit N5 calibration without valid OOF channels")
    return np.asarray(logits), np.asarray(labels), np.asarray(centers, dtype=object)


def fit_hierarchical_calibration(
    records: Sequence[Mapping[str, Any]],
    *,
    l2_weight: float = 1.0,
    max_iter: int = 100,
) -> HierarchicalCalibration:
    logits_np, labels_np, centers_np = _records_to_calibration_arrays(records)
    logits = torch.tensor(logits_np, dtype=torch.float64)
    labels = torch.tensor(labels_np, dtype=torch.float64)
    center_index = torch.tensor(
        [KNOWN_CENTERS.index(str(value)) if str(value) in KNOWN_CENTERS else -1 for value in centers_np],
        dtype=torch.long,
    )
    raw_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    biases = torch.nn.Parameter(torch.zeros(len(KNOWN_CENTERS), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([raw_temperature, biases], lr=0.5, max_iter=max(1, int(max_iter)), line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        temperature = F.softplus(raw_temperature) + 1e-3
        row_bias = torch.zeros_like(logits)
        known = center_index >= 0
        row_bias[known] = biases[center_index[known]]
        objective = F.binary_cross_entropy_with_logits(logits / temperature + row_bias, labels)
        objective = objective + float(l2_weight) * biases.square().mean()
        objective.backward()
        return objective

    optimizer.step(closure)
    temperature = float((F.softplus(raw_temperature) + 1e-3).detach().cpu())
    center_biases = {name: float(biases[idx].detach().cpu()) for idx, name in enumerate(KNOWN_CENTERS)}
    return HierarchicalCalibration(
        temperature=temperature,
        center_biases=center_biases,
        l2_weight=float(l2_weight),
        n_channels=int(len(logits_np)),
        n_patients=int(len({str(record["subject_id"]) for record in records})),
    )


def apply_hierarchical_calibration(
    records: Sequence[Mapping[str, Any]],
    calibration: HierarchicalCalibration,
) -> list[dict[str, Any]]:
    calibrated: list[dict[str, Any]] = []
    for source in records:
        record = dict(source)
        score_nez = np.asarray(source["score_nez"], dtype=np.float64)
        clipped = np.clip(score_nez, 1e-6, 1.0 - 1e-6)
        logits = np.log(clipped / (1.0 - clipped))
        center = str(source.get("center", "unknown")).strip().lower()
        bias = float(calibration.center_biases.get(center, 0.0))
        calibrated_score = 1.0 / (1.0 + np.exp(-(logits / calibration.temperature + bias)))
        record["score_nez_uncalibrated"] = score_nez.astype(np.float32)
        record["score_ez_uncalibrated"] = (1.0 - score_nez).astype(np.float32)
        record["score_nez"] = calibrated_score.astype(np.float32)
        record["score_ez"] = (1.0 - calibrated_score).astype(np.float32)
        record["scores"] = record["score_nez"]
        record["n5_calibration_temperature"] = float(calibration.temperature)
        record["n5_calibration_center_bias"] = bias
        calibrated.append(record)
    return calibrated


def threshold_candidates(records: Sequence[Mapping[str, Any]]) -> np.ndarray:
    scores = np.concatenate([
        np.asarray(record["score_nez"], dtype=np.float64)[np.asarray(record["channel_mask"], dtype=bool)]
        for record in records
    ])
    scores = np.unique(scores[np.isfinite(scores)])
    if scores.size == 0:
        return np.asarray([0.5], dtype=np.float64)
    if scores.size > 399:
        scores = np.quantile(scores, np.linspace(0.005, 0.995, 399))
    candidates = np.unique(np.concatenate(([0.0], scores, [1.0])))
    return candidates[(candidates >= 0.0) & (candidates <= 1.0)]


def select_constrained_nez_threshold(
    records: Sequence[Mapping[str, Any]],
    summarize: Callable[[Sequence[Mapping[str, Any]], float], Mapping[str, float]],
    *,
    min_ez_f1: float,
    min_balanced_accuracy: float,
    min_worst_center_f1: float,
) -> tuple[float, dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    feasible: list[dict[str, Any]] = []
    for threshold in threshold_candidates(records):
        summary = dict(summarize(records, float(threshold)))
        row = {"classification_threshold": float(threshold), **summary}
        row["constraints_satisfied"] = bool(
            float(summary.get("patient_macro_ez_f1", 0.0)) >= min_ez_f1
            and float(summary.get("patient_macro_balanced_accuracy", 0.0)) >= min_balanced_accuracy
            and float(summary.get("worst_center_f1", 0.0)) >= min_worst_center_f1
        )
        rows.append(row)
        if row["constraints_satisfied"]:
            feasible.append(row)
    if not feasible:
        best_observed = max(
            rows,
            key=lambda row: min(
                float(row.get("patient_macro_ez_f1", 0.0)) - min_ez_f1,
                float(row.get("patient_macro_balanced_accuracy", 0.0)) - min_balanced_accuracy,
                float(row.get("worst_center_f1", 0.0)) - min_worst_center_f1,
            ),
        )
        raise RuntimeError(
            "N5 constrained threshold selection found no feasible OOF threshold; "
            f"closest threshold={best_observed['classification_threshold']:.6f}, "
            f"ez_f1={float(best_observed.get('patient_macro_ez_f1', 0.0)):.6f}, "
            f"balanced_accuracy={float(best_observed.get('patient_macro_balanced_accuracy', 0.0)):.6f}, "
            f"worst_center_f1={float(best_observed.get('worst_center_f1', 0.0)):.6f}"
        )
    # Smaller center gap is better, hence the negation in the final tie breaker.
    selected = max(
        feasible,
        key=lambda row: (
            float(row.get("patient_macro_nez_f1", 0.0)),
            float(row.get("patient_macro_nez_recall", 0.0)),
            float(row.get("patient_macro_nez_precision", 0.0)),
            float(row.get("worst_center_nez_f1", 0.0)),
            -float(row.get("center_gap_nez_f1", math.inf)),
        ),
    )
    return float(selected["classification_threshold"]), dict(selected), rows


def assert_oof_subject_partition(records: Sequence[Mapping[str, Any]], train_subjects: Sequence[str], test_subjects: Sequence[str]) -> dict[str, Any]:
    expected = set(map(str, train_subjects))
    observed = {str(record["subject_id"]) for record in records}
    forbidden = set(map(str, test_subjects))
    leakage = sorted(observed & forbidden)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if leakage or missing or extra:
        raise RuntimeError(f"Invalid N5 inner OOF partition: leakage={leakage}, missing={missing}, extra={extra}")
    return {
        "outer_train_patient_count": len(expected),
        "oof_patient_count": len(observed),
        "outer_test_patient_count": len(forbidden),
        "test_patient_overlap": leakage,
        "complete_outer_train_oof_coverage": True,
    }
