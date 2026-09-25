"""Cross-fitted patient-adaptive thresholding in robust logit space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch import nn
import torch.nn.functional as F


PATH_SUMMARY_FIELDS = (
    "r_q05", "r_q10", "r_q25", "r_q50", "r_q75", "r_q90", "r_q95",
    "r_mean", "r_std", "r_min", "r_max", "mean_probability_entropy",
    "r_q75_minus_q25", "r_q90_minus_q10", "anchor_distance_mean",
    "anchor_distance_std", "seizure_agreement_mean", "seizure_agreement_std",
    "causal_valid_fraction", "causal_source_consistency_mean",
    "causal_suppression_release_mean", "log1p_valid_channel_count",
    "log1p_valid_seizure_count",
)


def robust_standardize_nez_logits(logits: torch.Tensor, channel_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Patient-wise median/IQR standardization excluding padding."""
    standardized = torch.zeros_like(logits)
    medians = torch.zeros(logits.shape[0], device=logits.device, dtype=logits.dtype)
    iqrs = torch.zeros_like(medians)
    for patient_idx in range(logits.shape[0]):
        valid = channel_mask[patient_idx].bool()
        if not torch.any(valid):
            continue
        values = logits[patient_idx, valid]
        median = torch.quantile(values, 0.50)
        iqr = torch.quantile(values, 0.75) - torch.quantile(values, 0.25)
        scale = iqr.clamp_min(1e-3)
        standardized[patient_idx, valid] = (values - median) / scale
        medians[patient_idx] = median
        iqrs[patient_idx] = iqr
    return standardized, medians, iqrs


def _mean_std(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if values.numel() == 0:
        zero = values.sum() * 0.0
        return zero, zero
    return values.mean(), values.std(unbiased=False)


def build_path_summary(outputs: dict[str, torch.Tensor], batch: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the fixed 23-scalar label- and center-blind PATH summary."""
    mask = outputs.get("channel_mask", batch["channel_mask"]).bool()
    standardized, _, _ = robust_standardize_nez_logits(outputs["final_nez_logit"], mask)
    rows: list[torch.Tensor] = []
    cp = outputs.get("causal_propagation_features", torch.zeros((*mask.shape, 6), device=mask.device, dtype=standardized.dtype))
    cp_valid = outputs.get("causal_feature_valid", torch.zeros_like(mask)).bool()
    seizure_mask = batch.get("seizure_mask", torch.zeros((mask.shape[0], 0), device=mask.device, dtype=torch.bool)).bool()
    for patient_idx in range(mask.shape[0]):
        valid = mask[patient_idx]
        r = standardized[patient_idx, valid]
        if r.numel() == 0:
            rows.append(torch.zeros(23, device=standardized.device, dtype=standardized.dtype))
            continue
        quantiles = torch.quantile(r, torch.tensor([.05, .10, .25, .50, .75, .90, .95], device=r.device, dtype=r.dtype))
        probability = torch.sigmoid(r).clamp(1e-7, 1.0 - 1e-7)
        entropy = -(probability * probability.log() + (1.0 - probability) * (1.0 - probability).log()).mean()
        anchor_mean, anchor_std = _mean_std(outputs["anchor_distance"][patient_idx, valid])
        seizure_mean, seizure_std = _mean_std(outputs["seizure_nez_agreement"][patient_idx, valid])
        causal_valid = valid & cp_valid[patient_idx]
        causal_fraction = causal_valid.sum().to(r.dtype) / valid.sum().clamp_min(1).to(r.dtype)
        if torch.any(causal_valid):
            consistency = cp[patient_idx, causal_valid, 5].mean()
            release = cp[patient_idx, causal_valid, 1].mean()
        else:
            consistency = r.sum() * 0.0
            release = r.sum() * 0.0
        rows.append(torch.stack((
            *quantiles.unbind(), r.mean(), r.std(unbiased=False), r.min(), r.max(), entropy,
            quantiles[4] - quantiles[2], quantiles[5] - quantiles[1], anchor_mean,
            anchor_std, seizure_mean, seizure_std, causal_fraction, consistency, release,
            torch.log1p(valid.sum().to(r.dtype)), torch.log1p(seizure_mask[patient_idx].sum().to(r.dtype)),
        )))
    summary = torch.stack(rows)
    if summary.shape[1] != len(PATH_SUMMARY_FIELDS) or not torch.isfinite(summary).all():
        raise RuntimeError("PATH summary violated its 23-dimensional finite contract")
    return summary, standardized


class PatientAdaptiveThresholdHead(nn.Module):
    def __init__(self, hidden_dim: int = 16, max_abs_threshold: float = 2.5) -> None:
        super().__init__()
        self.max_abs_threshold = float(max_abs_threshold)
        self.network = nn.Sequential(nn.Linear(23, int(hidden_dim)), nn.GELU(), nn.Dropout(0.10), nn.Linear(int(hidden_dim), 1))

    def forward(self, summary: torch.Tensor) -> torch.Tensor:
        if summary.ndim != 2 or summary.shape[1] != 23:
            raise ValueError(f"PATH summary must be [B,23], got {tuple(summary.shape)}")
        return self.max_abs_threshold * torch.tanh(self.network(summary).squeeze(-1))


def apply_patient_adaptive_threshold(
    standardized_nez_logit: torch.Tensor,
    predicted_threshold: torch.Tensor,
    channel_mask: torch.Tensor,
) -> dict[str, torch.Tensor]:
    predicted_nez = channel_mask.bool() & (standardized_nez_logit >= predicted_threshold[:, None])
    predicted_ez = channel_mask.bool() & ~predicted_nez
    return {
        "decision_probability_nez": torch.sigmoid(standardized_nez_logit - predicted_threshold[:, None]) * channel_mask,
        "predicted_nez_mask": predicted_nez,
        "predicted_ez_mask": predicted_ez,
    }


def oracle_standardized_threshold(standardized_logits: np.ndarray, labels_nez: np.ndarray) -> dict[str, float]:
    """Deterministic diagnostic target computed only for inner-OOF patients."""
    scores = np.asarray(standardized_logits, dtype=np.float64)
    labels = np.asarray(labels_nez, dtype=np.int64)
    if scores.ndim != 1 or labels.shape != scores.shape or scores.size == 0:
        raise ValueError("Oracle threshold requires non-empty aligned one-dimensional arrays")
    unique = np.unique(scores)
    candidates = [float(unique[0] - 1e-6), float(unique[-1] + 1e-6), 0.0]
    candidates.extend(float((left + right) / 2.0) for left, right in zip(unique[:-1], unique[1:]))
    best: tuple[tuple[float, float, float, float, float], dict[str, float]] | None = None
    for threshold in sorted(set(candidates)):
        prediction = (scores >= threshold).astype(np.int64)
        macro = float(f1_score(labels, prediction, average="macro", zero_division=0))
        nez = float(f1_score(labels, prediction, pos_label=1, zero_division=0))
        ez = float(f1_score(labels, prediction, pos_label=0, zero_division=0))
        balanced = float(balanced_accuracy_score(labels, prediction))
        key = (round(macro, 12), round(min(nez, ez), 12), round(balanced, 12), -abs(threshold), -threshold)
        row = {"oracle_standardized_threshold": threshold, "oracle_macro_f1": macro, "oracle_nez_f1": nez, "oracle_ez_f1": ez, "oracle_balanced_accuracy": balanced}
        if best is None or key > best[0]:
            best = (key, row)
    assert best is not None
    return best[1]


def path_training_loss(
    predicted_threshold: torch.Tensor,
    oracle_threshold: torch.Tensor,
    standardized_logits: Sequence[torch.Tensor],
    labels_nez: Sequence[torch.Tensor],
    *,
    threshold_weight: float = 0.10,
    soft_f1_weight: float = 0.20,
    threshold_l2_weight: float = 0.005,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    pbce_rows: list[torch.Tensor] = []
    soft_rows: list[torch.Tensor] = []
    for patient_idx, (scores, labels) in enumerate(zip(standardized_logits, labels_nez)):
        logits = scores - predicted_threshold[patient_idx]
        class_losses = []
        for value in (1, 0):
            selected = labels == value
            if torch.any(selected):
                class_losses.append(F.binary_cross_entropy_with_logits(logits[selected], labels[selected].to(logits.dtype)))
        pbce_rows.append(torch.stack(class_losses).mean())
        probability = torch.sigmoid(logits)
        y = labels.to(probability.dtype)
        f_nez = (2 * (probability * y).sum() + 1e-6) / (probability.sum() + y.sum() + 1e-6)
        p_ez, y_ez = 1 - probability, 1 - y
        f_ez = (2 * (p_ez * y_ez).sum() + 1e-6) / (p_ez.sum() + y_ez.sum() + 1e-6)
        soft_rows.append(1.0 - 0.5 * (f_nez + f_ez))
    pbce = torch.stack(pbce_rows).mean()
    soft_f1 = torch.stack(soft_rows).mean()
    regression = F.huber_loss(predicted_threshold, oracle_threshold, delta=0.5)
    l2 = predicted_threshold.square().mean()
    total = pbce + float(soft_f1_weight) * soft_f1 + float(threshold_weight) * regression + float(threshold_l2_weight) * l2
    return total, {"path_pbce": pbce, "path_soft_f1_loss": soft_f1, "path_threshold_huber": regression, "path_threshold_l2": l2}


__all__ = [
    "PATH_SUMMARY_FIELDS", "PatientAdaptiveThresholdHead", "apply_patient_adaptive_threshold",
    "build_path_summary", "oracle_standardized_threshold", "path_training_loss",
    "robust_standardize_nez_logits",
]
