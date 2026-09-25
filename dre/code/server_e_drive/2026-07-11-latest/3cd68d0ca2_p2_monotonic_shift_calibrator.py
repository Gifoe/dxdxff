"""Leak-free patient-constant logit shift calibration for P2_RTC_SHIFT."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score


FEATURE_NAMES = (
    "logit_mean", "logit_std", "logit_median", "logit_iqr", "logit_q10", "logit_q25",
    "logit_q75", "logit_q90", "mean_prediction_entropy", "log1p_n_channels",
    "mean_robust_tail_probability", "mean_tail_gap", "mean_seizure_agreement",
    "mean_valid_seizure_count", "mean_tail_reliability",
)


def _valid(record: dict[str, Any]) -> np.ndarray:
    return np.asarray(record["channel_mask"], dtype=bool)


def patient_feature_matrix(records: Sequence[dict[str, Any]]) -> np.ndarray:
    rows: list[list[float]] = []
    for record in records:
        valid = _valid(record)
        logits = np.asarray(record["final_nez_logit"], dtype=float)[valid]
        prob = 1.0 / (1.0 + np.exp(-np.clip(logits, -40.0, 40.0)))
        entropy = -(prob * np.log(np.clip(prob, 1e-7, 1.0)) + (1.0 - prob) * np.log(np.clip(1.0 - prob, 1e-7, 1.0)))
        def mean_field(name: str) -> float:
            value = np.asarray(record.get(name, np.zeros_like(valid, dtype=float)), dtype=float)
            if value.ndim > 1:
                value = np.nanmean(value, axis=tuple(range(value.ndim - 1)))
            return float(np.nanmean(value[valid])) if value.size else 0.0
        rows.append([
            float(np.mean(logits)), float(np.std(logits)), float(np.median(logits)),
            float(np.quantile(logits, .75) - np.quantile(logits, .25)),
            float(np.quantile(logits, .10)), float(np.quantile(logits, .25)),
            float(np.quantile(logits, .75)), float(np.quantile(logits, .90)),
            float(np.mean(entropy)), float(np.log1p(valid.sum())),
            mean_field("seizure_nez_robust_tail_probability"), mean_field("seizure_nez_tail_gap"),
            mean_field("seizure_agreement"), mean_field("valid_seizure_count"), mean_field("tail_reliability"),
        ])
    return np.asarray(rows, dtype=np.float64)


def _loss(
    base_logits: list[torch.Tensor], labels: list[torch.Tensor], shifts: torch.Tensor, weights: torch.Tensor, l2: float,
) -> torch.Tensor:
    bce_rows, f1_rows = [], []
    for idx, (logits, target) in enumerate(zip(base_logits, labels)):
        shifted = logits + shifts[idx]
        probability = torch.sigmoid(shifted)
        bce_rows.append(F.binary_cross_entropy_with_logits(shifted, target))
        tp_nez = (probability * target).sum(); fp_nez = (probability * (1.0 - target)).sum(); fn_nez = ((1.0 - probability) * target).sum()
        f1_nez = 2 * tp_nez / (2 * tp_nez + fp_nez + fn_nez + 1e-6)
        ez_probability, ez_target = 1.0 - probability, 1.0 - target
        tp_ez = (ez_probability * ez_target).sum(); fp_ez = (ez_probability * (1.0 - ez_target)).sum(); fn_ez = ((1.0 - ez_probability) * ez_target).sum()
        f1_ez = 2 * tp_ez / (2 * tp_ez + fp_ez + fn_ez + 1e-6)
        f1_rows.append(1.0 - 0.5 * (f1_nez + f1_ez))
    return torch.stack(bce_rows).mean() + 0.20 * torch.stack(f1_rows).mean() + float(l2) * weights.square().sum()


@dataclass
class PatientShiftCalibrator:
    scaler: StandardScaler
    weights: np.ndarray
    bias: float
    b_max: float
    selected_lambda: float

    def predict(self, records: Sequence[dict[str, Any]]) -> np.ndarray:
        features = self.scaler.transform(patient_feature_matrix(records))
        raw = features @ self.weights + self.bias
        return self.b_max * np.tanh(raw)


def _fit(features: np.ndarray, records: Sequence[dict[str, Any]], *, l2: float, b_max: float, seed: int) -> PatientShiftCalibrator:
    scaler = StandardScaler().fit(features)
    x = torch.as_tensor(scaler.transform(features), dtype=torch.float32)
    base_logits, labels = [], []
    for record in records:
        valid = _valid(record)
        base_logits.append(torch.as_tensor(np.asarray(record["final_nez_logit"], dtype=np.float32)[valid]))
        labels.append(torch.as_tensor(np.asarray(record["labels_nez"], dtype=np.float32)[valid]))
    torch.manual_seed(int(seed))
    weights = torch.nn.Parameter(torch.zeros(x.shape[1], dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros((), dtype=torch.float32))
    optimizer = torch.optim.Adam([weights, bias], lr=0.03)
    for _ in range(250):
        optimizer.zero_grad(set_to_none=True)
        shifts = float(b_max) * torch.tanh(x @ weights + bias)
        loss = _loss(base_logits, labels, shifts, weights, float(l2))
        loss.backward(); optimizer.step()
    return PatientShiftCalibrator(scaler, weights.detach().numpy(), float(bias.detach()), float(b_max), float(l2))


def _macro_scores(records: Sequence[dict[str, Any]], shifts: np.ndarray) -> tuple[float, float]:
    macro, ez = [], []
    for record, shift in zip(records, shifts):
        valid = _valid(record)
        labels = np.asarray(record["labels_nez"], dtype=int)[valid]
        prediction = (np.asarray(record["final_nez_logit"], dtype=float)[valid] + float(shift) >= 0.0).astype(int)
        macro.append(float(f1_score(labels, prediction, average="macro", zero_division=0)))
        ez.append(float(f1_score(labels, prediction, pos_label=0, zero_division=0)))
    return float(np.mean(macro)), float(np.mean(ez))


def fit_patient_shift_calibrator(records: Sequence[dict[str, Any]], *, seed: int, b_max: float, lambda_grid: Sequence[float]) -> tuple[PatientShiftCalibrator, dict[str, Any]]:
    if len(records) < 5:
        raise ValueError("Patient shift calibration requires at least five inner-OOF patients")
    ids = [str(record["subject_id"]) for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Inner-OOF records must contain each patient exactly once")
    features = patient_feature_matrix(records)
    splitter = KFold(n_splits=5, shuffle=True, random_state=int(seed))
    candidates: list[dict[str, float]] = []
    for value in lambda_grid:
        shifts = np.zeros(len(records), dtype=float)
        for train, heldout in splitter.split(features):
            fitted = _fit(features[train], [records[index] for index in train], l2=float(value), b_max=b_max, seed=seed)
            shifts[heldout] = fitted.predict([records[index] for index in heldout])
        macro, ez = _macro_scores(records, shifts)
        candidates.append({"lambda": float(value), "cv_macro_f1": macro, "cv_ez_f1": ez})
    selected = max(candidates, key=lambda row: (row["cv_macro_f1"], row["cv_ez_f1"], row["lambda"]))
    calibrator = _fit(features, records, l2=selected["lambda"], b_max=b_max, seed=seed)
    audit = {"feature_names": list(FEATURE_NAMES), "candidate_scores": candidates, "selected_lambda": selected["lambda"], "b_max": b_max, "scaler_mean": calibrator.scaler.mean_.tolist(), "scaler_scale": calibrator.scaler.scale_.tolist()}
    return calibrator, audit


def apply_patient_shift(records: Sequence[dict[str, Any]], calibrator: PatientShiftCalibrator) -> tuple[list[dict[str, Any]], dict[str, float]]:
    shifts = calibrator.predict(records)
    result: list[dict[str, Any]] = []
    for record, shift in zip(records, shifts):
        enriched = dict(record)
        # Retain float64 here so the persisted calibrated-minus-raw delta is
        # exactly the single patient shift rather than a float32 rounding artifact.
        raw = np.asarray(record["final_nez_logit"], dtype=np.float64)
        valid = _valid(record)
        calibrated = raw + float(shift)
        score = 1.0 / (1.0 + np.exp(-np.clip(calibrated, -40.0, 40.0)))
        enriched.update({
            "raw_final_nez_logit": raw, "calibrated_nez_logit": calibrated,
            "final_nez_logit": calibrated, "score_nez": score, "score_ez": 1.0 - score,
            "patient_shift": np.full(raw.shape, float(shift), dtype=np.float64),
        })
        result.append(enriched)
    abs_shift = np.abs(shifts)
    return result, {"mean_abs_patient_shift": float(abs_shift.mean()), "p95_abs_patient_shift": float(np.quantile(abs_shift, .95)), "shift_saturation_rate": float(np.mean(abs_shift >= calibrator.b_max - 1e-6)), "ranking_pair_flip_fraction": 0.0}


__all__ = ["FEATURE_NAMES", "PatientShiftCalibrator", "apply_patient_shift", "fit_patient_shift_calibrator", "patient_feature_matrix"]
