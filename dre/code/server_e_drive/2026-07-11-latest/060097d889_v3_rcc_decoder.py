"""Leak-free raw-NEZ-probability decoder used by V3-RCC."""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score

FORMAL_DECISION_RULE = "fold_validation_global_raw_probability_threshold"
TRUE_K_DECISION_RULE = "true_k_topk_diagnostic"


def raw_threshold_candidates(values: np.ndarray, max_candidates: int = 1000) -> np.ndarray:
    values = np.unique(np.asarray(values, dtype=np.float64)[np.isfinite(values)])
    if values.size == 0:
        raise ValueError("No finite validation scores")
    if values.size > max_candidates:
        values = np.unique(np.quantile(values, np.linspace(0.0, 1.0, max_candidates)))
    mid = values if values.size == 1 else (values[:-1] + values[1:]) / 2.0
    return np.unique(np.clip(np.concatenate(([0.0, 1.0], mid)), 0.0, 1.0))


def _metrics(records: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    macro, balanced, ez_f1, centers = [], [], [], {}
    for row in records:
        valid = np.asarray(row["channel_mask"], dtype=bool)
        y = np.asarray(row["labels_nez"], dtype=int)[valid]
        pred = (np.asarray(row["score_nez"], dtype=float)[valid] >= threshold).astype(int)
        f1 = float(f1_score(y, pred, labels=[0, 1], average="macro", zero_division=0))
        macro.append(f1)
        balanced.append(float(balanced_accuracy_score(y, pred)))
        ez_f1.append(float(f1_score(y, pred, labels=[0], average="macro", zero_division=0)))
        centers.setdefault(str(row.get("center", "unknown")).lower(), []).append(f1)
    center_values = [float(np.mean(value)) for value in centers.values()]
    return {"patient_macro_f1": float(np.mean(macro)), "patient_balanced_accuracy": float(np.mean(balanced)), "patient_ez_f1": float(np.mean(ez_f1)), "center_gap_f1": float(max(center_values) - min(center_values)) if center_values else 0.0}


def fit_raw_probability_threshold(validation_records: list[dict[str, Any]], *, max_candidates: int = 1000) -> dict[str, Any]:
    values = np.concatenate([np.asarray(row["score_nez"], dtype=float)[np.asarray(row["channel_mask"], dtype=bool)] for row in validation_records])
    best = None
    for threshold in raw_threshold_candidates(values, max_candidates):
        metrics = _metrics(validation_records, float(threshold))
        key = (metrics["patient_macro_f1"], metrics["patient_balanced_accuracy"], metrics["patient_ez_f1"], -metrics["center_gap_f1"], -abs(float(threshold) - 0.5), -float(threshold))
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return {"threshold": best[1], **best[2], "n_candidates": int(raw_threshold_candidates(values, max_candidates).size), "threshold_source": "outer_train_fixed_validation", "test_labels_used": False}


def apply_raw_probability_threshold(score_nez: np.ndarray, threshold: float) -> dict[str, Any]:
    score = np.asarray(score_nez, dtype=float)
    pred_nez = score >= float(threshold)
    return {"predicted_nez": pred_nez, "predicted_ez": ~pred_nez, "selected_threshold": float(threshold), "decision_rule": FORMAL_DECISION_RULE, "threshold_source": "outer_train_fixed_validation", "true_count_used_for_prediction": False, "center_specific_threshold": False, "analysis_status": "PRIMARY_LEGAL", "formal_prediction": True}


__all__ = ["FORMAL_DECISION_RULE", "TRUE_K_DECISION_RULE", "raw_threshold_candidates", "fit_raw_probability_threshold", "apply_raw_probability_threshold"]
