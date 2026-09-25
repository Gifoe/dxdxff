"""Leak-free patient-relative decoder and true-K diagnostic for V3-QBC."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


FORMAL_DECISION_RULE = "fold_validation_global_threshold_on_patient_robust_z"
FORMAL_PREDICTION_SOURCE = "standardized_nez_logit"
TRUE_K_DECISION_RULE = "true_k_topk_diagnostic"


def robust_standardize_patient_logits(
    values: np.ndarray | Iterable[float], *, mad_eps: float = 1e-5, clamp_value: float = 4.0
) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("Patient logits must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise ValueError("Patient logits must be finite")
    median = float(np.median(array))
    scale = 1.4826 * float(np.median(np.abs(array - median)))
    if scale < mad_eps:
        scale = float(np.std(array))
    if scale < mad_eps:
        return np.zeros_like(array)
    return np.clip((array - median) / scale, -clamp_value, clamp_value)


def apply_global_robust_z_threshold(
    final_nez_logit: np.ndarray | Iterable[float], threshold: float
) -> dict[str, Any]:
    z = robust_standardize_patient_logits(final_nez_logit)
    pred_nez = z >= float(threshold)
    return {
        "standardized_nez_logit": z,
        "predicted_nez": pred_nez,
        "predicted_ez": ~pred_nez,
        "selected_threshold": float(threshold),
        "decision_rule": FORMAL_DECISION_RULE,
        "formal_prediction_source": FORMAL_PREDICTION_SOURCE,
        "threshold_source": "outer_train_fixed_validation",
        "true_count_used_for_prediction": False,
        "center_used_as_model_input": False,
        "analysis_status": "PRIMARY_LEGAL",
        "formal_prediction": True,
    }


def _patient_metrics(rows: list[dict[str, Any]], threshold: float) -> dict[str, float]:
    macro_f1 = []
    balanced = []
    ez_f1 = []
    by_center: dict[str, list[float]] = {}
    for row in rows:
        y_nez = np.asarray(row["label_nez"], dtype=np.int64)
        pred = apply_global_robust_z_threshold(row["final_nez_logit"], threshold)["predicted_nez"].astype(int)
        patient_f1 = float(f1_score(y_nez, pred, labels=[0, 1], average="macro", zero_division=0))
        macro_f1.append(patient_f1)
        balanced.append(float(balanced_accuracy_score(y_nez, pred)))
        ez_f1.append(float(f1_score(y_nez, pred, labels=[0], average="macro", zero_division=0)))
        by_center.setdefault(str(row.get("center", "unknown")).lower(), []).append(patient_f1)
    center_means = [float(np.mean(values)) for values in by_center.values() if values]
    return {
        "patient_macro_f1": float(np.mean(macro_f1)),
        "patient_balanced_accuracy": float(np.mean(balanced)),
        "patient_ez_f1": float(np.mean(ez_f1)),
        "center_gap_f1": float(max(center_means) - min(center_means)) if center_means else 0.0,
    }


def _threshold_candidates(rows: list[dict[str, Any]], max_candidates: int) -> np.ndarray:
    values = np.concatenate(
        [robust_standardize_patient_logits(row["final_nez_logit"]) for row in rows]
    )
    unique = np.unique(values)
    if unique.size > max_candidates:
        unique = np.unique(np.quantile(unique, np.linspace(0.0, 1.0, max_candidates)))
    if unique.size <= 1:
        midpoints = unique
    else:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.unique(np.concatenate((np.asarray([-4.0, 4.0]), midpoints)))


def fit_global_robust_z_threshold(
    validation_patients: list[dict[str, Any]], *, max_candidates: int = 2000
) -> dict[str, Any]:
    if not validation_patients:
        raise ValueError("Cannot fit a threshold without validation patients")
    candidates = _threshold_candidates(validation_patients, max_candidates=max_candidates)
    best: tuple[tuple[float, ...], float, dict[str, float]] | None = None
    for threshold in candidates:
        metrics = _patient_metrics(validation_patients, float(threshold))
        key = (
            metrics["patient_macro_f1"],
            metrics["patient_balanced_accuracy"],
            metrics["patient_ez_f1"],
            -metrics["center_gap_f1"],
            -abs(float(threshold)),
            -float(threshold),
        )
        if best is None or key > best[0]:
            best = (key, float(threshold), metrics)
    assert best is not None
    return {
        "threshold": best[1],
        **best[2],
        "n_candidates": int(candidates.size),
        "threshold_source": "outer_train_fixed_validation",
        "test_labels_used": False,
    }


def true_k_diagnostic_prediction(
    final_nez_logit: np.ndarray | Iterable[float], true_ez_count: int
) -> dict[str, Any]:
    logits = np.asarray(final_nez_logit, dtype=np.float64)
    if logits.ndim != 1 or not np.isfinite(logits).all():
        raise ValueError("final_nez_logit must be a finite one-dimensional array")
    k = int(true_ez_count)
    if k < 0 or k > logits.size:
        raise ValueError(f"true_ez_count={k} is outside [0,{logits.size}]")
    order = np.argsort(logits, kind="stable")  # smallest NEZ logits are most EZ-like
    pred_ez = np.zeros(logits.size, dtype=bool)
    pred_ez[order[:k]] = True
    return {
        "predicted_ez": pred_ez,
        "predicted_nez": ~pred_ez,
        "decision_rule": TRUE_K_DECISION_RULE,
        "formal_prediction_source": "ez_rank_score",
        "threshold_source": "true_count",
        "true_count_used_for_prediction": True,
        "analysis_status": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
        "formal_prediction": False,
    }


__all__ = [
    "FORMAL_DECISION_RULE",
    "FORMAL_PREDICTION_SOURCE",
    "TRUE_K_DECISION_RULE",
    "robust_standardize_patient_logits",
    "fit_global_robust_z_threshold",
    "apply_global_robust_z_threshold",
    "true_k_diagnostic_prediction",
]
