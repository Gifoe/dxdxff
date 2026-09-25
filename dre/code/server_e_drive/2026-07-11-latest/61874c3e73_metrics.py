from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


FIXED_DECISION_THRESHOLD = 0.5
FIXED_THRESHOLD_SOURCE = "fixed_predefined_0_5"


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    target, probability = np.asarray(target, dtype=float), np.asarray(probability, dtype=float)
    edges, value = np.linspace(0.0, 1.0, bins + 1), 0.0
    for index in range(bins):
        mask = (probability >= edges[index]) & (probability <= edges[index + 1] if index == bins - 1 else probability < edges[index + 1])
        if mask.any():
            value += mask.mean() * abs(probability[mask].mean() - target[mask].mean())
    return float(value)


def compute_metrics(
    target_success: Sequence[int],
    probability_success: Sequence[float],
    threshold: float = FIXED_DECISION_THRESHOLD,
    *,
    prediction_success: Sequence[int] | None = None,
) -> dict[str, float | int]:
    y = np.asarray(target_success, dtype=np.int64)
    p = np.asarray(probability_success, dtype=np.float64)
    pred = np.asarray(prediction_success, dtype=np.int64) if prediction_success is not None else (p >= float(threshold)).astype(np.int64)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    two_class = np.unique(y).size == 2
    y_failure, p_failure, pred_failure = 1 - y, 1.0 - p, 1 - pred
    return {
        "success_auprc": float(average_precision_score(y, p)) if two_class else float("nan"),
        "failure_auprc": float(average_precision_score(y_failure, p_failure)) if two_class else float("nan"),
        "auroc": float(roc_auc_score(y, p)) if two_class else float("nan"),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if two_class else float("nan"),
        "accuracy": float(accuracy_score(y, pred)),
        "success_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "success_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "failure_precision": float(precision_score(y_failure, pred_failure, pos_label=1, zero_division=0)),
        "failure_recall": float(recall_score(y_failure, pred_failure, pos_label=1, zero_division=0)),
        "specificity_for_failure": float(tn / max(tn + fp, 1)),
        "brier": float(brier_score_loss(y, p)),
        "ece": expected_calibration_error(y, p),
        "true_failure_pred_failure": int(tn),
        "true_failure_pred_success": int(fp),
        "true_success_pred_failure": int(fn),
        "true_success_pred_success": int(tp),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def bootstrap_metrics(
    target_success: Sequence[int],
    probability_success: Sequence[float],
    threshold: float = FIXED_DECISION_THRESHOLD,
    *,
    prediction_success: Sequence[int] | None = None,
    repeats: int = 2000,
    seed: int = 42,
) -> tuple[pd.DataFrame, int]:
    if float(threshold) != FIXED_DECISION_THRESHOLD:
        raise ValueError("Bootstrap protocol requires the predefined 0.5 threshold")
    y, p = np.asarray(target_success, dtype=np.int64), np.asarray(probability_success, dtype=np.float64)
    stored = np.asarray(prediction_success, dtype=np.int64) if prediction_success is not None else (p >= FIXED_DECISION_THRESHOLD).astype(np.int64)
    rng, rows, skipped = np.random.default_rng(seed), [], 0
    for repeat in range(int(repeats)):
        indices = rng.integers(0, len(y), len(y))
        if np.unique(y[indices]).size < 2:
            skipped += 1
            continue
        rows.append({
            "repeat": repeat,
            "success_label": 1,
            "failure_label": 0,
            "decision_threshold": FIXED_DECISION_THRESHOLD,
            "threshold_source": FIXED_THRESHOLD_SOURCE,
            "inner_cv": False,
            **compute_metrics(y[indices], p[indices], prediction_success=stored[indices]),
        })
    return pd.DataFrame(rows), skipped


__all__ = ["FIXED_DECISION_THRESHOLD", "FIXED_THRESHOLD_SOURCE", "bootstrap_metrics", "compute_metrics", "expected_calibration_error"]
