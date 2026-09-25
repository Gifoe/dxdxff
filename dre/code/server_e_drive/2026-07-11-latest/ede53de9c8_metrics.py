from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


@dataclass(frozen=True)
class MetricBundle:
    values: dict[str, float]
    confusion_matrix: tuple[tuple[int, int], tuple[int, int]]
    undefined_reasons: dict[str, str]


def expected_calibration_error(target: np.ndarray, probability: np.ndarray, bins: int = 10) -> float:
    y = np.asarray(target, dtype=np.int64)
    p = np.asarray(probability, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    total = max(len(y), 1)
    value = 0.0
    for index in range(int(bins)):
        if index == int(bins) - 1:
            mask = (p >= edges[index]) & (p <= edges[index + 1])
        else:
            mask = (p >= edges[index]) & (p < edges[index + 1])
        if not np.any(mask):
            continue
        value += float(np.sum(mask) / total) * abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
    return float(value)


def _safe_ratio(numerator: int, denominator: int, key: str, reasons: dict[str, str]) -> float:
    if denominator == 0:
        reasons[key] = "zero_denominator"
        return float("nan")
    return float(numerator / denominator)


def compute_patient_metrics(
    target: np.ndarray,
    probability: np.ndarray,
    threshold: float | None = None,
    *,
    predicted: np.ndarray | None = None,
) -> MetricBundle:
    y = np.asarray(target, dtype=np.int64).reshape(-1)
    p = np.asarray(probability, dtype=np.float64).reshape(-1)
    if y.shape != p.shape or y.size == 0:
        raise ValueError("target and probability must be non-empty vectors with equal shape.")
    if not np.isfinite(p).all() or not np.isin(y, [0, 1]).all():
        raise ValueError("Metrics require finite probabilities and binary targets.")
    if predicted is None:
        if threshold is None:
            raise ValueError("Metrics require either an explicit threshold or stored predictions.")
        labels = (p >= float(threshold)).astype(np.int64)
    else:
        labels = np.asarray(predicted, dtype=np.int64).reshape(-1)
        if labels.shape != y.shape or not np.isin(labels, [0, 1]).all():
            raise ValueError("Stored predictions must be a binary vector matching target shape.")
    tn, fp, fn, tp = confusion_matrix(y, labels, labels=[0, 1]).ravel().tolist()
    reasons: dict[str, str] = {}
    values: dict[str, float] = {
        "macro_f1": float(f1_score(y, labels, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y, labels)),
        "brier": float(np.mean((p - y) ** 2)),
        "ece": expected_calibration_error(y, p),
        "sensitivity": _safe_ratio(tp, tp + fn, "sensitivity", reasons),
        "specificity": _safe_ratio(tn, tn + fp, "specificity", reasons),
        "ppv": _safe_ratio(tp, tp + fp, "ppv", reasons),
        "npv": _safe_ratio(tn, tn + fn, "npv", reasons),
    }
    if np.unique(y).size < 2:
        values["balanced_accuracy"] = float("nan")
        reasons["balanced_accuracy"] = "single_class_target"
        values["auroc"] = float("nan")
        reasons["auroc"] = "single_class_target"
        values["auprc"] = float("nan")
        reasons["auprc"] = "single_class_target"
    else:
        values["balanced_accuracy"] = float(balanced_accuracy_score(y, labels))
        values["auroc"] = float(roc_auc_score(y, p))
        values["auprc"] = float(average_precision_score(y, p))
    return MetricBundle(values, ((int(tn), int(fp)), (int(fn), int(tp))), reasons)


def metric_bundle_to_dict(bundle: MetricBundle) -> dict[str, Any]:
    return {
        **bundle.values,
        "confusion_matrix": [list(row) for row in bundle.confusion_matrix],
        "undefined_reasons": dict(bundle.undefined_reasons),
    }


__all__ = ["MetricBundle", "compute_patient_metrics", "expected_calibration_error", "metric_bundle_to_dict"]
