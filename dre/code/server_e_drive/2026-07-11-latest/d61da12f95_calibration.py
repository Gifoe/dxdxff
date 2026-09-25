from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .metrics import compute_patient_metrics


class ProtocolLeakageError(ValueError):
    """Raised when held-out rows are passed to a train-only selector."""


def _assert_inner_oof_roles(roles: Sequence[str]) -> None:
    invalid = sorted({str(role) for role in roles if str(role) != "inner_oof"})
    if invalid:
        raise ProtocolLeakageError(f"Only inner_oof rows may fit this object; found roles: {invalid}")


@dataclass(frozen=True)
class ThresholdSelection:
    threshold: float
    macro_f1: float
    curve: pd.DataFrame


def select_macro_f1_threshold(table: pd.DataFrame) -> ThresholdSelection:
    required = {"role", "outcome", "probability"}
    if not required.issubset(table.columns):
        raise ValueError(f"Threshold table is missing columns: {sorted(required - set(table.columns))}")
    _assert_inner_oof_roles(table["role"].astype(str).tolist())
    target = table["outcome"].to_numpy(dtype=np.int64)
    probability = table["probability"].to_numpy(dtype=np.float64)
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], probability)))
    rows = []
    for threshold in candidates:
        metric = compute_patient_metrics(target, probability, float(threshold)).values["macro_f1"]
        rows.append({"threshold": float(threshold), "macro_f1": float(metric)})
    curve = pd.DataFrame(rows).sort_values("threshold", kind="stable").reset_index(drop=True)
    best = curve.sort_values(["macro_f1", "threshold"], ascending=[False, True], kind="stable").iloc[0]
    tied = curve[np.isclose(curve["macro_f1"], float(best["macro_f1"]))].copy()
    tied["distance_to_half"] = (tied["threshold"] - 0.5).abs()
    chosen = tied.sort_values(["distance_to_half", "threshold"], kind="stable").iloc[0]
    return ThresholdSelection(float(chosen["threshold"]), float(chosen["macro_f1"]), curve)


class PlattCalibrator:
    def __init__(self, regularization_c: float = 1.0) -> None:
        self.regularization_c = float(regularization_c)
        self.estimator: LogisticRegression | None = None
        self.constant_probability: float | None = None

    def fit(self, logits: np.ndarray, targets: np.ndarray, *, roles: Sequence[str] | None = None) -> "PlattCalibrator":
        if roles is not None:
            _assert_inner_oof_roles(roles)
        x = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        y = np.asarray(targets, dtype=np.int64).reshape(-1)
        if x.shape[0] != y.shape[0] or x.shape[0] == 0:
            raise ValueError("Platt fit requires non-empty aligned logits and targets.")
        if np.unique(y).size < 2:
            self.constant_probability = float(np.mean(y))
            self.estimator = None
        else:
            self.estimator = LogisticRegression(C=self.regularization_c, penalty="l2", solver="lbfgs", max_iter=2000)
            self.estimator.fit(x, y)
            self.constant_probability = None
        return self

    def predict(self, logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float64).reshape(-1, 1)
        if self.estimator is not None:
            return self.estimator.predict_proba(x)[:, 1]
        if self.constant_probability is not None:
            return np.full(x.shape[0], self.constant_probability, dtype=np.float64)
        raise RuntimeError("PlattCalibrator must be fitted before predict.")


__all__ = ["PlattCalibrator", "ProtocolLeakageError", "ThresholdSelection", "select_macro_f1_threshold"]
