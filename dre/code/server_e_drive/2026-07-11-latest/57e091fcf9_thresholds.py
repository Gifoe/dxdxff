from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, f1_score


@dataclass(frozen=True)
class Task1Threshold:
    threshold: float
    patient_macro_f1: float
    patient_ez_f1: float
    balanced_accuracy: float
    source: str = "inner_oof_patient_macro_f1"


def _patient_metrics(table: pd.DataFrame, threshold: float) -> tuple[float, float, float]:
    patient_macro = []
    patient_ez = []
    all_true: list[int] = []
    all_pred: list[int] = []
    for _, group in table.groupby("subject_id", sort=False):
        true = group["label_nez"].to_numpy(dtype=int)
        pred = (group["score_nez_probability"].to_numpy(dtype=float) >= threshold).astype(int)
        patient_macro.append(f1_score(true, pred, average="macro", labels=[0, 1], zero_division=0))
        patient_ez.append(f1_score(true == 0, pred == 0, zero_division=0))
        all_true.extend(true.tolist())
        all_pred.extend(pred.tolist())
    balanced = balanced_accuracy_score(all_true, all_pred) if len(set(all_true)) == 2 else float("nan")
    return float(np.mean(patient_macro)), float(np.mean(patient_ez)), float(balanced)


def select_patient_macro_threshold(table: pd.DataFrame) -> Task1Threshold:
    required = {"subject_id", "label_nez", "score_nez_probability"}
    if not required.issubset(table):
        raise ValueError(f"Threshold table missing columns: {sorted(required - set(table))}")
    candidates = []
    for threshold in np.round(np.arange(0.05, 0.951, 0.01), 2):
        macro, ez, balanced = _patient_metrics(table, float(threshold))
        candidates.append((macro, ez, np.nan_to_num(balanced, nan=-1.0), -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    best = max(candidates)
    return Task1Threshold(best[-1], best[0], best[1], best[2])


__all__ = ["Task1Threshold", "select_patient_macro_threshold"]
