"""Validation-only fold-global threshold selection."""
from __future__ import annotations

import numpy as np
import pandas as pd
def threshold_candidates(step: float = 0.005) -> np.ndarray:
    if not np.isclose(step, 0.005, atol=1e-12):
        raise ValueError("The ablation protocol fixes threshold step at 0.005")
    return np.arange(0.0, 1.000001, step)


def patient_equal_macro_f1(frame: pd.DataFrame, threshold: float, *, score_column: str) -> float:
    values = []
    for _, patient in frame.groupby("subject_id", sort=False):
        label = patient.label_nez.to_numpy(int)
        prediction = (patient[score_column].to_numpy(float) >= threshold).astype(int)
        class_f1 = []
        for target in (0, 1):
            tp = int(((label == target) & (prediction == target)).sum())
            fp = int(((label != target) & (prediction == target)).sum())
            fn = int(((label == target) & (prediction != target)).sum())
            denominator = 2 * tp + fp + fn
            class_f1.append(0.0 if denominator == 0 else (2.0 * tp / denominator))
        values.append(0.5 * sum(class_f1))
    return float(np.mean(values))


def select_fold_threshold(validation: pd.DataFrame, *, score_column: str, step: float = 0.005) -> tuple[float, pd.DataFrame]:
    rows = []
    for threshold in threshold_candidates(step):
        rows.append({"threshold": float(threshold), "validation_patient_macro_f1": patient_equal_macro_f1(validation, float(threshold), score_column=score_column)})
    search = pd.DataFrame(rows)
    search["distance_to_point5"] = (search.threshold - 0.5).abs()
    chosen = search.sort_values(["validation_patient_macro_f1", "distance_to_point5", "threshold"], ascending=[False, True, True], kind="mergesort").iloc[0]
    search["selected"] = np.isclose(search.threshold, float(chosen.threshold))
    if int(search.selected.sum()) != 1:
        raise RuntimeError("Exactly one threshold must be selected")
    return float(chosen.threshold), search
