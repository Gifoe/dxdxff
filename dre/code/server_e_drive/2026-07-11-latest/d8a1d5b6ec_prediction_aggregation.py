from __future__ import annotations

import numpy as np
import pandas as pd


def _iqr(values: pd.Series) -> float:
    return float(values.quantile(0.75) - values.quantile(0.25))


def aggregate_task1_window_probabilities(rows: pd.DataFrame) -> pd.DataFrame:
    required = {"subject_id", "center", "seizure_id", "channel_name", "window_id", "label_nez", "score_nez_probability"}
    if not required.issubset(rows):
        raise ValueError(f"Window prediction rows missing columns: {sorted(required - set(rows))}")
    seizure = (
        rows.groupby(["subject_id", "center", "seizure_id", "channel_name", "label_nez"], as_index=False)
        .agg(
            seizure_probability=("score_nez_probability", "median"),
            seizure_probability_mean=("score_nez_probability", "mean"),
            seizure_probability_std=("score_nez_probability", "std"),
            seizure_probability_iqr=("score_nez_probability", _iqr),
            valid_windows=("window_id", "nunique"),
        )
    )
    channel = (
        seizure.groupby(["subject_id", "center", "channel_name", "label_nez"], as_index=False)
        .agg(
            score_nez_probability=("seizure_probability", "median"),
            score_nez_probability_mean=("seizure_probability", "mean"),
            score_nez_probability_std=("seizure_probability", "std"),
            score_nez_probability_iqr=("seizure_probability", _iqr),
            valid_window_count=("valid_windows", "sum"),
            valid_seizure_count=("seizure_id", "nunique"),
        )
    )
    channel["score_ez_probability"] = 1.0 - channel["score_nez_probability"]
    return channel


__all__ = ["aggregate_task1_window_probabilities"]
