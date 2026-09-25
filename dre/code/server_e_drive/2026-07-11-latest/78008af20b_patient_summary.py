from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd

from outcome_hifos.dataset import OutcomePatientExample


def _iqr(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.nanpercentile(values, 75, axis=axis) - np.nanpercentile(values, 25, axis=axis)


def _distribution(values: np.ndarray, axis: int = 0) -> list[np.ndarray]:
    return [
        np.nanmean(values, axis=axis),
        np.nanstd(values, axis=axis),
        np.nanmedian(values, axis=axis),
        _iqr(values, axis=axis),
        np.nanmin(values, axis=axis),
        np.nanmax(values, axis=axis),
    ]


def build_patient_summary_matrix(examples: Sequence[OutcomePatientExample]) -> tuple[pd.DataFrame, dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    feature_names: list[str] | None = None
    patient_stat_names = ("mean", "std", "median", "iqr", "min", "max")
    seizure_stat_names = ("mean", "std", "median", "iqr", "min", "max")
    for example in sorted(examples, key=lambda item: item.subject_id):
        seizure_summaries = []
        for run, present in zip(example.model_input["feature_runs"], example.model_input["seizure_channel_mask"]):
            array = np.asarray(run, dtype=np.float64)
            mask = np.asarray(present, dtype=bool)
            valid = array[:, mask, :].reshape(-1, array.shape[-1])
            valid = np.where(np.isfinite(valid), valid, np.nan)
            seizure_summaries.append(np.concatenate(_distribution(valid, axis=0)))
        across = np.stack(seizure_summaries, axis=0)
        vector = np.concatenate(_distribution(across, axis=0))
        if feature_names is None:
            dimension = int(np.asarray(example.model_input["feature_runs"][0]).shape[-1])
            seizure_names = [f"f{index:03d}__within_seizure_{stat}" for stat in seizure_stat_names for index in range(dimension)]
            feature_names = [f"{name}__across_seizure_{stat}" for stat in patient_stat_names for name in seizure_names]
        row = {"subject_id": example.subject_id, "center": example.center, "target": int(example.target)}
        row.update({name: float(value) for name, value in zip(feature_names, vector)})
        rows.append(row)
    return pd.DataFrame(rows), {
        "feature_names": feature_names or [],
        "within_seizure_statistics": list(seizure_stat_names),
        "across_seizure_statistics": list(patient_stat_names),
        "count_features_in_primary": False,
        "label_blind": True,
    }


__all__ = ["build_patient_summary_matrix"]
