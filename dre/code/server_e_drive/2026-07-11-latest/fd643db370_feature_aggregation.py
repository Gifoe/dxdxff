from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd


def _iqr(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.nanpercentile(values, 75, axis=axis) - np.nanpercentile(values, 25, axis=axis)


def _stats(values: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "median": np.nanmedian(values, axis=0),
        "iqr": _iqr(values, axis=0),
        "mean": np.nanmean(values, axis=0),
        "std": np.nanstd(values, axis=0),
    }


def build_channel_feature_table(records: Sequence[Mapping[str, Any]]) -> tuple[pd.DataFrame, dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, np.ndarray]]] = defaultdict(list)
    feature_names: list[str] | None = None
    for record in records:
        features = np.asarray(record["features"], dtype=np.float64)
        channels = [str(value) for value in record["channel_names"]]
        labels = np.asarray(record["labels_nez"], dtype=np.int64)
        current_names = [str(value) for value in record["feature_names"]]
        if features.ndim != 3 or features.shape[1] != len(channels) or features.shape[2] != len(current_names):
            raise ValueError("Task 1 feature record must be [window, channel, feature] with aligned names.")
        if labels.shape != (len(channels),) or not set(labels.tolist()) <= {0, 1}:
            raise ValueError("Task 1 labels_nez must be one binary label per channel (NEZ=1, EZ=0).")
        if feature_names is None:
            feature_names = current_names
        elif feature_names != current_names:
            raise ValueError("Task 1 feature-name order differs across runs.")
        for channel_index, channel in enumerate(channels):
            window_values = features[:, channel_index, :]
            finite_rows = np.isfinite(window_values).all(axis=1)
            valid = window_values[finite_rows]
            if valid.size == 0:
                continue
            seizure = _stats(valid)
            seizure["valid_window_count"] = np.asarray([finite_rows.sum()], dtype=np.float64)
            key = (str(record["subject_id"]), str(record.get("center", "unknown")), channel, int(labels[channel_index]))
            grouped[key].append(seizure)

    names = feature_names or []
    rows: list[dict[str, Any]] = []
    for (subject, center, channel, label), seizures in sorted(grouped.items()):
        row: dict[str, Any] = {
            "subject_id": subject,
            "center": center,
            "channel_name": channel,
            "label_nez": label,
            "clinical_true_nez": label,
            "clinical_true_ez": 1 - label,
            "valid_seizure_count": len(seizures),
            "valid_window_count": int(sum(int(item["valid_window_count"][0]) for item in seizures)),
        }
        for stat in ("median", "iqr", "mean", "std"):
            across = np.stack([item[stat] for item in seizures], axis=0)
            across_stats = _stats(across)
            for feature_index, feature_name in enumerate(names):
                for across_name, values in across_stats.items():
                    row[f"{feature_name}__seizure_{stat}__{across_name}"] = float(values[feature_index])
        rows.append(row)
    table = pd.DataFrame(rows)
    feature_columns = [column for column in table if "__seizure_" in column]
    values = table[feature_columns].to_numpy(dtype=float) if feature_columns else np.empty((len(table), 0))
    manifest = {
        "raw_feature_names": names,
        "window_aggregation": ["median", "IQR", "mean", "std", "valid_window_count"],
        "seizure_aggregation": ["median", "IQR", "mean", "std", "valid_seizure_count"],
        "feature_names": feature_columns,
        "final_dimension": len(feature_columns),
        "finite_ratio": float(np.isfinite(values).mean()) if values.size else 1.0,
        "missing_ratio": float(np.isnan(values).mean()) if values.size else 0.0,
        "label_semantics": {"NEZ": 1, "EZ": 0},
    }
    return table, manifest


__all__ = ["build_channel_feature_table"]
