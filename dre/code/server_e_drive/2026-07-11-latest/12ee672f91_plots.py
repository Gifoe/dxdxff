from __future__ import annotations

import numpy as np
import pandas as pd


def calibration_table(predictions: pd.DataFrame, bins: int = 10) -> pd.DataFrame:
    rows = []
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    for variant, group in predictions.groupby("variant", sort=True):
        for index in range(int(bins)):
            lower, upper = edges[index], edges[index + 1]
            mask = (group["probability"] >= lower) & ((group["probability"] <= upper) if index == bins - 1 else (group["probability"] < upper))
            selected = group[mask]
            rows.append(
                {
                    "variant": variant,
                    "bin": index,
                    "lower": lower,
                    "upper": upper,
                    "patient_count": int(len(selected)),
                    "mean_probability": float(selected["probability"].mean()) if len(selected) else float("nan"),
                    "observed_rate": float(selected["outcome"].mean()) if len(selected) else float("nan"),
                }
            )
    return pd.DataFrame(rows)


__all__ = ["calibration_table"]
