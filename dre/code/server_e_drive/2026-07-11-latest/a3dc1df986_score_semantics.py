"""Canonical NEZ=1/EZ=0 probability semantics."""
from __future__ import annotations

import numpy as np
import pandas as pd


def as_probability(values, *, name: str = "score") -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all() or np.any((array < 0.0) | (array > 1.0)):
        raise ValueError(f"{name} must be finite and in [0, 1]")
    return array


def sigmoid(values) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if not np.isfinite(array).all():
        raise ValueError("logits must be finite")
    return 1.0 / (1.0 + np.exp(-array))


def canonical_probability_columns(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result["score_nez_probability"] = as_probability(result["score_nez"], name="score_nez")
    result["score_ez_probability"] = 1.0 - result["score_nez_probability"]
    if not (result["label_nez"].astype(int) + result["label_ez"].astype(int) == 1).all():
        raise ValueError("Labels must obey NEZ=1 and EZ=0 exactly")
    return result

