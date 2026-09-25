"""Frozen and ablation-only fusion operators."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .score_semantics import as_probability


def probability_fusion(p2, v3, *, v3_weight: float) -> np.ndarray:
    if not 0.0 <= float(v3_weight) <= 1.0:
        raise ValueError("v3_weight must be in [0, 1]")
    p2_array = as_probability(p2, name="p2_score_nez")
    v3_array = as_probability(v3, name="v3_score_nez")
    if p2_array.shape != v3_array.shape:
        raise ValueError("P2 and V3 score shapes differ")
    return (1.0 - float(v3_weight)) * p2_array + float(v3_weight) * v3_array


def logit_fusion(p2, v3, *, v3_weight: float, eps: float = 1e-6) -> np.ndarray:
    p2_array = np.clip(as_probability(p2, name="p2_score_nez"), eps, 1.0 - eps)
    v3_array = np.clip(as_probability(v3, name="v3_score_nez"), eps, 1.0 - eps)
    z = (1.0 - v3_weight) * np.log(p2_array / (1.0 - p2_array)) + v3_weight * np.log(v3_array / (1.0 - v3_array))
    return 1.0 / (1.0 + np.exp(-z))


def patient_rank_fusion(frame: pd.DataFrame, *, v3_weight: float) -> np.ndarray:
    output = pd.Series(index=frame.index, dtype=float)
    for _, patient in frame.groupby("subject_id", sort=False):
        n = len(patient)
        if n == 1:
            p2_rank = v3_rank = np.asarray([0.5])
        else:
            p2_rank = (rankdata(patient.p2_score_nez, method="average") - 1.0) / (n - 1.0)
            v3_rank = (rankdata(patient.v3_score_nez, method="average") - 1.0) / (n - 1.0)
        output.loc[patient.index] = (1.0 - v3_weight) * p2_rank + v3_weight * v3_rank
    return output.to_numpy(float)


def stable_shuffle_seed(*parts: object) -> int:
    import hashlib

    text = "|".join(str(part) for part in parts)
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "little")


def shuffle_v3_within_patient(frame: pd.DataFrame, *, experiment: str, fold: int, partition: str, repeat: int, seed: int) -> tuple[pd.DataFrame, dict]:
    result = frame.copy()
    changed = 0
    eligible = 0
    for subject, patient in frame.groupby("subject_id", sort=False):
        before = patient.v3_score_nez.to_numpy(float)
        rng = np.random.default_rng(stable_shuffle_seed(experiment, fold, partition, subject, repeat, seed))
        after = before[rng.permutation(len(before))]
        if len(before) > 1:
            eligible += 1
            changed += int(not np.array_equal(before, after))
        if not np.array_equal(np.sort(before), np.sort(after)):
            raise RuntimeError("Within-patient V3 shuffle changed the score multiset")
        result.loc[patient.index, "v3_score_nez"] = after
    audit = {"eligible_multi_channel_patients": eligible, "changed_order_patients": changed}
    return result, audit

