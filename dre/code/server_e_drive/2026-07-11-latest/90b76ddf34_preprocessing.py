from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


@dataclass
class NGBRPreprocessor:
    input_features: list[str]
    retained_features: list[str]
    medians: dict[str, float]
    lower: dict[str, float]
    upper: dict[str, float]
    scaler: StandardScaler
    audit: list[dict]

    @classmethod
    def fit(cls, frame: pd.DataFrame, features: Sequence[str], outer_fold: int) -> "NGBRPreprocessor":
        medians: dict[str, float] = {}; lower: dict[str, float] = {}; upper: dict[str, float] = {}; retained = []; audit = []
        work = pd.DataFrame(index=frame.index)
        for name in features:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float); finite = values[np.isfinite(values)]
            if not finite.size:
                audit.append({"outer_fold": outer_fold, "feature": name, "status": "removed", "reason": "all_missing_outer_train", "train_only_verified": True}); continue
            medians[name] = float(np.median(finite)); imputed = np.where(np.isfinite(values), values, medians[name])
            if float(np.std(imputed)) < 1e-12:
                audit.append({"outer_fold": outer_fold, "feature": name, "status": "removed", "reason": "constant_outer_train", "train_only_verified": True}); continue
            lower[name], upper[name] = map(float, np.quantile(imputed, [.01, .99])); work[name] = np.clip(imputed, lower[name], upper[name]); retained.append(name)
            audit.append({"outer_fold": outer_fold, "feature": name, "status": "retained", "reason": "", "median": medians[name], "winsor_q01": lower[name], "winsor_q99": upper[name], "train_only_verified": True})
        if not retained: raise ValueError("NGBR preprocessing removed every feature")
        scaler = StandardScaler().fit(work[retained].to_numpy(dtype=float))
        return cls(list(features), retained, medians, lower, upper, scaler, audit)

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        work = pd.DataFrame(index=frame.index)
        for name in self.retained_features:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=float); values = np.where(np.isfinite(values), values, self.medians[name]); work[name] = np.clip(values, self.lower[name], self.upper[name])
        return self.scaler.transform(work[self.retained_features].to_numpy(dtype=float))


def hfo_fold_eligible(train: pd.DataFrame) -> tuple[bool, dict]:
    valid = train["n_hfo_valid_seizures"].to_numpy(dtype=float) > 0; y = train["outcome_true"].to_numpy(dtype=int)
    counts = {"hfo_valid_patients": int(valid.sum()), "hfo_valid_success": int((valid & (y == 1)).sum()), "hfo_valid_failure": int((valid & (y == 0)).sum())}
    enabled = counts["hfo_valid_patients"] >= 30 and counts["hfo_valid_success"] >= 10 and counts["hfo_valid_failure"] >= 10
    counts.update({"hfo_enabled_in_fold": enabled, "hfo_status": "HFO_ENABLED_IN_FOLD" if enabled else "HFO_DISABLED_IN_FOLD_INSUFFICIENT_VALID_PATIENTS"})
    return enabled, counts


__all__ = ["NGBRPreprocessor", "hfo_fold_eligible"]
