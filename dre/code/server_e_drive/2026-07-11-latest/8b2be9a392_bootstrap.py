"""Patient-level paired bootstrap utilities."""
from __future__ import annotations

import numpy as np
import pandas as pd


def paired_bootstrap(a: pd.DataFrame, b: pd.DataFrame, *, comparison: str, metric: str = "patient_macro_f1", repeats: int = 2000, seed: int = 42) -> dict:
    paired = a[["subject_id", metric]].merge(b[["subject_id", metric]], on="subject_id", suffixes=("_a", "_b"), validate="one_to_one")
    delta = paired[f"{metric}_a"].to_numpy(float) - paired[f"{metric}_b"].to_numpy(float)
    rng = np.random.default_rng(seed)
    draws = np.asarray([delta[rng.integers(0, len(delta), len(delta))].mean() for _ in range(repeats)])
    return {
        "comparison": comparison, "metric": metric,
        "estimate_a": float(paired[f"{metric}_a"].mean()), "estimate_b": float(paired[f"{metric}_b"].mean()),
        "delta": float(delta.mean()), "ci_low": float(np.quantile(draws, 0.025)), "ci_high": float(np.quantile(draws, 0.975)),
        "probability_delta_gt_zero": float((draws > 0).mean()), "bootstrap_repeats": int(repeats), "n_patients": int(len(delta)), "seed": int(seed),
    }

