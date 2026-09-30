#!/usr/bin/env python3
"""TRAIN-only runtime audit for the frozen-score D-MIL implementation."""

from __future__ import annotations

import argparse
import json
import time
import timeit
from pathlib import Path

import numpy as np
import pandas as pd

from dmil_core import FEATURES, logit, sigmoid
from run_train_oof import load_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-cache", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--parameters", type=Path, required=True)
    parser.add_argument("--sign-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.perf_counter()
    rows, cache_audit = load_cache(args.train_cache)
    feature_seconds = time.perf_counter() - started

    predictions = pd.read_csv(args.predictions)
    baseline = predictions[predictions.model == "V0_MEAN"]
    sign = pd.read_csv(args.sign_audit)
    full = sign[sign.model == "V4_FULL_DMIL"].set_index("coefficient")
    beta = np.asarray(
        [full.loc["beta_T", "median"], full.loc["beta_Q", "median"], full.loc["beta_S", "median"]],
        dtype=np.float64,
    )
    base = logit(baseline.mean_score.to_numpy(dtype=np.float64))
    features = baseline.loc[:, FEATURES].to_numpy(dtype=np.float64)

    def apply_adapter() -> None:
        sigmoid(base + features @ beta)

    timings = timeit.repeat(apply_adapter, number=100, repeat=7)
    formula_seconds = float(np.median(timings) / 100.0)
    parameters = pd.read_csv(args.parameters)
    fit_seconds = parameters.groupby("model").fit_seconds.sum().to_dict()
    payload = {
        "scope": "official TRAIN only; no TEST access",
        "frozen_cnn_parameters": 11302241,
        "dmil_trainable_parameters": 3,
        "train_cache_files": int(cache_audit["files"]),
        "labeled_channel_units": int(len(baseline)),
        "feature_aggregation_and_cache_load_seconds": feature_seconds,
        "fold_fit_threshold_score_seconds_by_variant": {key: float(value) for key, value in fit_seconds.items()},
        "fold_fit_threshold_score_seconds_all_variants": float(parameters.fit_seconds.sum()),
        "adapter_formula_seconds_per_full_train_pass_median": formula_seconds,
        "adapter_formula_microseconds_per_channel": formula_seconds / len(baseline) * 1e6,
        "benchmark_repeats": 7,
        "loops_per_repeat": 100,
        "adds_waveform_inference_cost": False,
        "notes": "Feature timing includes reading 296 frozen NPZ files. Formula timing excludes score-distribution feature extraction and applies the three-scalar correction to all labeled TRAIN units.",
    }
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
