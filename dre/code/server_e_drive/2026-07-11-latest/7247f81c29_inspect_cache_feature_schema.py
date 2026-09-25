from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np


STEP4A_REQUIRED_FEATURES = (
    "log_bp_high_gamma",
    "line_length_per_sec",
    "rms",
    "variance",
    "fast_slow_ratio",
    "low_freq_suppression",
    "broadband_electrodecrement",
    "early_high_gamma_slope",
    "early_line_length_slope",
    "onset_latency_high_gamma",
    "onset_latency_line_length",
    "onset_rank_high_gamma",
    "onset_rank_line_length",
)


def inspect_cache_feature_schema(cache_path: Path, required_features: tuple[str, ...] = STEP4A_REQUIRED_FEATURES) -> dict[str, Any]:
    with open(cache_path, "rb") as fin:
        payload = pickle.load(fin)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported cache payload type: {type(payload)!r}")
    run_records = payload.get("run_records", [])
    names = payload.get("window_feature_names")
    if names is None and run_records:
        names = run_records[0].get("sample", {}).get("window_feature_names")
    feature_names = [str(name) for name in names] if names is not None else []
    feature_dim = len(feature_names)
    nonfinite_records = 0
    sample_dim_mismatches = 0
    for record in run_records:
        sample = record.get("sample", {})
        features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
        sample_names = sample.get("window_feature_names", feature_names)
        if features.ndim == 3 and int(features.shape[-1]) != len(sample_names):
            sample_dim_mismatches += 1
        if features.size and not np.isfinite(features).all():
            nonfinite_records += 1
    missing = [name for name in required_features if name not in feature_names]
    return {
        "cache_path": str(cache_path),
        "num_run_records": len(run_records),
        "feature_dim": feature_dim,
        "first_50_feature_names": feature_names[:50],
        "nonfinite_window_feature_records": nonfinite_records,
        "sample_dim_mismatches": sample_dim_mismatches,
        "required_step4a_features_present": not missing,
        "missing_required_step4a_features": missing,
        "feature_names_unique": len(feature_names) == len(set(feature_names)),
        "window_feature_groups": payload.get("window_feature_groups"),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect NeuroEZ-C cache window feature schema.")
    parser.add_argument("cache_path", type=Path)
    return parser


def main() -> None:
    report = inspect_cache_feature_schema(build_parser().parse_args().cache_path)
    for key, value in report.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
