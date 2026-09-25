from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import pandas as pd


S5_12 = [
    "early_high_gamma_slope", "early_line_length_slope",
    "onset_latency_high_gamma", "onset_latency_line_length",
    "onset_rank_high_gamma", "onset_rank_line_length",
    "high_gamma_top20pct_mean", "line_length_top20pct_mean",
    "hfo80_150_event_rate", "hfo80_150_duration_fraction",
    "hfo80_150_mean_envelope_z", "hfo80_150_max_envelope_z",
]
BASE_20_FALLBACK = ["log_bp_high_gamma", "line_length_per_sec", "rms", "variance"]


def inspect(cache_path: Path, subjects_path: Path) -> dict[str, object]:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError("Cache must be a dict payload.")
    records = payload.get("run_records")
    patient_index = payload.get("patient_index")
    if not isinstance(records, list) or not isinstance(patient_index, dict):
        raise RuntimeError("Cache must contain run_records list and patient_index dict.")
    feature_names = [str(value) for value in payload.get("window_feature_names", [])]
    required_subjects = set(pd.read_csv(subjects_path)["subject_id"].astype(str))
    available_subjects = set(str(value) for value in patient_index)
    missing_subjects = sorted(required_subjects - available_subjects)
    missing_s5 = [name for name in S5_12 if name not in feature_names]
    missing_fallback = [name for name in BASE_20_FALLBACK if name not in feature_names]
    exact = not missing_subjects and not missing_s5
    fallback = not missing_subjects and not missing_fallback
    return {
        "cache_path": str(cache_path),
        "cache_version": payload.get("cache_version"),
        "n_run_records": len(records),
        "n_patients": len(patient_index),
        "n_reference_subjects": len(required_subjects),
        "missing_reference_subjects": missing_subjects,
        "window_feature_names": feature_names,
        "exact_s5_12_compatible": exact,
        "base20_nonrepro_compatible": fallback,
        "missing_s5_12_features": missing_s5,
        "missing_base20_fallback_features": missing_fallback,
        "conclusion": "exact_reproduction_supported" if exact else "nonrepro_base20_only" if fallback else "incompatible",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require_exact_s5", action="store_true")
    args = parser.parse_args()
    report = inspect(args.cache, args.subjects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_exact_s5 and not report["exact_s5_12_compatible"]:
        raise SystemExit(2)
    if not report["base20_nonrepro_compatible"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
