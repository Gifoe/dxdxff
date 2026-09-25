from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

S5_8 = [
    "early_high_gamma_slope", "early_line_length_slope",
    "onset_latency_high_gamma", "onset_latency_line_length",
    "onset_rank_high_gamma", "onset_rank_line_length",
    "high_gamma_top20pct_mean", "line_length_top20pct_mean",
]
S5_12 = S5_8 + [
    "hfo80_150_event_rate", "hfo80_150_duration_fraction",
    "hfo80_150_mean_envelope_z", "hfo80_150_max_envelope_z",
]
BASE_20_FALLBACK = ["log_bp_high_gamma", "line_length_per_sec", "rms", "variance"]


def _sample(record: dict) -> dict:
    value = record.get("sample", {})
    return value if isinstance(value, dict) else record


def inspect(cache_path: Path, subjects_path: Path) -> dict[str, object]:
    with cache_path.open("rb") as handle:
        payload = pickle.load(handle)
    records = payload.get("run_records")
    patient_index = payload.get("patient_index")
    if not isinstance(records, list) or not isinstance(patient_index, dict):
        raise RuntimeError("Cache must contain run_records list and patient_index dict.")
    names = [str(value) for value in payload.get("window_feature_names", [])]
    reference = set(pd.read_csv(subjects_path)["subject_id"].astype(str))
    available = set(map(str, patient_index))
    missing_subjects = sorted(reference - available)
    selected = [record for record in records if str(record.get("subject_id")) in reference]

    sums = np.zeros(len(S5_8), dtype=np.float64)
    sums_sq = np.zeros(len(S5_8), dtype=np.float64)
    minima = np.full(len(S5_8), np.inf)
    maxima = np.full(len(S5_8), -np.inf)
    count = 0
    finite = True
    missing_s5 = [name for name in S5_8 if name not in names]
    if not missing_s5:
        indices = [names.index(name) for name in S5_8]
        for record in selected:
            values = np.asarray(_sample(record).get("window_features"), dtype=np.float32)
            if values.ndim != 3 or values.shape[-1] != len(names):
                finite = False
                continue
            block = values[:, :, indices].reshape(-1, len(indices)).astype(np.float64)
            finite = finite and bool(np.isfinite(block).all())
            sums += np.nan_to_num(block).sum(axis=0)
            sums_sq += np.square(np.nan_to_num(block)).sum(axis=0)
            minima = np.minimum(minima, np.nanmin(block, axis=0))
            maxima = np.maximum(maxima, np.nanmax(block, axis=0))
            count += block.shape[0]
    variance = np.maximum(sums_sq / max(count, 1) - np.square(sums / max(count, 1)), 0.0)
    zero_variance = [S5_8[idx] for idx in range(len(S5_8)) if count and minima[idx] == maxima[idx]]
    all_zero = [S5_8[idx] for idx in range(len(S5_8)) if count and minima[idx] == 0.0 and maxima[idx] == 0.0]
    s5_ok = not missing_subjects and len(reference) == 90 and len(selected) == 281 and not missing_s5 and finite and not zero_variance and not all_zero
    missing_exact = [name for name in S5_12 if name not in names]
    missing_fallback = [name for name in BASE_20_FALLBACK if name not in names]
    exact_schema = not missing_subjects and len(reference) == 90 and len(selected) == 281 and not missing_exact
    hfo_names = S5_12[-4:]
    hfo_finite = True
    hfo_zero_variance: list[str] = []
    hfo_all_zero: list[str] = []
    hfo_minima = np.full(len(hfo_names), np.inf)
    hfo_maxima = np.full(len(hfo_names), -np.inf)
    hfo_count = 0
    sampling_rates: list[float] = []
    for record in selected:
        sample = _sample(record)
        sfreq = sample.get("raw_temporal_sfreq", sample.get("sfreq"))
        if sfreq is not None:
            try:
                sampling_rates.append(float(sfreq))
            except (TypeError, ValueError):
                pass
        if not exact_schema:
            continue
        values = np.asarray(sample.get("window_features"), dtype=np.float32)
        if values.ndim != 3 or values.shape[-1] != len(names):
            hfo_finite = False
            continue
        indices = [names.index(name) for name in hfo_names]
        block = values[:, :, indices].reshape(-1, len(indices)).astype(np.float64)
        hfo_finite = hfo_finite and bool(np.isfinite(block).all())
        hfo_minima = np.minimum(hfo_minima, np.nanmin(block, axis=0))
        hfo_maxima = np.maximum(hfo_maxima, np.nanmax(block, axis=0))
        hfo_count += block.shape[0]
    if hfo_count:
        hfo_zero_variance = [hfo_names[i] for i in range(len(hfo_names)) if hfo_minima[i] == hfo_maxima[i]]
        hfo_all_zero = [hfo_names[i] for i in range(len(hfo_names)) if hfo_minima[i] == 0.0 and hfo_maxima[i] == 0.0]
    sampling_supported = bool(sampling_rates) and min(sampling_rates) >= 320.0
    exact_value = exact_schema and hfo_finite and not hfo_zero_variance and not hfo_all_zero and sampling_supported
    exact = exact_schema and exact_value
    fallback = not missing_subjects and not missing_fallback
    return {
        "cache_path": str(cache_path),
        "cache_version": payload.get("cache_version"),
        "n_run_records": len(records),
        "n_reference_run_records": len(selected),
        "n_patients": len(patient_index),
        "n_reference_subjects": len(reference),
        "missing_reference_subjects": missing_subjects,
        "window_feature_names": names,
        "exact_s5_12_schema_compatible": exact_schema,
        "exact_s5_12_value_compatible": exact_value,
        "exact_s5_12_compatible": exact,
        "base20_nonrepro_compatible": fallback,
        "s5_8_compatible": s5_ok,
        "missing_s5_12_features": missing_exact,
        "missing_base20_fallback_features": missing_fallback,
        "missing_s5_8_features": missing_s5,
        "s5_8_all_finite": finite,
        "s5_8_zero_variance_features": zero_variance,
        "s5_8_all_zero_features": all_zero,
        "hfo_all_finite": hfo_finite,
        "hfo_zero_variance_features": hfo_zero_variance,
        "hfo_all_zero_features": hfo_all_zero,
        "hfo_sampling_rates": sorted(set(sampling_rates)),
        "hfo_sampling_supported": sampling_supported,
        "feature_mode": "S5_8_NO_HFO" if s5_ok else "INCOMPATIBLE",
        "conclusion": "s5_8_supported" if s5_ok else "exact12_supported" if exact else "base20_nonrepro_only" if fallback else "incompatible",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--subjects", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--require_exact_s5", action="store_true")
    parser.add_argument("--require_s5_8", action="store_true")
    args = parser.parse_args()
    report = inspect(args.cache, args.subjects)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if args.require_exact_s5 and not report["exact_s5_12_compatible"]:
        raise SystemExit(2)
    if args.require_s5_8 and not report["s5_8_compatible"]:
        raise SystemExit(4)
    if not args.require_exact_s5 and not args.require_s5_8 and not report["base20_nonrepro_compatible"]:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
