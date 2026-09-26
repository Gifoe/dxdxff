"""Read-only metadata audit for the trusted NeuroEZ window cache."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


NEEDED = (
    "log_bp_delta",
    "log_bp_theta",
    "log_bp_beta",
    "log_bp_low_gamma",
    "log_bp_high_gamma",
    "rms",
    "variance",
    "line_length_per_sec",
    "spectral_entropy",
)


def describe(value):
    if hasattr(value, "shape"):
        return {"type": type(value).__name__, "shape": list(value.shape), "dtype": str(value.dtype)}
    if isinstance(value, (list, tuple, dict)):
        return {"type": type(value).__name__, "length": len(value)}
    return {"type": type(value).__name__}


def sample_of(record):
    sample = record.get("sample")
    return sample if isinstance(sample, dict) else record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.feature_names.open(newline="", encoding="utf-8-sig") as handle:
        feature_names = [row["feature_name"] for row in csv.DictReader(handle)]
    with args.cohort.open(newline="", encoding="utf-8-sig") as handle:
        cohort_rows = list(csv.DictReader(handle))
    with args.folds.open(newline="", encoding="utf-8-sig") as handle:
        fold_rows = list(csv.DictReader(handle))
    cohort_ids = {str(row["subject_id"]) for row in cohort_rows}
    assert len(cohort_ids) == len(cohort_rows) == 80, "formal cohort is not exactly 80 unique subjects"
    assert all(name in feature_names for name in NEEDED), "nine required descriptors absent"

    with args.cache.open("rb") as handle:
        cache = pickle.load(handle)
    if not isinstance(cache, dict):
        raise TypeError(f"cache type {type(cache).__name__}; expected dict")
    run_records = cache.get("run_records")
    patient_index = cache.get("patient_index")
    if not isinstance(run_records, (list, tuple)) or not isinstance(patient_index, dict):
        raise TypeError("missing run_records list or patient_index dict")
    chosen = [record for record in run_records if str(record.get("subject_id")) in cohort_ids]
    chosen_ids = {str(record.get("subject_id")) for record in chosen}
    first = sample_of(chosen[0])
    features = np.asarray(first["window_features"])
    times = np.asarray(first.get("window_relative_centers_sec"))
    sample_shapes = {key: describe(value) for key, value in first.items()}
    if features.ndim != 3 or features.shape[-1] != len(feature_names):
        raise ValueError(f"feature shape {features.shape} and {len(feature_names)} names disagree")

    fold_summary = defaultdict(Counter)
    for row in fold_rows:
        fold_summary[str(row["fold_idx"])][str(row["partition"]).lower()] += 1
    time_shapes = Counter()
    pre_count = post_count = invalid_time_count = 0
    channels_unmatched = 0
    label_values = set()
    missing_labels = 0
    masked_label_channels = 0
    patients_with_masked_labels = 0
    feature_name_mismatches = 0
    nonfinite_feature_runs = 0
    nonmonotonic_time_runs = 0
    sample_rates = set()
    feature_scales = set()
    for patient_id in cohort_ids:
        meta = patient_index.get(patient_id)
        if meta is None:
            continue
        labels = np.asarray(meta.get("labels", []))
        canonical = list(meta.get("canonical_channels", []))
        if len(labels) != len(canonical):
            missing_labels += 1
        label_mask = np.asarray(meta.get("label_mask", np.ones(len(labels), dtype=bool)), dtype=bool)
        n_masked = int((~label_mask).sum())
        masked_label_channels += n_masked
        patients_with_masked_labels += bool(n_masked)
        label_values.update(np.unique(labels).tolist())
    for record in chosen:
        sample = sample_of(record)
        x = np.asarray(sample["window_features"])
        t = np.asarray(sample.get("window_relative_centers_sec"))
        feature_name_mismatches += list(map(str, sample.get("window_feature_names", []))) != feature_names
        nonfinite_feature_runs += not bool(np.isfinite(x).all())
        sample_rates.add(float(record.get("sfreq", sample.get("raw_temporal_sfreq", float("nan")))))
        feature_scales.update(map(float, sample.get("feature_scales_sec", [])))
        time_shapes[(x.shape[0], t.shape[0] if t.ndim else -1)] += 1
        if t.ndim != 1 or len(t) != len(x) or not np.isfinite(t).all():
            invalid_time_count += 1
        else:
            pre_count += bool(np.any(t < 0))
            post_count += bool(np.any(t >= 0))
            nonmonotonic_time_runs += not bool(np.all(np.diff(t) > 0))
        canonical = set(patient_index[str(record["subject_id"])]["canonical_channels"])
        local = list(sample.get("channel_names_norm", []))
        channels_unmatched += sum(name not in canonical for name in local)

    result = {
        "status": "READ_ONLY_CACHE_METADATA_AUDIT",
        "cache_keys": sorted(map(str, cache.keys())),
        "cache_label_semantics": cache.get("label_semantics"),
        "cache_run_records": len(run_records),
        "cache_patient_index": len(patient_index),
        "formal_cohort_count": len(cohort_ids),
        "formal_cohort_in_patient_index": len(cohort_ids & set(patient_index)),
        "formal_cohort_with_run_records": len(chosen_ids),
        "selected_run_count": len(chosen),
        "fold_partitions": {key: dict(value) for key, value in sorted(fold_summary.items())},
        "feature_dim": len(feature_names),
        "descriptor_indices": {name: feature_names.index(name) for name in NEEDED},
        "first_sample_schema": sample_shapes,
        "first_record_schema": {key: describe(value) for key, value in chosen[0].items()},
        "first_sample_times": {"first": times[:4].tolist(), "last": times[-4:].tolist(), "min": float(np.min(times)), "max": float(np.max(times))} if times.ndim == 1 and len(times) else None,
        "time_shapes": {str(key): count for key, count in time_shapes.items()},
        "selected_runs_with_pre": pre_count,
        "selected_runs_with_post": post_count,
        "invalid_time_runs": invalid_time_count,
        "unmatched_local_channel_mentions": channels_unmatched,
        "masked_label_channels": masked_label_channels,
        "patients_with_masked_labels": patients_with_masked_labels,
        "feature_name_mismatches": feature_name_mismatches,
        "nonfinite_feature_runs": nonfinite_feature_runs,
        "nonmonotonic_time_runs": nonmonotonic_time_runs,
        "sampling_rates": sorted(sample_rates),
        "feature_scales_sec": sorted(feature_scales),
        "patient_metadata_channel_length_mismatches": missing_labels,
        "patient_index_label_values": sorted(label_values),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "first_sample_schema"}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
