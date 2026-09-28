"""Metadata-only alignment audit of the frozen A1 feature and raw caches.

This intentionally never reports patient labels or performance outcomes.
"""

from __future__ import annotations

import argparse
import csv
import gc
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np


def norm(name: object) -> str:
    return "".join(str(name).upper().split()).replace("-", "").replace("_", "")


def metadata(record: dict) -> dict:
    sample = record.get("sample", {})
    if not isinstance(sample, dict):
        sample = {}
    raw = sample.get("raw_waveform", record.get("raw_waveform"))
    raw_shape = tuple(int(x) for x in np.shape(raw)) if raw is not None else None
    feature = sample.get("window_features")
    feature_shape = tuple(int(x) for x in np.shape(feature)) if feature is not None else None
    centers = sample.get("window_relative_centers_sec")
    center_shape = tuple(int(x) for x in np.shape(centers)) if centers is not None else None
    centers_arr = np.asarray(centers, dtype=np.float32).reshape(-1) if centers is not None else np.empty(0)
    sfreq = sample.get("raw_temporal_sfreq", record.get("raw_temporal_sfreq"))
    try:
        sfreq = float(sfreq)
    except (ValueError, TypeError):
        sfreq = None
    onset_keys = (
        "raw_temporal_onset_centered", "raw_window_onset_centered",
        "raw_waveform_onset_centered", "onset_centered", "raw_temporal_zero_is_onset",
    )
    onset_ref_keys = (
        "raw_temporal_reference", "raw_waveform_reference",
        "temporal_reference", "window_reference",
    )
    onset = any(bool(sample.get(k)) for k in onset_keys) or any(
        str(sample.get(k, "")).strip().lower() in
        {"onset", "seizure_onset", "seizure_onset_centered", "ictal_onset"}
        for k in onset_ref_keys
    )
    names = tuple(norm(n) for n in record.get("channel_names_norm", []))
    return {
        "subject_id": str(record.get("subject_id")),
        "run_id": str(record.get("run_id")),
        "sample_id": str(sample.get("sample_id", record.get("run_id"))),
        "names": names,
        "raw_shape": raw_shape,
        "feature_shape": feature_shape,
        "center_shape": center_shape,
        "center_min": float(centers_arr.min()) if centers_arr.size else None,
        "center_max": float(centers_arr.max()) if centers_arr.size else None,
        "center_step": float(np.median(np.diff(centers_arr))) if centers_arr.size > 1 else None,
        "sfreq": sfreq,
        "duration_sec": sample.get("raw_temporal_duration_sec"),
        "valid_samples": sample.get("raw_valid_samples"),
        "valid_start": sample.get("raw_valid_start_sample"),
        "seizure_onset_present": sample.get("seizure_onset_sec") is not None,
        "raw_finite": bool(np.isfinite(raw).all()) if raw is not None else False,
        "onset_verified": bool(onset),
        "raw_fields": sorted(k for k in sample if "raw" in k or "onset" in k),
    }


def load_metadata(path: Path) -> tuple[list[dict], int]:
    with path.open("rb") as handle:
        payload = pickle.load(handle)
    rows = [metadata(r) for r in payload["run_records"]]
    n_index = len(payload.get("patient_index", {}))
    del payload
    gc.collect()
    return rows, n_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    raw, raw_index_count = load_metadata(args.raw_cache)
    feature, feature_index_count = load_metadata(args.feature_cache)
    with args.manifest.open(newline="", encoding="utf-8-sig") as handle:
        manifest_rows = list(csv.DictReader(handle))
    a1_ids = {r["subject_id"] for r in manifest_rows}
    if len(a1_ids) != 80:
        raise RuntimeError(f"Expected exact A1 80 patients, manifest has {len(a1_ids)}")
    feature = [r for r in feature if r["subject_id"] in a1_ids]
    raw_by_key = {}
    for r in raw:
        raw_by_key.setdefault((r["subject_id"], r["run_id"]), []).append(r)
    matched = []
    missing = []
    mismatch = Counter()
    for f in feature:
        key = (f["subject_id"], f["run_id"])
        candidates = raw_by_key.get(key, [])
        if not candidates:
            missing.append(key)
            continue
        exact = [r for r in candidates if r["sample_id"] == f["sample_id"] and r["names"] == f["names"]]
        if len(exact) == 1:
            r = exact[0]
        elif len(candidates) == 1:
            r = candidates[0]
        else:
            mismatch["ambiguous_duplicate_run"] += 1
            continue
        if f["names"] != r["names"]:
            mismatch["channel_order_or_name"] += 1
        if f["sample_id"] != r["sample_id"]:
            mismatch["sample_id"] += 1
        if f["center_shape"] != r["center_shape"]:
            mismatch["window_center_shape"] += 1
        if not r["raw_shape"]:
            mismatch["missing_waveform"] += 1
        elif r["raw_shape"][0] != len(r["names"]):
            mismatch["raw_channel_axis"] += 1
        if not r["sfreq"] or not np.isfinite(r["sfreq"]) or r["sfreq"] <= 0:
            mismatch["invalid_sfreq"] += 1
        if not r["raw_finite"]:
            mismatch["raw_nonfinite"] += 1
        matched.append((f, r))
    subjects_f = {r["subject_id"] for r in feature}
    subjects_r = {r["subject_id"] for r in raw}
    report = {
        "audit_type": "metadata_and_alignment_only_no_outcomes",
        "feature_cache": str(args.feature_cache),
        "raw_cache": str(args.raw_cache),
        "feature_patient_index_count": feature_index_count,
        "raw_patient_index_count": raw_index_count,
        "feature_patients": len(subjects_f),
        "feature_runs": len(feature),
        "feature_channels_sum_by_run": sum(len(r["names"]) for r in feature),
        "raw_patients": len(subjects_r),
        "raw_runs": len(raw),
        "raw_duplicate_subject_run_keys": sum(len(v) > 1 for v in raw_by_key.values()),
        "raw_max_duplicate_multiplicity": max(map(len, raw_by_key.values()), default=0),
        "matched_runs": len(matched),
        "matched_patients": len({f["subject_id"] for f, _ in matched}),
        "missing_runs": len(missing),
        "missing_patients": len({s for s, _ in missing}),
        "mismatch_counts": dict(mismatch),
        "raw_sfreq_counts": dict(Counter(str(r["sfreq"]) for r in raw)),
        "raw_shape_counts": dict(Counter(str(r["raw_shape"]) for r in raw).most_common(20)),
        "feature_shape_counts": dict(Counter(str(r["feature_shape"]) for r in feature).most_common(20)),
        "raw_onset_verified_runs": sum(r["onset_verified"] for r in raw),
        "matched_onset_verified_runs": sum(r["onset_verified"] for _, r in matched),
        "raw_field_sets": dict(Counter(str(r["raw_fields"]) for r in raw)),
        "feature_only_patient_count": len(subjects_f - subjects_r),
        "raw_only_patient_count": len(subjects_r - subjects_f),
        "a1_manifest_patients": len(a1_ids),
        "a1_raw_covered_patients": len(subjects_f & subjects_r),
        "matched_center_range_counts": dict(Counter(str((f["center_min"], f["center_max"], f["center_step"])) for f, _ in matched)),
        "matched_raw_valid_interval_in_bounds": sum(
            r["raw_shape"] is not None and r["valid_start"] is not None and r["valid_samples"] is not None
            and 0 <= int(r["valid_start"]) < int(r["raw_shape"][1])
            and 0 < int(r["valid_samples"]) <= int(r["raw_shape"][1]) - int(r["valid_start"])
            for _, r in matched
        ),
        "matched_seizure_onset_field_present": sum(r["seizure_onset_present"] for _, r in matched),
        "matched_raw_duration_counts": dict(Counter(str(r["duration_sec"]) for _, r in matched)),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
