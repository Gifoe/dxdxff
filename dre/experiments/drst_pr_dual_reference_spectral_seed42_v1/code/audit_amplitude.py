"""Metadata and amplitude-scale audit; never indexes labels or outcomes."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def center(row: dict) -> str:
    source = str(row.get("source_center", row.get("center", ""))).lower()
    if "fudan" in source:
        return "Fudan"
    if "lzu" in source:
        return "LZU"
    if "hup" in source:
        return "HUP"
    prefix = str(row["subject_id"]).split(":", 1)[0].lower()
    if prefix == "hup":
        return "HUP"
    if prefix == "lzu":
        return "LZU"
    if prefix == "fudan":
        return "Fudan"
    if prefix in {"pediatric", "ped"}:
        return "pediatric (Fudan identity unverified)"
    return "multi-site"


def quantiles(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=np.float64)
    return {"n": int(len(arr)), "q05": float(np.quantile(arr, .05)),
            "median": float(np.median(arr)), "q95": float(np.quantile(arr, .95))}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    with a.raw_cache.open("rb") as stream:
        payload = pickle.load(stream)
    groups = defaultdict(list)
    patients = defaultdict(set)
    runs = Counter()
    metadata_fields = Counter()
    source_center_fields = Counter()
    subject_prefixes = Counter()
    waveforms = 0
    finite = 0
    sample_rates = Counter()
    for row in payload["run_records"]:
        sample = row.get("sample") or {}
        if not isinstance(sample, dict):
            continue
        raw = sample.get("raw_waveform", row.get("raw_waveform"))
        if raw is None:
            continue
        arr = np.asarray(raw)
        if arr.ndim != 2:
            raise RuntimeError("Invalid raw waveform shape")
        c = center(row)
        source_center_fields[str(row.get("source_center", row.get("center", "MISSING")))] += 1
        subject_prefixes[str(row["subject_id"]).split(":", 1)[0].lower()] += 1
        patients[c].add(str(row["subject_id"]))
        runs[c] += 1
        waveforms += 1
        finite += bool(np.isfinite(arr).all())
        sample_rates[str(sample.get("raw_temporal_sfreq"))] += 1
        for key in set(sample) | set(row):
            if any(token in key.lower() for token in ("unit", "scale", "refer", "gain", "amplitude")):
                metadata_fields[key] += 1
        # Fixed deterministic subset per record; full 60 s of each sampled channel.
        choose = np.linspace(0, arr.shape[0] - 1, min(8, arr.shape[0]), dtype=int)
        for j in choose:
            x = np.asarray(arr[j], dtype=np.float64)
            if not np.isfinite(x).all():
                raise RuntimeError("Nonfinite raw waveform")
            mad = 1.4826 * np.median(np.abs(x - np.median(x)))
            if mad > 0:
                groups[c].append(float(mad))
    if waveforms != finite:
        raise RuntimeError("Raw cache contains nonfinite waveforms")
    per_center = {name: {"subjects": len(patients[name]), "records": runs[name],
                         "sampled_full_waveform_channel_MAD": quantiles(groups[name])}
                  for name in sorted(runs)}
    medians = [row["sampled_full_waveform_channel_MAD"]["median"] for row in per_center.values()]
    ratio = max(medians) / min(medians) if medians and min(medians) > 0 else None
    report = {"audit_scope": "all raw-cache records, label/outcome-free",
              "raw_cache": str(a.raw_cache), "records_with_waveform": waveforms,
              "all_waveforms_finite": waveforms == finite,
              "sample_rate_counts": dict(sample_rates), "centers": per_center,
              "ratio_largest_to_smallest_center_median_MAD": ratio,
              "unit_scale_reference_metadata_fields": dict(metadata_fields),
              "source_center_field_counts": dict(source_center_fields),
              "subject_id_prefix_record_counts": dict(subject_prefixes),
              "common_physical_unit_verified": False,
              "per_record_scaling_at_cache_creation_verified_absent": False,
              "reference_or_rereference_provenance_verified": False,
              "preprocessing_choice": "per-patient-seizure-channel 60s fixed 1.4826*MAD scale",
              "choice_reason": "Common physical units and recording reference cannot be verified from cache metadata; no target outcomes used.",
              "no_label_or_performance_field_indexed": True}
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"centers": per_center, "ratio": ratio,
                      "choice": report["preprocessing_choice"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
