"""Audit raw_waveform coverage for frozen-FM NeuroEZ-C baselines."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.traditional_baselines.core import infer_center, load_window_cache


def _sfreq(record: Mapping[str, Any], sample: Mapping[str, Any]) -> float:
    value = record.get("sfreq", sample.get("raw_temporal_sfreq", sample.get("sfreq", 0.0)))
    try:
        return float(value)
    except Exception:
        return 0.0


def _raw_record_row(record: Mapping[str, Any], patient_index: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    subject_id = str(record.get("subject_id", ""))
    run_id = str(record.get("run_id", ""))
    sample = record.get("sample") or {}
    patient_meta = patient_index.get(subject_id, {}) or {}
    center = infer_center(subject_id, patient_meta, record, sample)
    raw = sample.get("raw_waveform")
    labels = np.asarray(record.get("labels", []))
    channel_names = list(record.get("channel_names_norm") or record.get("channel_names") or [])
    sfreq = _sfreq(record, sample)
    row: Dict[str, Any] = {
        "subject_id": subject_id,
        "run_id": run_id,
        "center": center,
        "has_raw_waveform": raw is not None,
        "raw_shape": "",
        "n_channels": 0,
        "n_times": 0,
        "sfreq": sfreq,
        "duration_sec": 0.0,
        "labels_len": int(labels.size),
        "channel_names_len": int(len(channel_names)),
        "nan_count": 0,
        "inf_count": 0,
        "all_zero_channel_count": 0,
        "usable_for_fm": False,
        "skip_reason": "",
    }
    if raw is None:
        row["skip_reason"] = "missing raw_waveform"
        return row
    arr = np.asarray(raw)
    row["raw_shape"] = str(tuple(arr.shape))
    if arr.ndim != 2:
        row["skip_reason"] = "raw_waveform is not 2D [n_channels,n_times]"
        return row
    n_channels, n_times = int(arr.shape[0]), int(arr.shape[1])
    row["n_channels"] = n_channels
    row["n_times"] = n_times
    row["duration_sec"] = float(n_times / sfreq) if sfreq > 0 else 0.0
    row["nan_count"] = int(np.isnan(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
    row["inf_count"] = int(np.isinf(arr).sum()) if np.issubdtype(arr.dtype, np.number) else 0
    row["all_zero_channel_count"] = int(np.isclose(arr, 0.0).all(axis=1).sum()) if n_channels else 0

    reasons = []
    if labels.size != n_channels:
        reasons.append("labels length mismatch")
    if len(channel_names) != n_channels:
        reasons.append("channel_names length mismatch")
    if sfreq <= 0:
        reasons.append("invalid sfreq")
    if n_times <= 0 or n_channels <= 0:
        reasons.append("empty raw_waveform")
    if reasons:
        row["skip_reason"] = "; ".join(reasons)
    else:
        row["usable_for_fm"] = True
    return row


def _safe_stats(values: Sequence[float]) -> Dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return {"min": 0.0, "median": 0.0, "max": 0.0}
    return {"min": float(np.min(arr)), "median": float(np.median(arr)), "max": float(np.max(arr))}


def audit_raw_waveform_cache(window_cache_path: str | Path, output_dir: str | Path) -> Dict[str, Any]:
    cache = load_window_cache(window_cache_path)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = [_raw_record_row(record, cache["patient_index"]) for record in cache["run_records"]]
    records_df = pd.DataFrame(rows)
    records_df.to_csv(out / "fm_raw_waveform_records.csv", index=False)

    usable = records_df[records_df["usable_for_fm"].astype(bool)]
    duration_stats = _safe_stats(usable["duration_sec"].astype(float).tolist())
    channel_stats = _safe_stats(usable["n_channels"].astype(float).tolist())
    audit = {
        "input_cache_path": str(window_cache_path),
        "output_dir": str(out),
        "n_records_total": int(len(records_df)),
        "n_records_with_raw_waveform": int(records_df["has_raw_waveform"].astype(bool).sum()),
        "n_records_missing_raw_waveform": int((~records_df["has_raw_waveform"].astype(bool)).sum()),
        "n_records_usable_for_fm": int(usable.shape[0]),
        "n_patients": int(records_df["subject_id"].nunique()),
        "center_counts": records_df["center"].value_counts().sort_index().to_dict(),
        "sfreq_counts": records_df["sfreq"].astype(str).value_counts().sort_index().to_dict(),
        "raw_duration_sec_min": duration_stats["min"],
        "raw_duration_sec_median": duration_stats["median"],
        "raw_duration_sec_max": duration_stats["max"],
        "n_channels_min": channel_stats["min"],
        "n_channels_median": channel_stats["median"],
        "n_channels_max": channel_stats["max"],
        "total_nan_count": int(records_df["nan_count"].sum()),
        "total_inf_count": int(records_df["inf_count"].sum()),
        "total_all_zero_channels": int(records_df["all_zero_channel_count"].sum()),
        "whether_all_records_usable_for_fm": bool(len(records_df) > 0 and usable.shape[0] == len(records_df)),
        "recommended_next_action": "build raw-window manifest" if usable.shape[0] == len(records_df) else "inspect skipped raw records before FM embedding extraction",
    }
    (out / "fm_raw_waveform_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    return audit


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--window_cache_path", required=True)
    parser.add_argument("--output_dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    audit_raw_waveform_cache(args.window_cache_path, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

