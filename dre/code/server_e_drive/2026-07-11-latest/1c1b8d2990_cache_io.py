from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import numpy as np

from scripts.traditional_baselines.core import load_window_cache


def load_cache(path: str | Path) -> Dict[str, Any]:
    return load_window_cache(path)


def build_run_lookup(cache: Mapping[str, Any]) -> Dict[str, Mapping[str, Any]]:
    lookup: Dict[str, Mapping[str, Any]] = {}
    for record in cache["run_records"]:
        run_id = str(record.get("run_id", ""))
        subject_id = str(record.get("subject_id", ""))
        if run_id:
            lookup[run_id] = record
        if subject_id and run_id:
            lookup[f"{subject_id}::{run_id}"] = record
    return lookup


def load_raw_window_from_cache(cache_lookup: Mapping[str, Mapping[str, Any]], manifest_row: Mapping[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
    run_id = str(manifest_row["run_id"])
    record = cache_lookup.get(run_id) or cache_lookup.get(f"{manifest_row.get('subject_id')}::{run_id}")
    if record is None:
        raise KeyError(f"run_id not found in cache: {run_id}")
    sample = record.get("sample") or {}
    raw = np.asarray(sample.get("raw_waveform"), dtype=np.float32)
    if raw.ndim != 2:
        raise ValueError("raw_waveform is not 2D")
    channel_idx = int(manifest_row["channel_idx"])
    start = int(manifest_row["window_start_sample"])
    end = int(manifest_row["window_end_sample"])
    if channel_idx < 0 or channel_idx >= raw.shape[0]:
        raise IndexError("channel_idx out of bounds")
    if start < 0 or end > raw.shape[1] or start >= end:
        raise IndexError("window sample range out of bounds")
    return raw[channel_idx, start:end].astype(np.float32), {"sfreq": float(manifest_row["sfreq"])}

