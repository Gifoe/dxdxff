from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data_sources import BidsRun
from utils import norm_text, safe_float


def parse_all_seizure_intervals(events_path: Path | None) -> list[tuple[float, float]]:
    if events_path is None or not events_path.exists():
        return []
    try:
        df = pd.read_csv(events_path, sep="\t")
    except Exception:
        return []
    if "trial_type" not in df.columns or "onset" not in df.columns:
        return []
    onsets: list[float] = []
    offsets: list[float] = []
    for _, row in df.iterrows():
        onset = safe_float(row.get("onset"))
        if onset is None:
            continue
        trial_type = norm_text(row.get("trial_type", ""))
        if trial_type == "sz onset":
            onsets.append(onset)
        elif trial_type == "sz offset":
            offsets.append(onset)
    intervals: list[tuple[float, float]] = []
    used_offsets: set[int] = set()
    for onset in sorted(onsets):
        for idx, offset in enumerate(offsets):
            if idx not in used_offsets and offset > onset:
                used_offsets.add(idx)
                intervals.append((float(onset), float(offset)))
                break
    return intervals


def interval_complement(
    intervals: Sequence[tuple[float, float]],
    *,
    duration_sec: float,
    pre_buffer_sec: float,
    post_buffer_sec: float,
) -> list[tuple[float, float]]:
    excluded = []
    for onset, offset in intervals:
        start = max(0.0, float(onset) - float(pre_buffer_sec))
        end = min(float(duration_sec), float(offset) + float(post_buffer_sec))
        if end > start:
            excluded.append((start, end))
    excluded.sort()
    merged: list[list[float]] = []
    for start, end in excluded:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    current = 0.0
    keep: list[tuple[float, float]] = []
    for start, end in merged:
        if start > current:
            keep.append((current, start))
        current = max(current, end)
    if current < duration_sec:
        keep.append((current, float(duration_sec)))
    return keep


def segments_from_intervals(
    intervals: Sequence[tuple[float, float]],
    *,
    sfreq: float,
    window_sec: float,
    step_sec: float,
    total_samples: int,
    ictal_interval: tuple[float, float] | None = None,
) -> list[dict[str, Any]]:
    window_samples = max(2, int(round(float(window_sec) * float(sfreq))))
    step_samples = max(1, int(round(float(step_sec) * float(sfreq))))
    segments: list[dict[str, Any]] = []
    for interval_start, interval_end in intervals:
        first_start = max(0, int(math.ceil(float(interval_start) * sfreq)))
        last_start = min(int(total_samples) - window_samples, int(math.floor(float(interval_end) * sfreq)) - window_samples)
        if last_start < first_start:
            continue
        for start_sample in range(first_start, last_start + 1, step_samples):
            end_sample = start_sample + window_samples
            start_sec = float(start_sample) / max(float(sfreq), 1e-8)
            end_sec = float(end_sample) / max(float(sfreq), 1e-8)
            if ictal_interval is not None:
                onset, offset = ictal_interval
                overlap = max(0.0, min(end_sec, offset) - max(start_sec, onset))
                if overlap / max(float(window_sec), 1e-8) < 0.5:
                    continue
            center_sec = 0.5 * (start_sec + end_sec)
            segments.append(
                {
                    "segment_role": "strict_ictal_feature" if ictal_interval else "interictal_feature",
                    "target_duration_sec": float(window_sec),
                    "start_sec": start_sec,
                    "end_sec": end_sec,
                    "start_sample": int(start_sample),
                    "end_sample": int(end_sample),
                    "used_duration_sec": float(end_sec - start_sec),
                    "relative_center_sec": float(center_sec - ictal_interval[0]) if ictal_interval else float(center_sec),
                }
            )
    return segments


def remote_intervals_for_run(run: BidsRun, args) -> tuple[list[tuple[float, float]], float | None]:
    intervals = parse_all_seizure_intervals(run.events_path)
    if not intervals or run.json_path is None or not run.json_path.exists():
        return [], None
    try:
        meta = json.loads(run.json_path.read_text(encoding="utf-8"))
    except Exception:
        return [], None
    duration = safe_float(meta.get("RecordingDuration"))
    if duration is None:
        return [], None
    candidates = [
        (float(args.interictal_pre_buffer_sec), float(args.interictal_post_buffer_sec)),
        (float(args.interictal_fallback_buffer_sec), float(args.interictal_fallback_buffer_sec)),
    ]
    for pre_buffer, post_buffer in candidates:
        keep = interval_complement(intervals, duration_sec=duration, pre_buffer_sec=pre_buffer, post_buffer_sec=post_buffer)
        if sum(max(0.0, end - start) for start, end in keep) >= float(args.window_sec):
            return keep, pre_buffer
    return [], None
