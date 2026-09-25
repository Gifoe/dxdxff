from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

import numpy as np


def _safe_float(value):
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        return None
    if np.isnan(value_f):
        return None
    return value_f


def is_ictal_run(task: str, phase_group: str) -> bool:
    phase_group_lower = str(phase_group).lower()
    if "interictal" in phase_group_lower:
        return False
    if "ictal" in phase_group_lower:
        return True

    task_lower = str(task).lower()
    return "ictal" in task_lower and "interictal" not in task_lower


def has_valid_ictal_bounds(run_info: Dict[str, Any]) -> bool:
    if not is_ictal_run(str(run_info.get("task", "")), str(run_info.get("phase_group", ""))):
        return False

    onset = _safe_float(run_info.get("seizure_onset"))
    offset = _safe_float(run_info.get("seizure_offset"))
    if onset is None or offset is None:
        return False
    return float(offset) > float(onset)


def _coerce_duration_list(feature_durations_sec: Iterable[float]) -> List[float]:
    durations = sorted({float(value) for value in feature_durations_sec if float(value) > 0.0})
    if not durations:
        raise ValueError("feature_durations_sec must contain at least one positive duration.")
    return durations


def _compute_segment_artifact_stats(
    segment_data,
    sfreq,
    *,
    subseg_sec: float = 1.0,
    amplitude_uv_threshold: float = 2000.0,
    bad_subseg_channel_ratio_threshold: float = 0.25,
    bad_channel_subseg_ratio_threshold: float = 0.50,
):
    n_channels, n_samples = segment_data.shape
    if n_channels == 0 or n_samples == 0:
        return {"bad_segment_ratio": 1.0, "bad_channel_ratio": 1.0, "n_subsegments": 0}

    subseg_len = max(1, int(round(float(subseg_sec) * float(sfreq))))
    amp_threshold = float(amplitude_uv_threshold) * 1e-6

    bad_segment_flags = []
    bad_channel_hits = np.zeros(n_channels, dtype=np.int32)

    for start in range(0, n_samples, subseg_len):
        end = min(start + subseg_len, n_samples)
        segment = segment_data[:, start:end]
        if segment.shape[1] == 0:
            continue

        amplitude_range = np.max(segment, axis=1) - np.min(segment, axis=1)
        flatline = np.all(np.isclose(segment, segment[:, :1], atol=1e-12), axis=1)
        bad_channels = (amplitude_range > amp_threshold) | flatline

        bad_channel_hits += bad_channels.astype(np.int32)
        bad_segment_flags.append(bad_channels.mean() > float(bad_subseg_channel_ratio_threshold))

    n_subsegments = len(bad_segment_flags)
    if n_subsegments == 0:
        return {"bad_segment_ratio": 1.0, "bad_channel_ratio": 1.0, "n_subsegments": 0}

    bad_segment_ratio = float(np.mean(bad_segment_flags))
    channel_bad_subseg_ratio = bad_channel_hits / float(n_subsegments)
    bad_channel_ratio = float(np.mean(channel_bad_subseg_ratio > float(bad_channel_subseg_ratio_threshold)))

    return {
        "bad_segment_ratio": bad_segment_ratio,
        "bad_channel_ratio": bad_channel_ratio,
        "n_subsegments": n_subsegments,
    }


def _build_onset_segment(
    *,
    sfreq: float,
    total_samples: int,
    onset_sec: float,
    offset_sec: float,
    target_duration_sec: float,
    segment_role: str,
) -> Dict[str, Any]:
    ictal_duration_total_sec = float(max(offset_sec - onset_sec, 0.0))
    planned_duration_sec = float(min(ictal_duration_total_sec, float(target_duration_sec)))
    end_sec = onset_sec + planned_duration_sec

    start_sample = int(round(onset_sec * sfreq))
    end_sample = int(round(end_sec * sfreq))
    start_sample = max(0, min(start_sample, int(total_samples)))
    end_sample = min(max(start_sample + 1, end_sample), int(total_samples))

    actual_start_sec = float(start_sample) / max(sfreq, 1e-8)
    actual_end_sec = float(end_sample) / max(sfreq, 1e-8)
    used_duration_sec = float(actual_end_sec - actual_start_sec)

    return {
        "segment_role": str(segment_role),
        "target_duration_sec": float(target_duration_sec),
        "start_sec": float(actual_start_sec),
        "end_sec": float(actual_end_sec),
        "start_sample": int(start_sample),
        "end_sample": int(end_sample),
        "used_duration_sec": float(used_duration_sec),
    }


def create_multiscale_ictal_sample(
    raw,
    data,
    run_info: Dict[str, Any],
    *,
    feature_durations_sec: Iterable[float],
    raw_temporal_duration_sec: float,
    artifact_subseg_sec: float = 1.0,
    bad_subseg_channel_ratio_threshold: float = 0.25,
    bad_segment_ratio_threshold: float = 0.40,
    bad_channel_ratio_threshold: float = 0.20,
    bad_channel_subseg_ratio_threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    if not has_valid_ictal_bounds(run_info):
        return None

    feature_durations = _coerce_duration_list(feature_durations_sec)
    raw_temporal_duration_sec = float(raw_temporal_duration_sec)
    if raw_temporal_duration_sec <= 0.0:
        raise ValueError("raw_temporal_duration_sec must be positive.")

    sfreq = float(raw.info["sfreq"])
    recording_duration_sec = float(raw.n_times) / max(sfreq, 1e-8)
    onset_sec = max(0.0, float(_safe_float(run_info.get("seizure_onset")) or 0.0))
    offset_sec = min(recording_duration_sec, float(_safe_float(run_info.get("seizure_offset")) or 0.0))
    if onset_sec >= recording_duration_sec or offset_sec <= onset_sec:
        return None

    ictal_duration_total_sec = float(offset_sec - onset_sec)
    longest_analysis_duration_sec = max(max(feature_durations), raw_temporal_duration_sec)
    master_segment = _build_onset_segment(
        sfreq=sfreq,
        total_samples=int(raw.n_times),
        onset_sec=onset_sec,
        offset_sec=offset_sec,
        target_duration_sec=longest_analysis_duration_sec,
        segment_role="artifact_check",
    )

    segment_data = data[:, master_segment["start_sample"] : master_segment["end_sample"]]
    artifact_stats = _compute_segment_artifact_stats(
        segment_data,
        sfreq,
        subseg_sec=artifact_subseg_sec,
        bad_subseg_channel_ratio_threshold=bad_subseg_channel_ratio_threshold,
        bad_channel_subseg_ratio_threshold=bad_channel_subseg_ratio_threshold,
    )
    unusable = (
        artifact_stats["bad_segment_ratio"] > float(bad_segment_ratio_threshold)
        and artifact_stats["bad_channel_ratio"] > float(bad_channel_ratio_threshold)
    )

    feature_segments = [
        _build_onset_segment(
            sfreq=sfreq,
            total_samples=int(raw.n_times),
            onset_sec=onset_sec,
            offset_sec=offset_sec,
            target_duration_sec=duration_sec,
            segment_role="feature",
        )
        for duration_sec in feature_durations
    ]
    raw_segment = _build_onset_segment(
        sfreq=sfreq,
        total_samples=int(raw.n_times),
        onset_sec=onset_sec,
        offset_sec=offset_sec,
        target_duration_sec=raw_temporal_duration_sec,
        segment_role="raw_temporal",
    )

    return {
        "sample_id": f"{run_info.get('run_id', 'unknown')}__ictal_hybrid",
        "analysis_phase": "ictal",
        "seizure_onset_sec": float(onset_sec),
        "seizure_offset_sec": float(offset_sec),
        "ictal_duration_total_sec": float(ictal_duration_total_sec),
        "ictal_duration_used_sec": float(master_segment["used_duration_sec"]),
        "feature_scales_sec": [float(value) for value in feature_durations],
        "feature_segments": feature_segments,
        "raw_segment": raw_segment,
        "unusable_mask": bool(unusable),
        "bad_segment_ratio": float(artifact_stats["bad_segment_ratio"]),
        "bad_channel_ratio": float(artifact_stats["bad_channel_ratio"]),
        "artifact_subsegments": int(artifact_stats["n_subsegments"]),
    }


__all__ = [
    "create_multiscale_ictal_sample",
    "has_valid_ictal_bounds",
    "is_ictal_run",
]
