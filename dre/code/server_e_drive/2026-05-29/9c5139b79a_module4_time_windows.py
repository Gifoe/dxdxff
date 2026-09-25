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


def _build_interval_segment(
    *,
    sfreq: float,
    total_samples: int,
    start_sec: float,
    end_sec: float,
    target_duration_sec: float,
    segment_role: str,
) -> Dict[str, Any]:
    start_sample = int(round(float(start_sec) * sfreq))
    end_sample = int(round(float(end_sec) * sfreq))
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
    return _build_interval_segment(
        sfreq=sfreq,
        total_samples=total_samples,
        start_sec=onset_sec,
        end_sec=onset_sec + planned_duration_sec,
        target_duration_sec=target_duration_sec,
        segment_role=segment_role,
    )


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


def create_prepost_ictal_sample(
    raw,
    data,
    run_info: Dict[str, Any],
    *,
    context_duration_sec: float = 30.0,
    raw_temporal_sfreq: float = 1000.0,
    artifact_subseg_sec: float = 1.0,
    bad_subseg_channel_ratio_threshold: float = 0.25,
    bad_segment_ratio_threshold: float = 0.40,
    bad_channel_ratio_threshold: float = 0.20,
    bad_channel_subseg_ratio_threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    if not has_valid_ictal_bounds(run_info):
        return None

    context_duration_sec = float(context_duration_sec)
    if context_duration_sec <= 0.0:
        raise ValueError("context_duration_sec must be positive.")

    sfreq = float(raw.info["sfreq"])
    recording_duration_sec = float(raw.n_times) / max(sfreq, 1e-8)
    onset_sec = max(0.0, float(_safe_float(run_info.get("seizure_onset")) or 0.0))
    offset_sec = min(recording_duration_sec, float(_safe_float(run_info.get("seizure_offset")) or 0.0))
    if onset_sec >= recording_duration_sec or offset_sec <= onset_sec:
        return None

    pre_start_sec = max(0.0, onset_sec - context_duration_sec)
    pre_end_sec = onset_sec
    post_start_sec = onset_sec
    post_end_sec = min(offset_sec, onset_sec + context_duration_sec, recording_duration_sec)

    if pre_end_sec <= pre_start_sec or post_end_sec <= post_start_sec:
        return None

    pre_segment = _build_interval_segment(
        sfreq=sfreq,
        total_samples=int(raw.n_times),
        start_sec=pre_start_sec,
        end_sec=pre_end_sec,
        target_duration_sec=context_duration_sec,
        segment_role="pre_onset",
    )
    post_segment = _build_interval_segment(
        sfreq=sfreq,
        total_samples=int(raw.n_times),
        start_sec=post_start_sec,
        end_sec=post_end_sec,
        target_duration_sec=context_duration_sec,
        segment_role="post_onset",
    )

    pre_data = data[:, pre_segment["start_sample"] : pre_segment["end_sample"]]
    post_data = data[:, post_segment["start_sample"] : post_segment["end_sample"]]
    if pre_data.shape[-1] == 0 or post_data.shape[-1] == 0:
        return None

    artifact_data = np.concatenate([pre_data, post_data], axis=-1)
    artifact_stats = _compute_segment_artifact_stats(
        artifact_data,
        sfreq,
        subseg_sec=artifact_subseg_sec,
        bad_subseg_channel_ratio_threshold=bad_subseg_channel_ratio_threshold,
        bad_channel_subseg_ratio_threshold=bad_channel_subseg_ratio_threshold,
    )
    unusable = (
        artifact_stats["bad_segment_ratio"] > float(bad_segment_ratio_threshold)
        and artifact_stats["bad_channel_ratio"] > float(bad_channel_ratio_threshold)
    )

    return {
        "sample_id": f"{run_info.get('run_id', 'unknown')}__prepost_onset",
        "analysis_phase": "prepost_onset",
        "seizure_onset_sec": float(onset_sec),
        "seizure_offset_sec": float(offset_sec),
        "ictal_duration_total_sec": float(offset_sec - onset_sec),
        "ictal_duration_used_sec": float(post_segment["used_duration_sec"]),
        "context_duration_sec": float(context_duration_sec),
        "feature_scales_sec": [float(context_duration_sec), float(context_duration_sec)],
        "feature_segments": [pre_segment, post_segment],
        "raw_segments": [pre_segment, post_segment],
        "raw_temporal_duration_sec": float(context_duration_sec * 2.0),
        "raw_temporal_sfreq": float(raw_temporal_sfreq),
        "unusable_mask": bool(unusable),
        "bad_segment_ratio": float(artifact_stats["bad_segment_ratio"]),
        "bad_channel_ratio": float(artifact_stats["bad_channel_ratio"]),
        "artifact_subsegments": int(artifact_stats["n_subsegments"]),
    }


def create_sliding_onset_ictal_sample(
    raw,
    data,
    run_info: Dict[str, Any],
    *,
    context_duration_sec: float = 30.0,
    window_duration_sec: float = 2.0,
    window_step_sec: float = 1.0,
    raw_temporal_sfreq: float = 1000.0,
    artifact_subseg_sec: float = 1.0,
    bad_subseg_channel_ratio_threshold: float = 0.25,
    bad_segment_ratio_threshold: float = 0.40,
    bad_channel_ratio_threshold: float = 0.20,
    bad_channel_subseg_ratio_threshold: float = 0.50,
) -> Optional[Dict[str, Any]]:
    if not has_valid_ictal_bounds(run_info):
        return None

    context_duration_sec = float(context_duration_sec)
    window_duration_sec = float(window_duration_sec)
    window_step_sec = float(window_step_sec)
    if context_duration_sec <= 0.0:
        raise ValueError("context_duration_sec must be positive.")
    if window_duration_sec <= 0.0 or window_step_sec <= 0.0:
        raise ValueError("window_duration_sec and window_step_sec must be positive.")

    sfreq = float(raw.info["sfreq"])
    recording_duration_sec = float(raw.n_times) / max(sfreq, 1e-8)
    onset_sec = max(0.0, float(_safe_float(run_info.get("seizure_onset")) or 0.0))
    offset_sec = min(recording_duration_sec, float(_safe_float(run_info.get("seizure_offset")) or 0.0))
    if onset_sec >= recording_duration_sec or offset_sec <= onset_sec:
        return None

    analysis_start_sec = max(0.0, onset_sec - context_duration_sec)
    analysis_end_sec = min(onset_sec + context_duration_sec, recording_duration_sec)
    if analysis_start_sec >= onset_sec or analysis_end_sec <= onset_sec:
        return None

    analysis_segment = _build_interval_segment(
        sfreq=sfreq,
        total_samples=int(raw.n_times),
        start_sec=analysis_start_sec,
        end_sec=analysis_end_sec,
        target_duration_sec=context_duration_sec * 2.0,
        segment_role="onset_context",
    )

    window_samples = max(2, int(round(window_duration_sec * sfreq)))
    step_samples = max(1, int(round(window_step_sec * sfreq)))
    first_start = int(analysis_segment["start_sample"])
    last_start = int(analysis_segment["end_sample"]) - window_samples
    if last_start < first_start:
        return None

    sliding_windows: List[Dict[str, Any]] = []
    for start_sample in range(first_start, last_start + 1, step_samples):
        end_sample = start_sample + window_samples
        start_sec = float(start_sample) / max(sfreq, 1e-8)
        end_sec = float(end_sample) / max(sfreq, 1e-8)
        center_sec = 0.5 * (start_sec + end_sec)
        sliding_windows.append(
            {
                "segment_role": "sliding_feature",
                "target_duration_sec": float(window_duration_sec),
                "start_sec": float(start_sec),
                "end_sec": float(end_sec),
                "start_sample": int(start_sample),
                "end_sample": int(end_sample),
                "used_duration_sec": float(end_sec - start_sec),
                "relative_start_sec": float(start_sec - onset_sec),
                "relative_end_sec": float(end_sec - onset_sec),
                "relative_center_sec": float(center_sec - onset_sec),
            }
        )

    if not sliding_windows:
        return None
    centers = np.asarray([window["relative_center_sec"] for window in sliding_windows], dtype=np.float32)
    if not (np.any(centers < 0.0) and np.any(centers >= 0.0)):
        return None

    segment_data = data[:, int(analysis_segment["start_sample"]) : int(analysis_segment["end_sample"])]
    if segment_data.shape[-1] < window_samples:
        return None

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

    return {
        "sample_id": f"{run_info.get('run_id', 'unknown')}__onset_sliding_{window_duration_sec:g}s_{window_step_sec:g}s",
        "analysis_phase": "onset_sliding",
        "seizure_onset_sec": float(onset_sec),
        "seizure_offset_sec": float(offset_sec),
        "ictal_duration_total_sec": float(offset_sec - onset_sec),
        "ictal_duration_used_sec": float(max(0.0, analysis_end_sec - onset_sec)),
        "context_duration_sec": float(context_duration_sec),
        "window_duration_sec": float(window_duration_sec),
        "window_step_sec": float(window_step_sec),
        "feature_scales_sec": [float(window_duration_sec)],
        "feature_segments": sliding_windows,
        "analysis_segment": analysis_segment,
        "raw_segment": analysis_segment,
        "raw_temporal_duration_sec": float(context_duration_sec * 2.0),
        "raw_temporal_sfreq": float(raw_temporal_sfreq),
        "unusable_mask": bool(unusable),
        "bad_segment_ratio": float(artifact_stats["bad_segment_ratio"]),
        "bad_channel_ratio": float(artifact_stats["bad_channel_ratio"]),
        "artifact_subsegments": int(artifact_stats["n_subsegments"]),
        "n_sliding_windows": int(len(sliding_windows)),
    }


__all__ = [
    "create_multiscale_ictal_sample",
    "create_prepost_ictal_sample",
    "create_sliding_onset_ictal_sample",
    "has_valid_ictal_bounds",
    "is_ictal_run",
]
