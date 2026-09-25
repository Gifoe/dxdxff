from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from data_sources import BidsRun
from events import parse_all_seizure_intervals, segments_from_intervals
from ez_features import compute_window_feature_tensors, extract_run_feature_record
from module2_preprocessing import load_and_preprocess_edf
from module3_labels_metadata import parse_channel_labels


@dataclass
class InterictalBank:
    subject_id: str
    source_type: str
    run_id: str
    channel_names_norm: list[str]
    features: np.ndarray
    adjacency: np.ndarray
    buffer_used_sec: float | None = None


def _filter_contacts_to_preprocessed_channels(contacts_meta: pd.DataFrame, picked_channels_norm: Sequence[str]) -> pd.DataFrame:
    channel_order = {name: idx for idx, name in enumerate(picked_channels_norm)}
    contacts_meta = contacts_meta[contacts_meta["is_valid"] == 1].copy()
    contacts_meta = contacts_meta[contacts_meta["channel_name_norm"].isin(channel_order)].copy()
    if contacts_meta.empty:
        return contacts_meta
    contacts_meta["picked_order"] = contacts_meta["channel_name_norm"].map(channel_order)
    return contacts_meta.sort_values("picked_order").reset_index(drop=True)


def _contact_numbers(values: Sequence[object]) -> list[int | None]:
    out: list[int | None] = []
    for value in values:
        out.append(None if pd.isna(value) else int(value))
    return out


def _raw_placeholder(num_channels: int) -> tuple[np.ndarray, int, int]:
    return np.zeros((int(num_channels), 1), dtype=np.float32), 1, 0


def _make_run_info(run: BidsRun, onset: float | None, offset: float | None) -> dict[str, Any]:
    return {
        "subject_id": run.subject_id,
        "session": run.session,
        "run_id": run.run_id,
        "task": run.task,
        "phase_group": run.phase_group,
        "edf_path": str(run.edf_path),
        "channels_path": str(run.channels_path) if run.channels_path else None,
        "events_path": str(run.events_path) if run.events_path else None,
        "json_path": str(run.json_path) if run.json_path else None,
        "seizure_onset": onset,
        "seizure_offset": offset,
    }


def extract_b0_record(run: BidsRun, args) -> dict[str, Any] | None:
    intervals = parse_all_seizure_intervals(run.events_path)
    if not intervals:
        return None
    onset, offset = intervals[0]
    record = extract_run_feature_record(
        _make_run_info(run, onset, offset),
        target_sfreq=float(args.target_sfreq),
        prepost_context_sec=float(args.prepost_context_sec),
        raw_temporal_sfreq=float(args.raw_temporal_sfreq),
        ez_definition=str(args.ez_definition),
        spectral_min_freq=float(args.spectral_min_freq),
        spectral_max_freq=float(args.spectral_max_freq),
        graph_edge_quantile=float(args.graph_edge_quantile),
        graph_min_edge_weight=float(args.graph_min_edge_weight),
        sliding_window_sec=float(args.window_sec),
        sliding_step_sec=float(args.step_sec),
        extract_static_features=False,
        extract_window_tensors=True,
    )
    return record.to_dict() if record is not None else None


def extract_strict_ictal_records(run: BidsRun, args) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if run.channels_path is None:
        return [], [{"run_id": run.run_id, "drop_reason": "missing_channels_tsv"}]
    intervals = parse_all_seizure_intervals(run.events_path)
    if not intervals:
        return [], [{"run_id": run.run_id, "drop_reason": "missing_seizure_events"}]
    raw = None
    records: list[dict[str, Any]] = []
    qc_rows: list[dict[str, Any]] = []
    try:
        raw, picked_channels_norm, data, _ = load_and_preprocess_edf(
            str(run.edf_path),
            str(run.channels_path),
            target_sfreq=float(args.target_sfreq),
            bandpass_low=float(args.spectral_min_freq),
            bandpass_high=float(args.spectral_max_freq),
        )
        contacts_meta = parse_channel_labels(str(run.channels_path), ez_definition=str(args.ez_definition))
        contacts_meta = _filter_contacts_to_preprocessed_channels(contacts_meta, picked_channels_norm)
        if contacts_meta.empty:
            return [], [{"run_id": run.run_id, "drop_reason": "no_valid_labeled_channels"}]
        data = data[contacts_meta["picked_order"].to_numpy(dtype=int)]
        sfreq = float(raw.info["sfreq"])
        total_samples = int(raw.n_times)
        recording_duration = total_samples / max(sfreq, 1e-8)
        raw_waveform, raw_valid_samples, raw_valid_start = _raw_placeholder(data.shape[0])
        for seizure_idx, (onset, offset) in enumerate(intervals, start=1):
            onset = max(0.0, min(float(onset), recording_duration))
            offset = max(0.0, min(float(offset), recording_duration))
            duration = max(0.0, offset - onset)
            if duration < 2.0:
                qc_rows.append({"run_id": run.run_id, "seizure_index": seizure_idx, "drop_reason": "ictal_duration_lt_2s", "ictal_duration_sec": duration})
                continue
            windows = segments_from_intervals(
                [(onset, offset)],
                sfreq=sfreq,
                window_sec=float(args.window_sec),
                step_sec=float(args.step_sec),
                total_samples=total_samples,
                ictal_interval=(onset, offset),
            )
            low_coverage = int(duration < 5.0)
            min_required_windows = 1 if low_coverage else int(args.min_ictal_windows)
            if len(windows) < min_required_windows:
                qc_rows.append(
                    {
                        "run_id": run.run_id,
                        "seizure_index": seizure_idx,
                        "drop_reason": "too_few_ictal_windows",
                        "num_ictal_windows": len(windows),
                        "min_required_windows": min_required_windows,
                    }
                )
                continue
            window_features, window_adjacency, centers = compute_window_feature_tensors(
                data,
                windows,
                sfreq=sfreq,
                spectral_min_freq=float(args.spectral_min_freq),
                spectral_max_freq=float(args.spectral_max_freq),
                edge_quantile=float(args.graph_edge_quantile),
                min_edge_weight=float(args.graph_min_edge_weight),
            )
            run_id = f"{run.run_id}__sz-{seizure_idx:02d}"
            sample = {
                "sample_id": f"{run_id}__strict_ictal_{float(args.window_sec):g}s_{float(args.step_sec):g}s",
                "analysis_phase": "strict_ictal",
                "start_sec": float(onset),
                "end_sec": float(offset),
                "seizure_onset_sec": float(onset),
                "seizure_offset_sec": float(offset),
                "ictal_duration_total_sec": float(duration),
                "ictal_duration_used_sec": float(duration),
                "ictal_duration_sec": float(duration),
                "num_ictal_windows": int(window_features.shape[0]),
                "low_ictal_coverage": int(low_coverage),
                "bad_segment_ratio": 0.0,
                "bad_channel_ratio": 0.0,
                "artifact_subsegments": 0,
                "feature_scales_sec": [float(args.window_sec)],
                "feature_scale_used_secs": [float(item["used_duration_sec"]) for item in windows],
                "raw_temporal_duration_sec": 1.0,
                "raw_temporal_sfreq": float(args.raw_temporal_sfreq),
                "raw_valid_samples": int(raw_valid_samples),
                "raw_valid_start_sample": int(raw_valid_start),
                "spectral_features": np.zeros((data.shape[0], 0), dtype=np.float32),
                "graph_features": np.zeros((data.shape[0], 0), dtype=np.float32),
                "window_features": window_features,
                "window_adjacency": window_adjacency,
                "window_relative_centers_sec": centers,
                "raw_waveform": raw_waveform,
            }
            records.append(
                {
                    "subject_id": run.subject_id,
                    "run_id": run_id,
                    "task": "ictal",
                    "phase_group": "ictal",
                    "channel_names_norm": contacts_meta["channel_name_norm"].tolist(),
                    "contact_groups": contacts_meta["contact_group"].fillna("").astype(str).tolist(),
                    "contact_numbers": _contact_numbers(contacts_meta["contact_number"].tolist()),
                    "labels": contacts_meta["is_ez"].to_numpy(dtype=np.float32, copy=True),
                    "sfreq": float(sfreq),
                    "sample": sample,
                    "metadata": {"source_center": "hup", "source_bids_run_id": run.run_id, "seizure_index": seizure_idx},
                }
            )
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()
    return records, qc_rows


def extract_interictal_bank_from_run(
    run: BidsRun,
    args,
    *,
    source_type: str,
    intervals: Sequence[tuple[float, float]] | None = None,
    buffer_used_sec: float | None = None,
) -> tuple[InterictalBank | None, dict[str, Any]]:
    if run.channels_path is None:
        return None, {"run_id": run.run_id, "drop_reason": "missing_channels_tsv"}
    raw = None
    try:
        raw, picked_channels_norm, data, _ = load_and_preprocess_edf(
            str(run.edf_path),
            str(run.channels_path),
            target_sfreq=float(args.target_sfreq),
            bandpass_low=float(args.spectral_min_freq),
            bandpass_high=float(args.spectral_max_freq),
        )
        contacts_meta = parse_channel_labels(str(run.channels_path), ez_definition=str(args.ez_definition))
        contacts_meta = _filter_contacts_to_preprocessed_channels(contacts_meta, picked_channels_norm)
        if contacts_meta.empty:
            return None, {"run_id": run.run_id, "drop_reason": "no_valid_channels"}
        data = data[contacts_meta["picked_order"].to_numpy(dtype=int)]
        sfreq = float(raw.info["sfreq"])
        total_samples = int(raw.n_times)
        duration = total_samples / max(sfreq, 1e-8)
        windows = segments_from_intervals(
            list(intervals) if intervals is not None else [(0.0, duration)],
            sfreq=sfreq,
            window_sec=float(args.window_sec),
            step_sec=float(args.step_sec),
            total_samples=total_samples,
        )
        if not windows:
            return None, {"run_id": run.run_id, "drop_reason": "no_interictal_windows"}
        features, adjacency, _ = compute_window_feature_tensors(
            data,
            windows,
            sfreq=sfreq,
            spectral_min_freq=float(args.spectral_min_freq),
            spectral_max_freq=float(args.spectral_max_freq),
            edge_quantile=float(args.graph_edge_quantile),
            min_edge_weight=float(args.graph_min_edge_weight),
        )
        max_windows = int(args.max_interictal_windows_per_run)
        if max_windows > 0 and features.shape[0] > max_windows:
            idx = np.linspace(0, features.shape[0] - 1, max_windows).round().astype(int)
            features = features[idx]
            adjacency = adjacency[idx]
        return (
            InterictalBank(run.subject_id, source_type, run.run_id, contacts_meta["channel_name_norm"].tolist(), features, adjacency, buffer_used_sec),
            {"run_id": run.run_id, "num_interictal_windows": int(features.shape[0]), "source_type": source_type, "buffer_used_sec": buffer_used_sec},
        )
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()
