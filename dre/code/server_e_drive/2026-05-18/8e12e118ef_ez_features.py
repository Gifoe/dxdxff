from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.integrate import trapezoid
from scipy.signal import welch

from graph_channel import (
    GRAPH_FEATURE_NAMES,
    compute_graph_node_features,
    compute_thresholded_abs_pearson_adjacency,
)
from module2_preprocessing import load_and_preprocess_edf
from module3_labels_metadata import parse_channel_labels
from module4_time_windows import create_sliding_onset_ictal_sample, has_valid_ictal_bounds


DEFAULT_FEATURE_SCALES_SEC: Sequence[float] = (2.0,)
SLIDING_WINDOW_SUMMARY_NAMES: Sequence[str] = (
    "window_mean",
    "window_std",
    "window_max",
    "window_slope",
    "pre_window_mean",
    "post_window_mean",
    "post_minus_pre_window_mean",
)
PREPOST_PHASE_NAMES: Sequence[str] = SLIDING_WINDOW_SUMMARY_NAMES

SPECTRAL_BANDS: Sequence[Tuple[str, float, float]] = (
    ("delta", 1.0, 4.0),
    ("theta", 4.0, 8.0),
    ("alpha", 8.0, 13.0),
    ("beta", 13.0, 30.0),
    ("low_gamma", 30.0, 80.0),
    ("high_gamma", 80.0, 150.0),
)

BASE_SPECTRAL_FEATURE_NAMES: Sequence[str] = (
    "log_bp_delta",
    "log_bp_theta",
    "log_bp_alpha",
    "log_bp_beta",
    "log_bp_low_gamma",
    "log_bp_high_gamma",
    "log_total_power",
    "rms",
    "variance",
    "line_length_per_sec",
    "spectral_entropy",
    "hjorth_mobility",
    "hjorth_complexity",
)

DYNAMIC_GRAPH_STAT_NAMES: Sequence[str] = SLIDING_WINDOW_SUMMARY_NAMES
DYNAMIC_SPECTRAL_FEATURE_NAMES: Sequence[str] = tuple(
    f"{summary_name}_{feature_name}"
    for summary_name in SLIDING_WINDOW_SUMMARY_NAMES
    for feature_name in BASE_SPECTRAL_FEATURE_NAMES
)
DYNAMIC_GRAPH_FEATURE_NAMES: Sequence[str] = tuple(
    f"{summary_name}_{graph_name}"
    for summary_name in SLIDING_WINDOW_SUMMARY_NAMES
    for graph_name in GRAPH_FEATURE_NAMES
)
WINDOW_NODE_FEATURE_NAMES: Sequence[str] = tuple(BASE_SPECTRAL_FEATURE_NAMES) + tuple(GRAPH_FEATURE_NAMES)


def parse_duration_list(value: Any) -> List[float]:
    if isinstance(value, str):
        parts = [part.strip() for part in value.split(",")]
        durations = [float(part) for part in parts if part]
    elif isinstance(value, Iterable):
        durations = [float(part) for part in value]
    else:
        raise TypeError(f"Unsupported feature duration specification: {type(value)!r}")

    durations = sorted({float(part) for part in durations if float(part) > 0.0})
    if not durations:
        raise ValueError("At least one positive feature duration is required.")
    return durations


def build_multiscale_feature_names(base_names: Sequence[str], scales_sec: Sequence[float]) -> Tuple[str, ...]:
    return tuple(f"{phase}_{base_name}" for phase in PREPOST_PHASE_NAMES for base_name in base_names)


def infer_feature_dim(feature_scales_sec: Sequence[float]) -> int:
    return len(DYNAMIC_SPECTRAL_FEATURE_NAMES) + len(DYNAMIC_GRAPH_FEATURE_NAMES)


FEATURE_DIM = infer_feature_dim(DEFAULT_FEATURE_SCALES_SEC)


def _trapz(y: np.ndarray, x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.asarray(trapezoid(y, x=x, axis=axis), dtype=np.float32)


@dataclass
class RunFeatureRecord:
    subject_id: str
    run_id: str
    task: str
    phase_group: str
    channel_names_norm: List[str]
    contact_groups: List[str]
    contact_numbers: List[Optional[int]]
    labels: np.ndarray
    sfreq: float
    sample: Dict[str, Any]
    metadata: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "subject_id": self.subject_id,
            "run_id": self.run_id,
            "task": self.task,
            "phase_group": self.phase_group,
            "channel_names_norm": list(self.channel_names_norm),
            "contact_groups": list(self.contact_groups),
            "contact_numbers": list(self.contact_numbers),
            "labels": np.asarray(self.labels, dtype=np.float32),
            "sfreq": float(self.sfreq),
            "sample": {
                "sample_id": str(self.sample["sample_id"]),
                "analysis_phase": str(self.sample["analysis_phase"]),
                "start_sec": float(self.sample["start_sec"]),
                "end_sec": float(self.sample["end_sec"]),
                "seizure_onset_sec": float(self.sample["seizure_onset_sec"]),
                "seizure_offset_sec": float(self.sample["seizure_offset_sec"]),
                "ictal_duration_total_sec": float(self.sample["ictal_duration_total_sec"]),
                "ictal_duration_used_sec": float(self.sample["ictal_duration_used_sec"]),
                "bad_segment_ratio": float(self.sample["bad_segment_ratio"]),
                "bad_channel_ratio": float(self.sample["bad_channel_ratio"]),
                "artifact_subsegments": int(self.sample["artifact_subsegments"]),
                "feature_scales_sec": [float(value) for value in self.sample["feature_scales_sec"]],
                "feature_scale_used_secs": [float(value) for value in self.sample["feature_scale_used_secs"]],
                "raw_temporal_duration_sec": float(self.sample["raw_temporal_duration_sec"]),
                "raw_temporal_sfreq": float(self.sample["raw_temporal_sfreq"]),
                "raw_valid_samples": int(self.sample["raw_valid_samples"]),
                "raw_valid_start_sample": int(self.sample.get("raw_valid_start_sample", 0)),
                "spectral_features": np.asarray(self.sample["spectral_features"], dtype=np.float32),
                "graph_features": np.asarray(self.sample["graph_features"], dtype=np.float32),
                "window_features": np.asarray(self.sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32),
                "window_adjacency": np.asarray(self.sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32),
                "window_relative_centers_sec": np.asarray(
                    self.sample.get("window_relative_centers_sec", np.zeros((0,))),
                    dtype=np.float32,
                ),
                "raw_waveform": np.asarray(self.sample["raw_waveform"], dtype=np.float32),
            },
            "metadata": dict(self.metadata),
        }


def _filter_contacts_to_preprocessed_channels(
    contacts_meta: pd.DataFrame,
    picked_channels_norm: Sequence[str],
) -> pd.DataFrame:
    channel_order = {name: idx for idx, name in enumerate(picked_channels_norm)}
    contacts_meta = contacts_meta[contacts_meta["is_valid"] == 1].copy()
    contacts_meta = contacts_meta[contacts_meta["channel_name_norm"].isin(channel_order)].copy()
    if contacts_meta.empty:
        return contacts_meta

    contacts_meta["picked_order"] = contacts_meta["channel_name_norm"].map(channel_order)
    contacts_meta = contacts_meta.sort_values("picked_order").reset_index(drop=True)
    return contacts_meta


def _compute_hjorth_parameters(channel_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    activity = np.var(channel_data, axis=-1)
    diff1 = np.diff(channel_data, axis=-1)
    diff2 = np.diff(diff1, axis=-1)

    diff1_var = np.var(diff1, axis=-1) + 1e-8
    diff2_var = np.var(diff2, axis=-1) + 1e-8
    activity_safe = activity + 1e-8

    mobility = np.sqrt(diff1_var / activity_safe)
    complexity = np.sqrt(diff2_var / diff1_var) / (mobility + 1e-8)
    return mobility.astype(np.float32, copy=False), complexity.astype(np.float32, copy=False)


def compute_spectral_channel_features(
    segment_data: np.ndarray,
    sfreq: float,
    *,
    spectral_min_freq: float = 1.0,
    spectral_max_freq: float = 150.0,
) -> np.ndarray:
    data = np.asarray(segment_data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("segment_data must have shape [channels, time].")

    num_channels, num_samples = data.shape
    if num_channels == 0:
        return np.zeros((0, len(BASE_SPECTRAL_FEATURE_NAMES)), dtype=np.float32)
    if num_samples < 2:
        raise ValueError("segment_data must contain at least 2 samples to compute spectral features.")

    preferred_nperseg = max(64, int(round(float(sfreq) * 2.0)))
    nperseg = int(min(num_samples, preferred_nperseg))
    noverlap = int(min(max(0, nperseg // 2), max(nperseg - 1, 0)))
    freqs, psd = welch(
        data,
        fs=float(sfreq),
        nperseg=nperseg,
        noverlap=noverlap,
        axis=-1,
        detrend=False,
    )
    psd = np.asarray(psd, dtype=np.float32)

    restricted_mask = (freqs >= float(spectral_min_freq)) & (freqs <= float(spectral_max_freq))
    restricted_freqs = freqs[restricted_mask]
    restricted_psd = psd[:, restricted_mask]
    if restricted_psd.shape[-1] == 0:
        raise ValueError(
            f"No spectral bins available between {spectral_min_freq} and {spectral_max_freq} Hz."
        )

    band_features = []
    for _, low, high in SPECTRAL_BANDS:
        band_mask = (restricted_freqs >= float(low)) & (restricted_freqs < float(high))
        if np.any(band_mask):
            band_power = _trapz(restricted_psd[:, band_mask], restricted_freqs[band_mask], axis=-1)
        else:
            band_power = np.zeros(num_channels, dtype=np.float32)
        band_features.append(np.log1p(band_power).astype(np.float32, copy=False))

    total_power = _trapz(restricted_psd, restricted_freqs, axis=-1).astype(np.float32, copy=False)
    log_total_power = np.log1p(total_power)

    rms = np.sqrt(np.mean(np.square(data), axis=-1) + 1e-8).astype(np.float32, copy=False)
    variance = np.var(data, axis=-1).astype(np.float32, copy=False)
    duration_sec = max(float(num_samples) / max(float(sfreq), 1e-8), 1e-6)
    line_length_per_sec = (
        np.sum(np.abs(np.diff(data, axis=-1)), axis=-1) / duration_sec
    ).astype(np.float32, copy=False)

    psd_norm = restricted_psd / np.clip(restricted_psd.sum(axis=-1, keepdims=True), 1e-8, None)
    spectral_entropy = (
        -np.sum(psd_norm * np.log(psd_norm + 1e-8), axis=-1) / np.log(psd_norm.shape[-1] + 1e-8)
    ).astype(np.float32, copy=False)

    hjorth_mobility, hjorth_complexity = _compute_hjorth_parameters(data)

    features = np.stack(
        [
            *band_features,
            log_total_power,
            rms,
            variance,
            line_length_per_sec,
            spectral_entropy,
            hjorth_mobility,
            hjorth_complexity,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)

    if features.shape[-1] != len(BASE_SPECTRAL_FEATURE_NAMES):
        raise RuntimeError(
            f"Expected {len(BASE_SPECTRAL_FEATURE_NAMES)} spectral features, got {features.shape[-1]}."
        )
    return features


def compute_dynamic_graph_channel_features(
    segment_data: np.ndarray,
    sfreq: float,
    *,
    window_sec: float = 2.0,
    step_sec: float = 1.0,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> np.ndarray:
    data = np.asarray(segment_data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("segment_data must have shape [channels, time].")

    num_channels, num_samples = data.shape
    num_base_features = len(GRAPH_FEATURE_NAMES)
    if num_channels == 0:
        return np.zeros((0, len(DYNAMIC_GRAPH_FEATURE_NAMES)), dtype=np.float32)
    if num_samples < 2:
        raise ValueError("segment_data must contain at least 2 samples to compute dynamic graph features.")

    window_samples = max(2, int(round(float(window_sec) * float(sfreq))))
    window_samples = min(window_samples, num_samples)
    step_samples = max(1, int(round(float(step_sec) * float(sfreq))))
    last_start = max(num_samples - window_samples, 0)
    starts = list(range(0, last_start + 1, step_samples))
    if starts[-1] != last_start:
        starts.append(last_start)

    graph_series = []
    for start in starts:
        end = min(start + window_samples, num_samples)
        graph_series.append(
            compute_graph_node_features(
                data[:, start:end],
                edge_quantile=edge_quantile,
                min_edge_weight=min_edge_weight,
            )
        )

    series = np.stack(graph_series, axis=0).astype(np.float32, copy=False)
    mean_feat = series.mean(axis=0)
    std_feat = series.std(axis=0)
    max_feat = series.max(axis=0)

    if series.shape[0] <= 1:
        slope_feat = np.zeros((num_channels, num_base_features), dtype=np.float32)
    else:
        time_axis = np.linspace(-0.5, 0.5, series.shape[0], dtype=np.float32)
        denom = float(np.sum(np.square(time_axis))) + 1e-8
        centered = series - mean_feat[None, :, :]
        slope_feat = np.sum(centered * time_axis[:, None, None], axis=0) / denom

    dynamic_features = np.concatenate(
        [mean_feat, std_feat, max_feat, slope_feat.astype(np.float32, copy=False)],
        axis=-1,
    ).astype(np.float32, copy=False)
    if dynamic_features.shape[-1] != len(DYNAMIC_GRAPH_FEATURE_NAMES):
        raise RuntimeError(
            f"Expected {len(DYNAMIC_GRAPH_FEATURE_NAMES)} dynamic graph features, "
            f"got {dynamic_features.shape[-1]}."
        )
    return dynamic_features


def _aggregate_window_feature_series(
    feature_series: np.ndarray,
    relative_centers_sec: Sequence[float],
) -> np.ndarray:
    series = np.asarray(feature_series, dtype=np.float32)
    if series.ndim != 3:
        raise ValueError("feature_series must have shape [windows, channels, features].")
    if series.shape[0] == 0:
        raise ValueError("feature_series must contain at least one sliding window.")

    centers = np.asarray(relative_centers_sec, dtype=np.float32)
    if centers.shape[0] != series.shape[0]:
        raise ValueError("relative_centers_sec length must match number of windows.")

    window_mean = series.mean(axis=0)
    window_std = series.std(axis=0)
    window_max = series.max(axis=0)

    if series.shape[0] <= 1:
        window_slope = np.zeros_like(window_mean, dtype=np.float32)
    else:
        centered_time = centers - float(centers.mean())
        denom = float(np.sum(np.square(centered_time))) + 1e-8
        centered_series = series - window_mean[None, :, :]
        window_slope = np.sum(centered_series * centered_time[:, None, None], axis=0) / denom

    pre_mask = centers < 0.0
    post_mask = centers >= 0.0
    pre_window_mean = series[pre_mask].mean(axis=0) if np.any(pre_mask) else window_mean
    post_window_mean = series[post_mask].mean(axis=0) if np.any(post_mask) else window_mean
    post_minus_pre = post_window_mean - pre_window_mean

    return np.concatenate(
        [
            window_mean,
            window_std,
            window_max,
            window_slope.astype(np.float32, copy=False),
            pre_window_mean,
            post_window_mean,
            post_minus_pre,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def compute_sliding_window_channel_features(
    data: np.ndarray,
    window_segments: Sequence[Dict[str, Any]],
    sfreq: float,
    *,
    spectral_min_freq: float = 1.0,
    spectral_max_freq: float = 150.0,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray]:
    signal_data = np.asarray(data, dtype=np.float32)
    if signal_data.ndim != 2:
        raise ValueError("data must have shape [channels, time].")
    if not window_segments:
        raise ValueError("window_segments must contain at least one sliding window.")

    spectral_series = []
    graph_series = []
    relative_centers = []
    for window_meta in window_segments:
        start_sample = int(window_meta["start_sample"])
        end_sample = int(window_meta["end_sample"])
        window_data = signal_data[:, start_sample:end_sample]
        if window_data.shape[-1] < 2:
            continue

        spectral_series.append(
            compute_spectral_channel_features(
                window_data,
                sfreq=float(sfreq),
                spectral_min_freq=spectral_min_freq,
                spectral_max_freq=spectral_max_freq,
            )
        )
        graph_series.append(
            compute_graph_node_features(
                window_data,
                edge_quantile=edge_quantile,
                min_edge_weight=min_edge_weight,
            )
        )
        relative_centers.append(float(window_meta["relative_center_sec"]))

    if not spectral_series or not graph_series:
        raise ValueError("No valid sliding windows were available for feature extraction.")

    spectral_features = _aggregate_window_feature_series(
        np.stack(spectral_series, axis=0),
        relative_centers,
    )
    graph_features = _aggregate_window_feature_series(
        np.stack(graph_series, axis=0),
        relative_centers,
    )

    expected_spectral_dim = len(DYNAMIC_SPECTRAL_FEATURE_NAMES)
    expected_graph_dim = len(DYNAMIC_GRAPH_FEATURE_NAMES)
    if spectral_features.shape[-1] != expected_spectral_dim:
        raise RuntimeError(f"Expected {expected_spectral_dim} spectral features, got {spectral_features.shape[-1]}.")
    if graph_features.shape[-1] != expected_graph_dim:
        raise RuntimeError(f"Expected {expected_graph_dim} graph features, got {graph_features.shape[-1]}.")
    return spectral_features, graph_features


def compute_window_feature_tensors(
    data: np.ndarray,
    window_segments: Sequence[Dict[str, Any]],
    sfreq: float,
    *,
    spectral_min_freq: float = 1.0,
    spectral_max_freq: float = 150.0,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build per-window node features and adjacency tensors for neural GNN models.

    Returns:
        window_features: [num_windows, channels, spectral_dim + graph_dim]
        window_adjacency: [num_windows, channels, channels]
        relative_centers_sec: [num_windows]
    """

    signal_data = np.asarray(data, dtype=np.float32)
    if signal_data.ndim != 2:
        raise ValueError("data must have shape [channels, time].")
    if not window_segments:
        raise ValueError("window_segments must contain at least one sliding window.")

    feature_series = []
    adjacency_series = []
    relative_centers = []
    for window_meta in window_segments:
        start_sample = int(window_meta["start_sample"])
        end_sample = int(window_meta["end_sample"])
        window_data = signal_data[:, start_sample:end_sample]
        if window_data.shape[-1] < 2:
            continue

        spectral = compute_spectral_channel_features(
            window_data,
            sfreq=float(sfreq),
            spectral_min_freq=spectral_min_freq,
            spectral_max_freq=spectral_max_freq,
        )
        graph_nodes = compute_graph_node_features(
            window_data,
            edge_quantile=edge_quantile,
            min_edge_weight=min_edge_weight,
        )
        adjacency = compute_thresholded_abs_pearson_adjacency(
            window_data,
            edge_quantile=edge_quantile,
            min_edge_weight=min_edge_weight,
        )
        feature_series.append(np.concatenate([spectral, graph_nodes], axis=-1).astype(np.float32, copy=False))
        adjacency_series.append(adjacency.astype(np.float32, copy=False))
        relative_centers.append(float(window_meta["relative_center_sec"]))

    if not feature_series:
        raise ValueError("No valid sliding windows were available for tensor feature extraction.")

    window_features = np.stack(feature_series, axis=0).astype(np.float32, copy=False)
    window_adjacency = np.stack(adjacency_series, axis=0).astype(np.float32, copy=False)
    relative_centers_sec = np.asarray(relative_centers, dtype=np.float32)

    expected_feature_dim = len(WINDOW_NODE_FEATURE_NAMES)
    if window_features.shape[-1] != expected_feature_dim:
        raise RuntimeError(f"Expected {expected_feature_dim} window node features, got {window_features.shape[-1]}.")
    return window_features, window_adjacency, relative_centers_sec


def _resample_channelwise(data: np.ndarray, target_samples: int) -> np.ndarray:
    if target_samples <= 0:
        raise ValueError("target_samples must be positive.")

    array = np.asarray(data, dtype=np.float32)
    if array.shape[-1] == target_samples:
        return array.astype(np.float32, copy=False)
    if array.shape[-1] <= 1:
        return np.repeat(array, target_samples, axis=-1).astype(np.float32, copy=False)
    return signal.resample(array, target_samples, axis=-1).astype(np.float32, copy=False)


def _build_padded_raw_waveform(
    segment_data: np.ndarray,
    *,
    used_duration_sec: float,
    target_duration_sec: float,
    target_sfreq: float,
) -> Tuple[np.ndarray, int]:
    target_total_samples = max(1, int(round(float(target_duration_sec) * float(target_sfreq))))
    valid_target_samples = int(round(float(used_duration_sec) * float(target_sfreq)))
    valid_target_samples = max(1, min(target_total_samples, valid_target_samples))

    resampled = _resample_channelwise(segment_data, valid_target_samples)
    raw_waveform = np.zeros((segment_data.shape[0], target_total_samples), dtype=np.float32)
    raw_waveform[:, :valid_target_samples] = resampled
    return raw_waveform, valid_target_samples


def _build_centered_raw_waveform(
    segment_data: np.ndarray,
    *,
    used_duration_sec: float,
    target_duration_sec: float,
    target_sfreq: float,
    segment_start_sec: float,
    center_sec: float,
) -> Tuple[np.ndarray, int, int]:
    target_total_samples = max(1, int(round(float(target_duration_sec) * float(target_sfreq))))
    valid_target_samples = int(round(float(used_duration_sec) * float(target_sfreq)))
    valid_target_samples = max(1, min(target_total_samples, valid_target_samples))

    resampled = _resample_channelwise(segment_data, valid_target_samples)
    raw_waveform = np.zeros((segment_data.shape[0], target_total_samples), dtype=np.float32)
    center_offset_sec = float(target_duration_sec) / 2.0
    target_start_sample = int(round((float(segment_start_sec) - float(center_sec) + center_offset_sec) * float(target_sfreq)))

    src_start = max(0, -target_start_sample)
    dst_start = max(0, target_start_sample)
    available = min(valid_target_samples - src_start, target_total_samples - dst_start)
    if available > 0:
        raw_waveform[:, dst_start : dst_start + available] = resampled[:, src_start : src_start + available]
    return raw_waveform, int(max(available, 0)), int(dst_start)


def extract_run_feature_record(
    run_info: Dict[str, Any],
    *,
    target_sfreq: float = 1000.0,
    feature_scales_sec: Iterable[float] = DEFAULT_FEATURE_SCALES_SEC,
    prepost_context_sec: float = 30.0,
    raw_temporal_duration_sec: float = 60.0,
    raw_temporal_sfreq: float = 1000.0,
    ez_definition: str = "soz_or_resected",
    spectral_min_freq: float = 10.0,
    spectral_max_freq: float = 300.0,
    graph_edge_quantile: float = 0.70,
    graph_min_edge_weight: float = 0.10,
    sliding_window_sec: float = 2.0,
    sliding_step_sec: float = 1.0,
    extract_static_features: bool = False,
    extract_window_tensors: bool = True,
) -> Optional[RunFeatureRecord]:
    raw = None
    try:
        edf_path = run_info.get("edf_path")
        channels_path = run_info.get("channels_path")
        if not edf_path or not channels_path or not has_valid_ictal_bounds(run_info):
            return None

        raw, picked_channels_norm, data, _ = load_and_preprocess_edf(
            edf_path,
            channels_path,
            target_sfreq=target_sfreq,
            bandpass_low=spectral_min_freq,
            bandpass_high=spectral_max_freq,
        )

        contacts_meta = parse_channel_labels(channels_path, ez_definition=ez_definition)
        contacts_meta = _filter_contacts_to_preprocessed_channels(contacts_meta, picked_channels_norm)
        if contacts_meta.empty:
            return None

        reorder_idx = contacts_meta["picked_order"].to_numpy(dtype=int)
        data = data[reorder_idx]

        sample_info = create_sliding_onset_ictal_sample(
            raw,
            data,
            run_info,
            context_duration_sec=float(prepost_context_sec),
            window_duration_sec=float(sliding_window_sec),
            window_step_sec=float(sliding_step_sec),
            raw_temporal_sfreq=float(raw_temporal_sfreq),
        )
        if sample_info is None or bool(sample_info["unusable_mask"]):
            return None

        if extract_static_features:
            spectral_features, graph_features = compute_sliding_window_channel_features(
                data,
                sample_info["feature_segments"],
                sfreq=float(raw.info["sfreq"]),
                spectral_min_freq=spectral_min_freq,
                spectral_max_freq=spectral_max_freq,
                edge_quantile=graph_edge_quantile,
                min_edge_weight=graph_min_edge_weight,
            )
        else:
            spectral_features = np.zeros((data.shape[0], 0), dtype=np.float32)
            graph_features = np.zeros((data.shape[0], 0), dtype=np.float32)

        if extract_window_tensors:
            window_features, window_adjacency, window_relative_centers_sec = compute_window_feature_tensors(
                data,
                sample_info["feature_segments"],
                sfreq=float(raw.info["sfreq"]),
                spectral_min_freq=spectral_min_freq,
                spectral_max_freq=spectral_max_freq,
                edge_quantile=graph_edge_quantile,
                min_edge_weight=graph_min_edge_weight,
            )
        else:
            window_features = np.zeros((0, data.shape[0], len(WINDOW_NODE_FEATURE_NAMES)), dtype=np.float32)
            window_adjacency = np.zeros((0, data.shape[0], data.shape[0]), dtype=np.float32)
            window_relative_centers_sec = np.zeros((0,), dtype=np.float32)

        raw_segment = sample_info["raw_segment"]
        raw_segment_data = data[:, int(raw_segment["start_sample"]) : int(raw_segment["end_sample"])]
        if raw_segment_data.shape[-1] < 2:
            return None
        raw_used_duration_sec = float(raw_segment["used_duration_sec"])
        raw_waveform, raw_valid_samples, raw_valid_start_sample = _build_centered_raw_waveform(
            raw_segment_data,
            used_duration_sec=raw_used_duration_sec,
            target_duration_sec=float(prepost_context_sec) * 2.0,
            target_sfreq=float(raw_temporal_sfreq),
            segment_start_sec=float(raw_segment["start_sec"]),
            center_sec=float(sample_info["seizure_onset_sec"]),
        )

        sample = {
            "sample_id": str(sample_info["sample_id"]),
            "analysis_phase": "onset_sliding",
            "start_sec": float(sample_info["analysis_segment"]["start_sec"]),
            "end_sec": float(sample_info["analysis_segment"]["end_sec"]),
            "seizure_onset_sec": float(sample_info["seizure_onset_sec"]),
            "seizure_offset_sec": float(sample_info["seizure_offset_sec"]),
            "ictal_duration_total_sec": float(sample_info["ictal_duration_total_sec"]),
            "ictal_duration_used_sec": float(sample_info["ictal_duration_used_sec"]),
            "bad_segment_ratio": float(sample_info["bad_segment_ratio"]),
            "bad_channel_ratio": float(sample_info["bad_channel_ratio"]),
            "artifact_subsegments": int(sample_info["artifact_subsegments"]),
            "feature_scales_sec": [float(sliding_window_sec)],
            "feature_scale_used_secs": [float(window["used_duration_sec"]) for window in sample_info["feature_segments"]],
            "raw_temporal_duration_sec": float(prepost_context_sec) * 2.0,
            "raw_temporal_sfreq": float(raw_temporal_sfreq),
            "raw_valid_samples": int(raw_valid_samples),
            "raw_valid_start_sample": int(raw_valid_start_sample),
            "spectral_features": spectral_features,
            "graph_features": graph_features,
            "window_features": window_features,
            "window_adjacency": window_adjacency,
            "window_relative_centers_sec": window_relative_centers_sec,
            "raw_waveform": raw_waveform,
        }

        metadata = {
            key: value
            for key, value in dict(run_info).items()
            if key not in {"edf_path", "channels_path", "events_path", "json_path"}
        }

        return RunFeatureRecord(
            subject_id=str(run_info.get("subject_id", "unknown")),
            run_id=str(run_info.get("run_id", "unknown")),
            task=str(run_info.get("task", "unknown")),
            phase_group=str(run_info.get("phase_group", "unknown")),
            channel_names_norm=contacts_meta["channel_name_norm"].tolist(),
            contact_groups=contacts_meta["contact_group"].fillna("").astype(str).tolist(),
            contact_numbers=[
                None if pd.isna(value) else int(value) for value in contacts_meta["contact_number"].tolist()
            ],
            labels=contacts_meta["is_ez"].to_numpy(dtype=np.float32, copy=True),
            sfreq=float(raw.info["sfreq"]),
            sample=sample,
            metadata=metadata,
        )
    finally:
        if raw is not None and hasattr(raw, "close"):
            raw.close()


__all__ = [
    "BASE_SPECTRAL_FEATURE_NAMES",
    "DEFAULT_FEATURE_SCALES_SEC",
    "FEATURE_DIM",
    "DYNAMIC_SPECTRAL_FEATURE_NAMES",
    "DYNAMIC_GRAPH_FEATURE_NAMES",
    "DYNAMIC_GRAPH_STAT_NAMES",
    "GRAPH_FEATURE_NAMES",
    "WINDOW_NODE_FEATURE_NAMES",
    "RunFeatureRecord",
    "SLIDING_WINDOW_SUMMARY_NAMES",
    "SPECTRAL_BANDS",
    "build_multiscale_feature_names",
    "compute_spectral_channel_features",
    "compute_dynamic_graph_channel_features",
    "compute_sliding_window_channel_features",
    "compute_window_feature_tensors",
    "extract_run_feature_record",
    "infer_feature_dim",
    "parse_duration_list",
]
