from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import welch

from graph_channel import GRAPH_FEATURE_NAMES, compute_graph_node_features
from module2_preprocessing import load_and_preprocess_edf
from module3_labels_metadata import parse_channel_labels
from module4_time_windows import create_multiscale_ictal_sample, has_valid_ictal_bounds


DEFAULT_FEATURE_SCALES_SEC: Sequence[float] = (5.0, 10.0, 20.0)

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
    return tuple(f"{base_name}@{scale_sec:g}s" for scale_sec in scales_sec for base_name in base_names)


def infer_feature_dim(feature_scales_sec: Sequence[float]) -> int:
    scales = parse_duration_list(feature_scales_sec)
    return len(scales) * (len(BASE_SPECTRAL_FEATURE_NAMES) + len(GRAPH_FEATURE_NAMES))


FEATURE_DIM = infer_feature_dim(DEFAULT_FEATURE_SCALES_SEC)


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
                "spectral_features": np.asarray(self.sample["spectral_features"], dtype=np.float32),
                "graph_features": np.asarray(self.sample["graph_features"], dtype=np.float32),
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
    trapz_fn = getattr(np, "trapezoid", getattr(np, "trapz", None))
    
    for _, low, high in SPECTRAL_BANDS:
        band_mask = (restricted_freqs >= float(low)) & (restricted_freqs < float(high))
        if np.any(band_mask):
            band_power = trapz_fn(restricted_psd[:, band_mask], restricted_freqs[band_mask], axis=-1)
        else:
            band_power = np.zeros(num_channels, dtype=np.float32)
        band_features.append(np.log1p(band_power).astype(np.float32, copy=False))

    total_power = trapz_fn(restricted_psd, restricted_freqs, axis=-1).astype(np.float32, copy=False)
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


def extract_run_feature_record(
    run_info: Dict[str, Any],
    *,
    target_sfreq: float = 512.0,
    feature_scales_sec: Iterable[float] = DEFAULT_FEATURE_SCALES_SEC,
    raw_temporal_duration_sec: float = 5.0,
    raw_temporal_sfreq: float = 256.0,
    ez_definition: str = "soz_or_resected",
    spectral_min_freq: float = 1.0,
    spectral_max_freq: float = 150.0,
    graph_edge_quantile: float = 0.70,
    graph_min_edge_weight: float = 0.10,
) -> Optional[RunFeatureRecord]:
    raw = None
    try:
        edf_path = run_info.get("edf_path")
        channels_path = run_info.get("channels_path")
        if not edf_path or not channels_path or not has_valid_ictal_bounds(run_info):
            return None

        feature_scales = parse_duration_list(feature_scales_sec)

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

        sample_info = create_multiscale_ictal_sample(
            raw,
            data,
            run_info,
            feature_durations_sec=feature_scales,
            raw_temporal_duration_sec=float(raw_temporal_duration_sec),
        )
        if sample_info is None or bool(sample_info["unusable_mask"]):
            return None

        spectral_blocks: List[np.ndarray] = []
        graph_blocks: List[np.ndarray] = []
        feature_scale_used_secs: List[float] = []
        for segment_meta in sample_info["feature_segments"]:
            start_sample = int(segment_meta["start_sample"])
            end_sample = int(segment_meta["end_sample"])
            segment_data = data[:, start_sample:end_sample]
            if segment_data.shape[-1] < 2:
                return None

            spectral_blocks.append(
                compute_spectral_channel_features(
                    segment_data,
                    sfreq=float(raw.info["sfreq"]),
                    spectral_min_freq=spectral_min_freq,
                    spectral_max_freq=spectral_max_freq,
                )
            )
            graph_blocks.append(
                compute_graph_node_features(
                    segment_data,
                    edge_quantile=graph_edge_quantile,
                    min_edge_weight=graph_min_edge_weight,
                )
            )
            feature_scale_used_secs.append(float(segment_meta["used_duration_sec"]))

        spectral_features = np.concatenate(spectral_blocks, axis=-1).astype(np.float32, copy=False)
        graph_features = np.concatenate(graph_blocks, axis=-1).astype(np.float32, copy=False)

        raw_segment = sample_info["raw_segment"]
        raw_segment_data = data[:, int(raw_segment["start_sample"]) : int(raw_segment["end_sample"])]
        if raw_segment_data.shape[-1] < 2:
            return None
        raw_waveform, raw_valid_samples = _build_padded_raw_waveform(
            raw_segment_data,
            used_duration_sec=float(raw_segment["used_duration_sec"]),
            target_duration_sec=float(raw_temporal_duration_sec),
            target_sfreq=float(raw_temporal_sfreq),
        )

        sample = {
            "sample_id": str(sample_info["sample_id"]),
            "analysis_phase": "ictal",
            "start_sec": float(sample_info["feature_segments"][-1]["start_sec"]),
            "end_sec": float(sample_info["feature_segments"][-1]["end_sec"]),
            "seizure_onset_sec": float(sample_info["seizure_onset_sec"]),
            "seizure_offset_sec": float(sample_info["seizure_offset_sec"]),
            "ictal_duration_total_sec": float(sample_info["ictal_duration_total_sec"]),
            "ictal_duration_used_sec": float(sample_info["ictal_duration_used_sec"]),
            "bad_segment_ratio": float(sample_info["bad_segment_ratio"]),
            "bad_channel_ratio": float(sample_info["bad_channel_ratio"]),
            "artifact_subsegments": int(sample_info["artifact_subsegments"]),
            "feature_scales_sec": list(feature_scales),
            "feature_scale_used_secs": feature_scale_used_secs,
            "raw_temporal_duration_sec": float(raw_temporal_duration_sec),
            "raw_temporal_sfreq": float(raw_temporal_sfreq),
            "raw_valid_samples": int(raw_valid_samples),
            "spectral_features": spectral_features,
            "graph_features": graph_features,
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
    "GRAPH_FEATURE_NAMES",
    "RunFeatureRecord",
    "SPECTRAL_BANDS",
    "build_multiscale_feature_names",
    "compute_spectral_channel_features",
    "extract_run_feature_record",
    "infer_feature_dim",
    "parse_duration_list",
]
