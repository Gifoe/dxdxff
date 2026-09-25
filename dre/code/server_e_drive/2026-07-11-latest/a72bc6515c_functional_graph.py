from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
from scipy import signal, stats


NETWORK_PHASES = ("preictal", "onset", "spread")
PHASES = NETWORK_PHASES
NON_BRAIN_PATTERN = re.compile(r"^(?:EKG|ECG|EMG|EOG|RESP|TRIG|DC|PULSE)", re.IGNORECASE)


@dataclass(frozen=True)
class GraphConfig:
    graph_source: str = "raw_aec_spearman"
    band_low: float = 30.0
    band_high: float = 80.0
    topk: int = 8
    envelope_sfreq: float = 32.0
    min_phase_sec: float = 1.0
    min_channels: int = 4
    min_envelope_samples: int = 64

    def preprocessing_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def normalize_channel_name(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "").upper())
    match = re.fullmatch(r"([A-Z]+)0*(\d+)(.*)", text)
    if match:
        text = f"{match.group(1)}{int(match.group(2)):02d}{match.group(3)}"
    return text


def is_brain_channel(name: Any, metadata: dict[str, Any] | None = None) -> bool:
    normalized = normalize_channel_name(name)
    if not normalized or NON_BRAIN_PATTERN.match(normalized):
        return False
    meta = metadata or {}
    status = str(meta.get("status", "good") or "good").strip().lower()
    if status not in {"", "good"}:
        return False
    if bool(meta.get("bad", meta.get("is_bad", False))):
        return False
    return True


def infer_onset_sample(sample: dict[str, Any], n_samples: int, sfreq: float) -> tuple[int | None, str]:
    for key in ("raw_onset_sample", "seizure_onset_sample", "onset_sample"):
        value = sample.get(key)
        if value is not None and np.isfinite(float(value)):
            index = int(round(float(value)))
            return (index, key) if 0 <= index < n_samples else (None, f"{key}_out_of_range")
    onset = sample.get("seizure_onset_sec")
    start = sample.get("start_sec")
    if onset is not None and start is not None:
        index = int(round((float(onset) - float(start)) * sfreq))
        return (index, "seizure_onset_sec_minus_start_sec") if 0 <= index < n_samples else (None, "derived_onset_out_of_range")
    origin = str(sample.get("raw_time_origin", sample.get("time_origin", ""))).lower()
    if origin in {"onset_at_midpoint", "seizure_onset_at_raw_target_midpoint"}:
        return n_samples // 2, "declared_midpoint"
    return None, "unknown_raw_time_origin"


def phase_bounds(onset_sample: int, n_samples: int, sfreq: float) -> dict[str, tuple[int, int]]:
    def clip(seconds: float) -> int:
        return int(np.clip(round(onset_sample + seconds * sfreq), 0, n_samples))

    return {
        "preictal": (clip(-30.0), clip(0.0)),
        "onset": (clip(0.0), clip(10.0)),
        "spread": (clip(10.0), clip(30.0)),
    }


def sparsify_topk(adjacency: np.ndarray, topk: int) -> np.ndarray:
    matrix = np.asarray(adjacency, dtype=np.float64).copy()
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("adjacency must be square")
    np.fill_diagonal(matrix, 0.0)
    n_nodes = matrix.shape[0]
    if n_nodes <= 1:
        return np.zeros_like(matrix, dtype=np.float32)
    k = min(max(int(topk), 0), n_nodes - 1)
    if k == 0:
        return np.zeros_like(matrix, dtype=np.float32)
    directed = np.zeros_like(matrix)
    for row in range(n_nodes):
        order = np.argsort(matrix[row], kind="stable")[-k:]
        positive = order[matrix[row, order] > 0.0]
        directed[row, positive] = matrix[row, positive]
    output = np.maximum(directed, directed.T)
    np.fill_diagonal(output, 0.0)
    return output.astype(np.float32, copy=False)


def _robust_z(values: np.ndarray) -> np.ndarray:
    median = np.nanmedian(values, axis=-1, keepdims=True)
    mad = np.nanmedian(np.abs(values - median), axis=-1, keepdims=True)
    scale = 1.4826 * mad
    std = np.nanstd(values, axis=-1, keepdims=True)
    scale = np.where(scale >= 1e-5, scale, std)
    scale = np.where(scale >= 1e-5, scale, 1.0)
    return np.nan_to_num((values - median) / scale)


def build_phase_graph(
    raw: np.ndarray,
    sfreq: float,
    channel_names: Sequence[str],
    *,
    channel_valid: np.ndarray | None = None,
    config: GraphConfig | None = None,
) -> dict[str, Any]:
    cfg = config or GraphConfig()
    data = np.asarray(raw, dtype=np.float64)
    if data.ndim != 2:
        return _invalid_graph(channel_names, "raw_not_channel_time")
    n_channels, n_times = data.shape
    valid = np.ones(n_channels, dtype=bool) if channel_valid is None else np.asarray(channel_valid, dtype=bool).copy()
    valid &= np.asarray([is_brain_channel(name) for name in channel_names], dtype=bool)
    valid &= np.isfinite(data).sum(axis=1) >= max(4, int(round(cfg.min_phase_sec * sfreq)))
    if n_times < int(round(cfg.min_phase_sec * sfreq)):
        return _invalid_graph(channel_names, "phase_shorter_than_minimum", valid)
    indices = np.where(valid)[0]
    if indices.size < cfg.min_channels:
        return _invalid_graph(channel_names, "fewer_than_four_valid_brain_channels", valid)
    high = min(float(cfg.band_high), 0.40 * float(sfreq))
    if high <= 35.0 or float(cfg.band_low) >= high:
        return _invalid_graph(channel_names, "nyquist_insufficient_for_30_80hz", valid)
    selected = np.nan_to_num(data[indices])
    sos = signal.butter(4, [float(cfg.band_low), high], btype="bandpass", fs=float(sfreq), output="sos")
    try:
        filtered = signal.sosfiltfilt(sos, selected, axis=-1)
    except ValueError:
        return _invalid_graph(channel_names, "phase_too_short_for_zero_phase_filter", valid)
    envelope = np.log1p(np.abs(signal.hilbert(filtered, axis=-1)))
    target_sfreq = min(float(cfg.envelope_sfreq), float(sfreq))
    if target_sfreq < sfreq:
        target_n = max(1, int(np.floor(n_times * target_sfreq / sfreq)))
        envelope = signal.resample(envelope, target_n, axis=-1)
    if envelope.shape[-1] < cfg.min_envelope_samples:
        return _invalid_graph(channel_names, "fewer_than_64_envelope_samples", valid)
    envelope = _robust_z(envelope)
    ranked = np.stack([stats.rankdata(row, method="average") for row in envelope], axis=0)
    correlation = np.corrcoef(ranked)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    correlation = np.maximum(correlation, 0.0)
    np.fill_diagonal(correlation, 0.0)
    sparse_local = sparsify_topk(correlation, cfg.topk)
    adjacency = np.zeros((n_channels, n_channels), dtype=np.float32)
    adjacency[np.ix_(indices, indices)] = sparse_local
    edge_rows, edge_cols = np.where(np.triu(adjacency, k=1) > 0.0)
    edge_weight = adjacency[edge_rows, edge_cols]
    return {
        "adjacency": adjacency,
        "channel_names": [normalize_channel_name(name) for name in channel_names],
        "valid_channel_mask": valid,
        "edge_index": np.stack((edge_rows, edge_cols), axis=0).astype(np.int64),
        "edge_weight": edge_weight.astype(np.float32),
        "n_nodes": int(valid.sum()),
        "n_edges": int(edge_weight.size),
        "graph_valid": bool(edge_weight.size > 0),
        "invalid_reason": "" if edge_weight.size else "no_positive_edges",
        "effective_band_high": high,
        "envelope_samples": int(envelope.shape[-1]),
    }


def _invalid_graph(channel_names: Sequence[str], reason: str, valid: np.ndarray | None = None) -> dict[str, Any]:
    n_channels = len(channel_names)
    mask = np.zeros(n_channels, dtype=bool) if valid is None else np.asarray(valid, dtype=bool)
    return {
        "adjacency": np.zeros((n_channels, n_channels), dtype=np.float32),
        "channel_names": [normalize_channel_name(name) for name in channel_names],
        "valid_channel_mask": mask,
        "edge_index": np.zeros((2, 0), dtype=np.int64),
        "edge_weight": np.zeros((0,), dtype=np.float32),
        "n_nodes": int(mask.sum()),
        "n_edges": 0,
        "graph_valid": False,
        "invalid_reason": str(reason),
    }


__all__ = [
    "GraphConfig",
    "NETWORK_PHASES",
    "PHASES",
    "build_phase_graph",
    "infer_onset_sample",
    "is_brain_channel",
    "normalize_channel_name",
    "phase_bounds",
    "sparsify_topk",
]
