from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

try:
    from ez_features import BASE_SPECTRAL_FEATURE_NAMES, WINDOW_NODE_FEATURE_NAMES
except Exception:
    BASE_SPECTRAL_FEATURE_NAMES = (
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
    WINDOW_NODE_FEATURE_NAMES = tuple(BASE_SPECTRAL_FEATURE_NAMES) + (
        "degree_norm",
        "strength_norm",
        "clustering_coeff",
        "eigenvector_centrality",
        "pagerank",
        "kcore_norm",
        "local_efficiency",
    )


def _normalize_raw_waveform(
    raw_waveform: np.ndarray,
    raw_valid_samples: int,
    *,
    raw_valid_start_sample: int = 0,
    args: Any | None = None,
) -> np.ndarray:
    raw = np.asarray(raw_waveform, dtype=np.float32).copy()
    if raw.ndim != 2:
        raise ValueError("raw_waveform must have shape [channels, time].")

    valid_start = max(0, min(int(raw_valid_start_sample), int(raw.shape[-1])))
    valid_samples = max(1, min(int(raw_valid_samples), int(raw.shape[-1]) - valid_start))
    valid_end = min(int(raw.shape[-1]), valid_start + valid_samples)
    valid_segment = raw[:, valid_start:valid_end]
    raw_view = str(getattr(args, "raw_view", "baseline_normalized") if args is not None else "baseline_normalized").lower()
    if raw_view == "baseline_normalized":
        onset_sample = int(raw.shape[-1] // 2)
        baseline_end = min(valid_end, max(valid_start + 1, onset_sample))
        baseline = raw[:, valid_start:baseline_end]
        if baseline.shape[-1] < 2:
            baseline = valid_segment
    else:
        baseline = valid_segment
    channel_mean = baseline.mean(axis=-1, keepdims=True)
    channel_std = np.clip(baseline.std(axis=-1, keepdims=True), 1e-5, None)
    raw[:, valid_start:valid_end] = (valid_segment - channel_mean) / channel_std
    if valid_start > 0:
        raw[:, :valid_start] = 0.0
    if valid_end < raw.shape[-1]:
        raw[:, valid_end:] = 0.0
    return raw.astype(np.float32, copy=False)


def _as_window_tensors(sample: Dict[str, Any], feature_dim_fallback: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_features = np.asarray(sample.get("window_features", np.zeros((0, 0, 0))), dtype=np.float32)
    window_adjacency = np.asarray(sample.get("window_adjacency", np.zeros((0, 0, 0))), dtype=np.float32)
    window_centers = np.asarray(sample.get("window_relative_centers_sec", np.zeros((0,))), dtype=np.float32)

    if window_features.ndim == 3 and window_features.shape[0] > 0 and window_features.shape[-1] > 0:
        if window_adjacency.ndim != 3 or window_adjacency.shape[0] != window_features.shape[0]:
            num_windows, num_channels = int(window_features.shape[0]), int(window_features.shape[1])
            window_adjacency = np.zeros((num_windows, num_channels, num_channels), dtype=np.float32)
        if window_centers.shape[0] != window_features.shape[0]:
            window_centers = np.arange(window_features.shape[0], dtype=np.float32)
        return window_features, window_adjacency.astype(np.float32, copy=False), window_centers.astype(np.float32, copy=False)

    spectral = np.asarray(sample.get("spectral_features", np.zeros((0, 0))), dtype=np.float32)
    graph = np.asarray(sample.get("graph_features", np.zeros((0, 0))), dtype=np.float32)
    num_channels = int(np.asarray(sample["labels"]).shape[0])
    if spectral.ndim == 2 and graph.ndim == 2 and spectral.shape[0] == num_channels and graph.shape[0] == num_channels and (spectral.shape[1] + graph.shape[1]) > 0:
        features = np.concatenate([spectral, graph], axis=-1)[None, :, :].astype(np.float32, copy=False)
    else:
        features = np.zeros((1, num_channels, max(1, int(feature_dim_fallback))), dtype=np.float32)
    adjacency = np.zeros((features.shape[0], num_channels, num_channels), dtype=np.float32)
    centers = np.zeros((features.shape[0],), dtype=np.float32)
    return features, adjacency, centers


def _get_bool(args: Any | None, name: str, default: bool) -> bool:
    if args is None:
        return bool(default)
    value = getattr(args, name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _baseline_window_mask(window_centers: np.ndarray, num_windows: int) -> np.ndarray:
    centers = np.asarray(window_centers, dtype=np.float32)
    pre_mask = centers < 0.0
    if centers.shape[0] != num_windows or not np.any(pre_mask):
        pre_mask = np.ones((num_windows,), dtype=bool)
    return pre_mask


def _sample_interictal_baseline(sample: Dict[str, Any], feature_shape: tuple[int, int]) -> np.ndarray | None:
    for key in ("interictal_window_features", "interictal_baseline_features", "baseline_window_features"):
        value = sample.get(key)
        if value is None:
            continue
        baseline = np.asarray(value, dtype=np.float32)
        if baseline.ndim == 3 and baseline.shape[1:] == feature_shape:
            return baseline
    return None


def _robust_baseline_stats(baseline: np.ndarray, eps: float) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(baseline, dtype=np.float32)
    median = np.median(values, axis=0, keepdims=True).astype(np.float32, copy=False)
    mad = np.median(np.abs(values - median), axis=0, keepdims=True).astype(np.float32, copy=False)
    robust_std = np.clip(1.4826 * mad, eps, None)
    return median, robust_std


def _percentile_against_baseline(features: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    bank = np.asarray(baseline, dtype=np.float32)
    if bank.ndim != 3 or bank.shape[1:] != values.shape[1:] or bank.shape[0] == 0:
        return np.zeros_like(values, dtype=np.float32)
    out = np.zeros_like(values, dtype=np.float32)
    for t_idx in range(values.shape[0]):
        out[t_idx] = np.mean(bank <= values[t_idx : t_idx + 1], axis=0)
    return out.astype(np.float32, copy=False)


def _self_comparison_features(window_features: np.ndarray, window_centers: np.ndarray, args: Any | None) -> np.ndarray:
    features = np.asarray(window_features, dtype=np.float32)
    feature_view = str(getattr(args, "feature_view", "self_comparison") if args is not None else "self_comparison").lower()
    if feature_view in {"absolute", "raw", "none"}:
        return features.astype(np.float32, copy=False)
    if feature_view != "self_comparison":
        raise ValueError(f"Unsupported feature_view={feature_view!r}.")

    centers = np.asarray(window_centers, dtype=np.float32)
    pre_mask = centers < 0.0
    if centers.shape[0] != features.shape[0] or not np.any(pre_mask):
        pre_mask = np.ones((features.shape[0],), dtype=bool)

    eps = float(getattr(args, "self_compare_eps", 1e-5) if args is not None else 1e-5)
    baseline = features[pre_mask]
    pre_mean = baseline.mean(axis=0, keepdims=True)
    pre_std = np.clip(baseline.std(axis=0, keepdims=True), eps, None)
    pre_mean_t = np.broadcast_to(pre_mean, features.shape)
    delta = features - pre_mean_t
    zdelta = delta / pre_std
    ratio = np.log((np.abs(features) + eps) / (np.abs(pre_mean_t) + eps))

    parts: List[np.ndarray] = []
    if _get_bool(args, "self_compare_include_abs", True):
        parts.append(features)
    if _get_bool(args, "self_compare_include_pre_mean", False):
        parts.append(pre_mean_t.astype(np.float32, copy=False))
    if _get_bool(args, "self_compare_include_delta", True):
        parts.append(delta.astype(np.float32, copy=False))
    if _get_bool(args, "self_compare_include_zdelta", True):
        parts.append(zdelta.astype(np.float32, copy=False))
    if _get_bool(args, "self_compare_include_ratio", True):
        parts.append(ratio.astype(np.float32, copy=False))
    if _get_bool(args, "self_compare_include_channel_rank", False):
        rank = np.zeros_like(delta, dtype=np.float32)
        if delta.shape[1] > 1:
            order = np.argsort(delta, axis=1)
            rank_values = np.argsort(order, axis=1).astype(np.float32) / float(delta.shape[1] - 1)
            rank = rank_values.astype(np.float32, copy=False)
        parts.append(rank)
    if not parts:
        parts.append(features)
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def _interictal_comparison_features(
    window_features: np.ndarray,
    window_centers: np.ndarray,
    sample: Dict[str, Any],
    args: Any | None,
) -> np.ndarray:
    features = np.asarray(window_features, dtype=np.float32)
    feature_view = str(getattr(args, "feature_view", "self_comparison") if args is not None else "self_comparison").lower()
    if feature_view not in {"interictal_comparison", "dual_state_comparison"}:
        return features.astype(np.float32, copy=False)

    eps = float(getattr(args, "self_compare_eps", 1e-5) if args is not None else 1e-5)
    interictal_min_windows = max(1, int(getattr(args, "interictal_min_windows", 10) if args is not None else 10))
    inter_baseline = _sample_interictal_baseline(sample, tuple(features.shape[1:]))
    has_real_interictal = inter_baseline is not None and inter_baseline.shape[0] >= interictal_min_windows
    if not has_real_interictal:
        pre_mask = _baseline_window_mask(window_centers, features.shape[0])
        inter_baseline = features[pre_mask]

    baseline_median, baseline_std = _robust_baseline_stats(inter_baseline, eps)
    baseline_t = np.broadcast_to(baseline_median, features.shape)
    delta = features - baseline_t
    zdelta = delta / baseline_std
    ratio = np.log((np.abs(features) + eps) / (np.abs(baseline_t) + eps))
    percentile = _percentile_against_baseline(features, inter_baseline)

    parts: List[np.ndarray] = []
    if _get_bool(args, "include_inter_abs", True):
        parts.append(features)
    if _get_bool(args, "include_inter_delta", True):
        parts.append(delta.astype(np.float32, copy=False))
    if _get_bool(args, "include_inter_zdelta", True):
        parts.append(zdelta.astype(np.float32, copy=False))
    if _get_bool(args, "include_inter_ratio", True):
        parts.append(ratio.astype(np.float32, copy=False))
    if _get_bool(args, "include_inter_percentile", True):
        parts.append(percentile.astype(np.float32, copy=False))
    if _get_bool(args, "interictal_missing_indicators", True):
        missing_value = 0.0 if has_real_interictal else 1.0
        parts.append(np.full((*features.shape[:2], 1), missing_value, dtype=np.float32))

    if feature_view == "dual_state_comparison" and _get_bool(args, "include_pre_comparison", True):
        pre_args = copy.copy(args) if args is not None else None
        if pre_args is not None:
            setattr(pre_args, "feature_view", "self_comparison")
        pre_features = _self_comparison_features(features, window_centers, pre_args)
        base_dim = int(features.shape[-1])
        if _get_bool(args, "self_compare_include_abs", True) and pre_features.shape[-1] >= base_dim:
            pre_features = pre_features[..., base_dim:]
        if pre_features.shape[-1] > 0:
            parts.append(pre_features.astype(np.float32, copy=False))

    if not parts:
        parts.append(features)
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def _append_patient_relative_features(features: np.ndarray, args: Any | None) -> np.ndarray:
    if not _get_bool(args, "include_patient_relative", False):
        return np.asarray(features, dtype=np.float32)
    values = np.asarray(features, dtype=np.float32)
    if values.ndim != 3 or values.shape[1] <= 1:
        return values
    parts = [values]
    if _get_bool(args, "include_patient_relative_z", True):
        mean = values.mean(axis=1, keepdims=True)
        std = np.clip(values.std(axis=1, keepdims=True), 1e-5, None)
        parts.append(((values - mean) / std).astype(np.float32, copy=False))
    if _get_bool(args, "include_patient_relative_rank", True):
        order = np.argsort(values, axis=1)
        rank = np.argsort(order, axis=1).astype(np.float32)
        rank /= float(max(values.shape[1] - 1, 1))
        parts.append(rank.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=-1).astype(np.float32, copy=False)


def _mixed_delta_adjacency(window_adjacency: np.ndarray, window_centers: np.ndarray, args: Any | None) -> np.ndarray:
    adjacency = np.asarray(window_adjacency, dtype=np.float32)
    adjacency_view = str(getattr(args, "adjacency_view", "mixed_abs_delta") if args is not None else "mixed_abs_delta").lower()
    if adjacency_view in {"absolute", "abs", "raw", "none"}:
        return adjacency.astype(np.float32, copy=False)
    if adjacency_view not in {"mixed_abs_delta", "mixed_interictal_delta", "mixed_dual_delta"}:
        raise ValueError(f"Unsupported adjacency_view={adjacency_view!r}.")
    if adjacency.ndim != 3 or adjacency.shape[0] == 0:
        return adjacency.astype(np.float32, copy=False)

    centers = np.asarray(window_centers, dtype=np.float32)
    pre_mask = _baseline_window_mask(centers, adjacency.shape[0])
    pre_adj = adjacency[pre_mask].mean(axis=0, keepdims=True)
    pre_delta_abs = np.abs(adjacency - pre_adj).astype(np.float32, copy=False)
    # Legacy caches do not contain a true interictal adjacency bank. Use the
    # pre-onset bank as a deterministic fallback so old caches remain runnable.
    inter_delta_abs = pre_delta_abs
    if adjacency_view == "mixed_interictal_delta":
        alpha = float(np.clip(float(getattr(args, "interictal_adjacency_alpha", 0.5) if args is not None else 0.5), 0.0, 1.0))
        mixed = alpha * adjacency + (1.0 - alpha) * inter_delta_abs
    elif adjacency_view == "mixed_dual_delta":
        w_abs = max(0.0, float(getattr(args, "dual_adj_abs_weight", 0.34) if args is not None else 0.34))
        w_inter = max(0.0, float(getattr(args, "dual_adj_inter_weight", 0.33) if args is not None else 0.33))
        w_pre = max(0.0, float(getattr(args, "dual_adj_pre_weight", 0.33) if args is not None else 0.33))
        denom = max(w_abs + w_inter + w_pre, 1e-8)
        mixed = (w_abs * adjacency + w_inter * inter_delta_abs + w_pre * pre_delta_abs) / denom
    else:
        alpha = float(np.clip(float(getattr(args, "delta_adjacency_alpha", 0.5) if args is not None else 0.5), 0.0, 1.0))
        mixed = alpha * adjacency + (1.0 - alpha) * pre_delta_abs
    eye = np.eye(mixed.shape[-1], dtype=bool)[None, :, :]
    mixed = mixed.copy()
    mixed[eye.repeat(mixed.shape[0], axis=0)] = 0.0
    return mixed.astype(np.float32, copy=False)


def _feature_series_by_name(window_features: np.ndarray, feature_name: str) -> np.ndarray | None:
    names = list(WINDOW_NODE_FEATURE_NAMES)
    if feature_name not in names:
        return None
    idx = names.index(feature_name)
    features = np.asarray(window_features, dtype=np.float32)
    if features.ndim != 3 or idx >= features.shape[-1]:
        return None
    return features[:, :, idx].astype(np.float32, copy=False)


def _delta_from_pre(series: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros_like(values, dtype=np.float32)
    pre_mask = centers < 0.0
    if centers.shape[0] != values.shape[0] or not np.any(pre_mask):
        baseline = values.mean(axis=0, keepdims=True)
    else:
        baseline = values[pre_mask].mean(axis=0, keepdims=True)
    return (values - baseline).astype(np.float32, copy=False)


def _physics_feature_concat_tensor(
    raw_window_features: np.ndarray,
    raw_window_adjacency: np.ndarray,
    window_centers: np.ndarray,
) -> np.ndarray:
    """Derive lightweight physics-inspired channel features from existing cache tensors.

    This is the first-stage PhysFeat path: it keeps the baseline model unchanged
    and only augments each channel/window feature vector with onset dynamics and
    graph-strength evidence already present in the cached window tensors.
    """

    features = np.asarray(raw_window_features, dtype=np.float32)
    if features.ndim != 3 or features.shape[0] == 0:
        return np.zeros((*features.shape[:2], 0), dtype=np.float32) if features.ndim >= 2 else np.zeros((0, 0, 0), dtype=np.float32)
    t, c, _ = features.shape
    centers = np.asarray(window_centers, dtype=np.float32)
    if centers.shape[0] != t:
        centers = np.arange(t, dtype=np.float32)

    zeros = np.zeros((t, c), dtype=np.float32)
    high_gamma = _feature_series_by_name(features, "log_bp_high_gamma")
    line_length = _feature_series_by_name(features, "line_length_per_sec")
    entropy = _feature_series_by_name(features, "spectral_entropy")
    rms = _feature_series_by_name(features, "rms")
    delta_band = _feature_series_by_name(features, "log_bp_delta")
    theta_band = _feature_series_by_name(features, "log_bp_theta")
    strength_node = _feature_series_by_name(features, "strength_norm")
    degree_node = _feature_series_by_name(features, "degree_norm")

    high_gamma = high_gamma if high_gamma is not None else zeros
    line_length = line_length if line_length is not None else zeros
    entropy = entropy if entropy is not None else zeros
    rms = rms if rms is not None else zeros
    delta_band = delta_band if delta_band is not None else zeros
    theta_band = theta_band if theta_band is not None else zeros
    strength_node = strength_node if strength_node is not None else zeros
    degree_node = degree_node if degree_node is not None else zeros

    high_gamma_delta = _delta_from_pre(high_gamma, centers)
    line_length_delta = _delta_from_pre(line_length, centers)
    entropy_delta = _delta_from_pre(entropy, centers)
    rms_delta = _delta_from_pre(rms, centers)
    low_freq_delta = 0.5 * (_delta_from_pre(delta_band, centers) + _delta_from_pre(theta_band, centers))
    ei_flattening_proxy = high_gamma_delta - low_freq_delta

    adjacency = np.asarray(raw_window_adjacency, dtype=np.float32)
    if adjacency.ndim == 3 and adjacency.shape[:2] == (t, c):
        graph_strength = adjacency.sum(axis=-1).astype(np.float32, copy=False)
    else:
        graph_strength = strength_node
    graph_strength_delta = _delta_from_pre(graph_strength, centers)

    physics = np.stack(
        [
            high_gamma,
            high_gamma_delta,
            line_length_delta,
            entropy_delta,
            rms_delta,
            ei_flattening_proxy,
            strength_node,
            degree_node,
            graph_strength,
            graph_strength_delta,
        ],
        axis=-1,
    )
    return np.nan_to_num(physics, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _prepare_window_tensors(
    sample: Dict[str, Any],
    *,
    feature_dim_fallback: int,
    args: Any | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    raw_features, raw_adjacency, centers = _as_window_tensors(sample, feature_dim_fallback=feature_dim_fallback)
    physics_features = (
        _physics_feature_concat_tensor(raw_features, raw_adjacency, centers)
        if (_get_bool(args, "use_physics_feature_concat", False) or _get_bool(args, "use_physics_features", False))
        else None
    )
    feature_view = str(getattr(args, "feature_view", "self_comparison") if args is not None else "self_comparison").lower()
    adjacency = raw_adjacency
    if feature_view in {"interictal_comparison", "dual_state_comparison"}:
        features = _interictal_comparison_features(raw_features, centers, sample, args)
    else:
        features = _self_comparison_features(raw_features, centers, args)
    if physics_features is not None and physics_features.shape[:2] == features.shape[:2]:
        features = np.concatenate([features, physics_features], axis=-1).astype(np.float32, copy=False)
        if _get_bool(args, "physics_missing_indicators", True):
            # EI/oscillation proxies are computed from the cache. Causal and
            # source-sink groups are unavailable in the legacy cache format.
            requested = {
                part.strip().lower()
                for part in str(getattr(args, "physics_feature_groups", "ei,oscillation") if args is not None else "ei,oscillation").split(",")
                if part.strip()
            }
            missing_names = [name for name in ("causal", "source_sink") if name in requested]
            if missing_names:
                indicators = np.ones((*features.shape[:2], len(missing_names)), dtype=np.float32)
                features = np.concatenate([features, indicators], axis=-1).astype(np.float32, copy=False)
    features = _append_patient_relative_features(features, args)
    adjacency = _mixed_delta_adjacency(adjacency, centers, args)
    return features, adjacency, centers


def _self_compare_offsets(args: Any | None, base_dim: int, feature_dim: int) -> Dict[str, int]:
    feature_view = str(getattr(args, "feature_view", "self_comparison") if args is not None else "self_comparison").lower()
    if feature_view in {"absolute", "raw", "none"}:
        return {"abs": 0} if feature_dim >= base_dim else {}
    if feature_view in {"interictal_comparison", "dual_state_comparison"}:
        return {"abs": 0, "zdelta": base_dim * 2 if feature_dim >= base_dim * 3 else 0}
    offsets: Dict[str, int] = {}
    cursor = 0
    parts = [
        ("abs", _get_bool(args, "self_compare_include_abs", True)),
        ("pre_mean", _get_bool(args, "self_compare_include_pre_mean", False)),
        ("delta", _get_bool(args, "self_compare_include_delta", True)),
        ("zdelta", _get_bool(args, "self_compare_include_zdelta", True)),
        ("ratio", _get_bool(args, "self_compare_include_ratio", True)),
        ("channel_rank", _get_bool(args, "self_compare_include_channel_rank", False)),
    ]
    for name, enabled in parts:
        if not enabled:
            continue
        if cursor + base_dim <= feature_dim:
            offsets[name] = cursor
        cursor += base_dim
    return offsets


def _channel_rank01(values: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    if valid_mask is None:
        valid_mask = np.isfinite(x)
    else:
        valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(x)
    out = np.zeros_like(x, dtype=np.float32)
    idx = np.flatnonzero(valid_mask)
    if idx.size <= 1:
        return out
    local_order = np.argsort(x[idx])
    local_ranks = np.empty(idx.size, dtype=np.float32)
    local_ranks[local_order] = np.linspace(0.0, 1.0, idx.size, dtype=np.float32)
    out[idx] = local_ranks
    return out


def _post_pre_channel_delta(series: np.ndarray, centers: np.ndarray) -> np.ndarray:
    values = np.asarray(series, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        return np.zeros((0,), dtype=np.float32)
    centers = np.asarray(centers, dtype=np.float32)
    pre_mask = centers < 0.0
    post_mask = centers >= 0.0
    if centers.shape[0] != values.shape[0] or not np.any(pre_mask) or not np.any(post_mask):
        return values.mean(axis=0).astype(np.float32, copy=False)
    return (values[post_mask].mean(axis=0) - values[pre_mask].mean(axis=0)).astype(np.float32, copy=False)


def _physics_risk_from_window_tensors(
    window_features: np.ndarray,
    window_adjacency: np.ndarray,
    window_centers: np.ndarray,
    channel_mask: np.ndarray,
    args: Any | None,
) -> np.ndarray:
    features = np.asarray(window_features, dtype=np.float32)
    adjacency = np.asarray(window_adjacency, dtype=np.float32)
    centers = np.asarray(window_centers, dtype=np.float32)
    valid_mask = np.asarray(channel_mask, dtype=bool)
    num_channels = int(valid_mask.shape[0])
    if features.ndim != 3 or features.shape[0] == 0 or features.shape[1] != num_channels:
        return np.zeros((num_channels,), dtype=np.float32)

    base_dim = int(len(WINDOW_NODE_FEATURE_NAMES))
    feature_dim = int(features.shape[-1])
    offsets = _self_compare_offsets(args, base_dim, feature_dim)
    preferred_offset = offsets.get("zdelta", offsets.get("delta", offsets.get("ratio", offsets.get("abs", 0))))
    if preferred_offset + base_dim > feature_dim:
        preferred_offset = 0

    feature_names = list(WINDOW_NODE_FEATURE_NAMES)
    spectral_names = list(BASE_SPECTRAL_FEATURE_NAMES)
    candidate_names = [
        "log_bp_high_gamma",
        "line_length_per_sec",
        "spectral_entropy",
        "rms",
        "variance",
        "strength_norm",
        "degree_norm",
        "pagerank",
    ]
    components: List[np.ndarray] = []
    for name in candidate_names:
        if name not in feature_names:
            continue
        idx = preferred_offset + feature_names.index(name)
        if idx >= feature_dim:
            continue
        values = features[:, :, idx]
        channel_values = values.mean(axis=0) if preferred_offset != offsets.get("abs", -1) else _post_pre_channel_delta(values, centers)
        components.append(_channel_rank01(channel_values, valid_mask))

    if adjacency.ndim == 3 and adjacency.shape[0] == features.shape[0] and adjacency.shape[1] == num_channels:
        strength = adjacency.sum(axis=-1)
        components.append(_channel_rank01(_post_pre_channel_delta(strength, centers), valid_mask))

    if not components:
        return np.zeros((num_channels,), dtype=np.float32)
    risk = np.mean(np.stack(components, axis=0), axis=0)
    risk = np.clip(np.nan_to_num(risk, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    risk[~valid_mask] = 0.0
    return risk.astype(np.float32, copy=False)


def infer_window_feature_dim(window_samples: Iterable[Dict[str, Any]], args: Any | None = None) -> int:
    for sample in window_samples:
        features, _, _ = _prepare_window_tensors(sample, feature_dim_fallback=1, args=args)
        if features.ndim == 3 and features.shape[-1] > 0:
            return int(features.shape[-1])
    return 1


@dataclass
class WindowTensorNormalizer:
    mean: np.ndarray
    std: np.ndarray

    @property
    def feature_dim(self) -> int:
        return int(self.mean.shape[0])

    def transform(self, window_features: np.ndarray) -> np.ndarray:
        features = np.asarray(window_features, dtype=np.float32)
        if features.shape[-1] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got {features.shape[-1]}.")
        return ((features - self.mean) / self.std).astype(np.float32, copy=False)


def fit_window_tensor_normalizer(window_samples: Iterable[Dict[str, Any]], args: Any | None = None) -> WindowTensorNormalizer:
    samples = list(window_samples)
    feature_dim = infer_window_feature_dim(samples, args=args)
    feature_sum = np.zeros((feature_dim,), dtype=np.float64)
    feature_sq_sum = np.zeros((feature_dim,), dtype=np.float64)
    count = 0
    for sample in samples:
        features, _, _ = _prepare_window_tensors(sample, feature_dim_fallback=feature_dim, args=args)
        if features.shape[-1] != feature_dim:
            continue
        flat = features.reshape(-1, feature_dim).astype(np.float64, copy=False)
        feature_sum += flat.sum(axis=0)
        feature_sq_sum += np.square(flat).sum(axis=0)
        count += int(flat.shape[0])

    if count <= 0:
        return WindowTensorNormalizer(mean=np.zeros((feature_dim,), dtype=np.float32), std=np.ones((feature_dim,), dtype=np.float32))
    mean = feature_sum / float(count)
    var = feature_sq_sum / float(count) - np.square(mean)
    std = np.sqrt(np.clip(var, 1e-8, None))
    return WindowTensorNormalizer(mean=mean.astype(np.float32), std=std.astype(np.float32))


def build_patient_examples(
    window_samples: Sequence[Dict[str, Any]],
    patient_index: Dict[str, Dict[str, Any]],
    *,
    normalizer: Optional[WindowTensorNormalizer] = None,
    subject_ids: Optional[Sequence[str]] = None,
    args: Any | None = None,
) -> List[Dict[str, Any]]:
    selected_subjects = set(subject_ids) if subject_ids is not None else None
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for sample in window_samples:
        subject_id = str(sample["subject_id"])
        if selected_subjects is not None and subject_id not in selected_subjects:
            continue
        grouped[subject_id].append(sample)

    feature_dim = int(normalizer.feature_dim) if normalizer is not None else infer_window_feature_dim(window_samples, args=args)
    positive_label = str(getattr(args, "positive_label", "nez") if args is not None else "nez").lower()
    if positive_label not in {"nez", "ez"}:
        raise ValueError(f"Unsupported positive_label={positive_label!r}; expected 'nez' or 'ez'.")
    examples: List[Dict[str, Any]] = []
    for subject_id in sorted(grouped.keys()):
        if subject_id not in patient_index:
            continue
        patient_meta = patient_index[subject_id]
        canonical_channels = list(patient_meta["canonical_channels"])
        channel_to_idx = {name: idx for idx, name in enumerate(canonical_channels)}
        labels_ez = np.asarray(patient_meta["labels"], dtype=np.float32)
        labels_nez = np.where(labels_ez >= 0.0, 1.0 - labels_ez, -1.0).astype(np.float32, copy=False)
        labels = labels_nez if positive_label == "nez" else labels_ez
        channel_mask = np.asarray(patient_meta.get("label_mask", np.ones(len(canonical_channels), dtype=bool)), dtype=bool)
        num_patient_channels = len(canonical_channels)

        seizure_features: List[np.ndarray] = []
        seizure_adjacency: List[np.ndarray] = []
        seizure_window_mask: List[np.ndarray] = []
        seizure_raw: List[np.ndarray] = []
        seizure_channel_mask: List[np.ndarray] = []
        seizure_physics_risk: List[np.ndarray] = []
        run_ids: List[str] = []
        sample_ids: List[str] = []
        seizure_onsets: List[float] = []
        seizure_offsets: List[float] = []

        for sample in grouped[subject_id]:
            local_channels = list(sample["channel_names_norm"])
            local_to_patient = [channel_to_idx.get(channel_name) for channel_name in local_channels]
            features, adjacency, centers = _prepare_window_tensors(sample, feature_dim_fallback=feature_dim, args=args)
            if normalizer is not None:
                features = normalizer.transform(features)
            if features.shape[-1] != feature_dim:
                continue

            raw_x = _normalize_raw_waveform(
                sample["raw_waveform"],
                int(sample.get("raw_valid_samples", 1)),
                raw_valid_start_sample=int(sample.get("raw_valid_start_sample", 0)),
                args=args,
            )
            num_windows = int(features.shape[0])
            raw_samples = int(raw_x.shape[-1])
            aligned_features = np.zeros((num_windows, num_patient_channels, feature_dim), dtype=np.float32)
            aligned_adj = np.zeros((num_windows, num_patient_channels, num_patient_channels), dtype=np.float32)
            aligned_raw = np.zeros((num_patient_channels, raw_samples), dtype=np.float32)
            aligned_channel_mask = np.zeros((num_patient_channels,), dtype=bool)
            aligned_physics_risk = np.zeros((num_patient_channels,), dtype=np.float32)
            local_physics_risk = _physics_risk_from_window_tensors(
                features,
                adjacency,
                centers,
                np.ones((len(local_channels),), dtype=bool),
                args,
            )

            for local_idx, patient_idx in enumerate(local_to_patient):
                if patient_idx is None:
                    continue
                aligned_features[:, patient_idx, :] = features[:, local_idx, :]
                aligned_raw[patient_idx, :] = raw_x[local_idx, :]
                aligned_channel_mask[patient_idx] = True
                aligned_physics_risk[patient_idx] = float(local_physics_risk[local_idx])

            for src_local, src_patient in enumerate(local_to_patient):
                if src_patient is None:
                    continue
                for dst_local, dst_patient in enumerate(local_to_patient):
                    if dst_patient is None:
                        continue
                    aligned_adj[:, src_patient, dst_patient] = adjacency[:, src_local, dst_local]

            seizure_features.append(aligned_features)
            seizure_adjacency.append(aligned_adj)
            seizure_window_mask.append(np.ones((num_windows,), dtype=bool))
            seizure_raw.append(aligned_raw)
            seizure_channel_mask.append(aligned_channel_mask)
            seizure_physics_risk.append(aligned_physics_risk)
            run_ids.append(str(sample["run_id"]))
            sample_ids.append(str(sample["sample_id"]))
            seizure_onsets.append(float(sample["seizure_onset_sec"]))
            seizure_offsets.append(float(sample["seizure_offset_sec"]))

        if not seizure_features:
            continue
        patient_physics_risk = np.mean(np.stack(seizure_physics_risk, axis=0), axis=0).astype(np.float32, copy=False)
        patient_physics_risk[~channel_mask] = 0.0
        examples.append(
            {
                "subject_id": subject_id,
                "canonical_channels": canonical_channels,
                "channel_meta": list(patient_meta.get("channel_meta", [])),
                "labels": labels,
                "labels_nez": labels_nez,
                "labels_ez": labels_ez,
                "label_semantics": "1=NEZ,0=EZ" if positive_label == "nez" else "1=EZ,0=NEZ",
                "channel_mask": channel_mask,
                "features": seizure_features,
                "adjacency": seizure_adjacency,
                "window_mask": seizure_window_mask,
                "raw_x": seizure_raw,
                "seizure_channel_mask": seizure_channel_mask,
                "seizure_physics_risk": seizure_physics_risk,
                "physics_risk": patient_physics_risk,
                "run_ids": run_ids,
                "sample_ids": sample_ids,
                "seizure_onsets": seizure_onsets,
                "seizure_offsets": seizure_offsets,
            }
        )
    return examples


class PatientEZDataset(Dataset):
    """Patient-level EZ localization dataset.

    ``labels`` follow the configured positive class. V3-small defaults to
    NEZ-positive labels: label 1 means NEZ and label 0 means EZ. The original
    EZ-positive view is always retained as ``labels_ez``.
    """

    def __init__(self, patient_examples: Sequence[Dict[str, Any]]) -> None:
        self.patient_examples = list(patient_examples)

    def __len__(self) -> int:
        return len(self.patient_examples)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.patient_examples[index]


def collate_patient_ez_batch(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    if not batch:
        raise ValueError("collate_patient_ez_batch received an empty batch.")

    batch_size = len(batch)
    max_seizures = max(len(item["features"]) for item in batch)
    max_windows = max(int(feature.shape[0]) for item in batch for feature in item["features"])
    max_channels = max(int(item["labels"].shape[0]) for item in batch)
    feature_dim = max(int(feature.shape[-1]) for item in batch for feature in item["features"])
    max_raw_samples = max(int(raw.shape[-1]) for item in batch for raw in item["raw_x"])

    features = torch.zeros((batch_size, max_seizures, max_windows, max_channels, feature_dim), dtype=torch.float32)
    adjacency = torch.zeros((batch_size, max_seizures, max_windows, max_channels, max_channels), dtype=torch.float32)
    raw_x = torch.zeros((batch_size, max_seizures, max_channels, max_raw_samples), dtype=torch.float32)
    labels = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    labels_nez = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    labels_ez = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    physics_risk = torch.zeros((batch_size, max_channels), dtype=torch.float32)
    channel_mask = torch.zeros((batch_size, max_channels), dtype=torch.bool)
    seizure_mask = torch.zeros((batch_size, max_seizures), dtype=torch.bool)
    seizure_channel_mask = torch.zeros((batch_size, max_seizures, max_channels), dtype=torch.bool)
    seizure_physics_risk = torch.zeros((batch_size, max_seizures, max_channels), dtype=torch.float32)
    window_mask = torch.zeros((batch_size, max_seizures, max_windows), dtype=torch.bool)

    subject_ids: List[str] = []
    canonical_channels: List[List[str]] = []
    channel_meta: List[List[Dict[str, Any]]] = []
    run_ids: List[List[str]] = []
    sample_ids: List[List[str]] = []
    seizure_onsets: List[List[float]] = []
    seizure_offsets: List[List[float]] = []
    label_semantics: List[str] = []

    for batch_idx, item in enumerate(batch):
        num_channels = int(item["labels"].shape[0])
        labels[batch_idx, :num_channels] = torch.as_tensor(item["labels"], dtype=torch.float32)
        labels_nez[batch_idx, :num_channels] = torch.as_tensor(item["labels_nez"], dtype=torch.float32)
        labels_ez[batch_idx, :num_channels] = torch.as_tensor(item["labels_ez"], dtype=torch.float32)
        physics_risk[batch_idx, :num_channels] = torch.as_tensor(item.get("physics_risk", np.zeros((num_channels,), dtype=np.float32)), dtype=torch.float32)
        channel_mask[batch_idx, :num_channels] = torch.as_tensor(item["channel_mask"], dtype=torch.bool)
        subject_ids.append(str(item["subject_id"]))
        canonical_channels.append(list(item["canonical_channels"]))
        channel_meta.append(list(item.get("channel_meta", [])))
        run_ids.append(list(item.get("run_ids", [])))
        sample_ids.append(list(item.get("sample_ids", [])))
        seizure_onsets.append([float(value) for value in item.get("seizure_onsets", [])])
        seizure_offsets.append([float(value) for value in item.get("seizure_offsets", [])])
        label_semantics.append(str(item.get("label_semantics", "1=NEZ,0=EZ")))

        for seizure_idx, seizure_features in enumerate(item["features"]):
            seizure_features = np.asarray(seizure_features, dtype=np.float32)
            current_adj = np.asarray(item["adjacency"][seizure_idx], dtype=np.float32)
            current_raw = np.asarray(item["raw_x"][seizure_idx], dtype=np.float32)
            current_channel_mask = np.asarray(item["seizure_channel_mask"][seizure_idx], dtype=bool)
            current_window_mask = np.asarray(item["window_mask"][seizure_idx], dtype=bool)
            t = int(seizure_features.shape[0])
            c = int(seizure_features.shape[1])
            f = int(seizure_features.shape[2])
            l = int(current_raw.shape[-1])
            risk_items = item.get("seizure_physics_risk")
            current_physics_risk = (
                np.asarray(risk_items[seizure_idx], dtype=np.float32)
                if risk_items is not None
                else np.zeros((c,), dtype=np.float32)
            )
            features[batch_idx, seizure_idx, :t, :c, :f] = torch.as_tensor(seizure_features, dtype=torch.float32)
            adjacency[batch_idx, seizure_idx, :t, :c, :c] = torch.as_tensor(current_adj, dtype=torch.float32)
            raw_x[batch_idx, seizure_idx, :c, :l] = torch.as_tensor(current_raw, dtype=torch.float32)
            seizure_channel_mask[batch_idx, seizure_idx, :c] = torch.as_tensor(current_channel_mask, dtype=torch.bool)
            seizure_physics_risk[batch_idx, seizure_idx, :c] = torch.as_tensor(current_physics_risk[:c], dtype=torch.float32)
            window_mask[batch_idx, seizure_idx, :t] = torch.as_tensor(current_window_mask, dtype=torch.bool)
            seizure_mask[batch_idx, seizure_idx] = True

    return {
        "features": features,
        "adjacency": adjacency,
        "raw_x": raw_x,
        "labels": labels,
        "labels_nez": labels_nez,
        "labels_ez": labels_ez,
        "physics_risk": physics_risk,
        "channel_mask": channel_mask,
        "seizure_mask": seizure_mask,
        "seizure_channel_mask": seizure_channel_mask,
        "seizure_physics_risk": seizure_physics_risk,
        "window_mask": window_mask,
        "subject_id": subject_ids,
        "canonical_channels": canonical_channels,
        "channel_meta": channel_meta,
        "run_ids": run_ids,
        "sample_ids": sample_ids,
        "seizure_onsets": seizure_onsets,
        "seizure_offsets": seizure_offsets,
        "label_semantics": label_semantics,
    }


__all__ = [
    "PatientEZDataset",
    "WindowTensorNormalizer",
    "build_patient_examples",
    "collate_patient_ez_batch",
    "fit_window_tensor_normalizer",
    "infer_window_feature_dim",
]
