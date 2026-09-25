from __future__ import annotations

from math import factorial
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import signal
from scipy.integrate import trapezoid
from scipy.signal import welch


SPECTRAL_BANDS: Sequence[tuple[str, float, float]] = (
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

GRAPH_FEATURE_NAMES: Sequence[str] = (
    "degree_norm",
    "strength_norm",
    "clustering_coeff",
    "eigenvector_centrality",
    "pagerank",
    "kcore_norm",
    "local_efficiency",
)

WINDOW_NODE_FEATURE_NAMES: Sequence[str] = tuple(BASE_SPECTRAL_FEATURE_NAMES) + tuple(GRAPH_FEATURE_NAMES)

CLINICAL_ONSET_CORE_FEATURE_NAMES: Sequence[str] = (
    "fast_slow_ratio",
    "low_freq_suppression",
    "broadband_electrodecrement",
    "early_high_gamma_slope",
    "early_line_length_slope",
    "onset_latency_high_gamma",
    "onset_latency_line_length",
    "onset_rank_high_gamma",
    "onset_rank_line_length",
)

BURSTNESS_FEATURE_NAMES: Sequence[str] = (
    "high_gamma_top20pct_mean",
    "line_length_top20pct_mean",
    "high_gamma_max_to_mean",
    "line_length_max_to_mean",
    "peak_time_high_gamma",
    "peak_time_line_length",
)

HFO_LITE_FEATURE_NAMES: Sequence[str] = (
    "hfo80_150_event_rate",
    "hfo80_150_duration_fraction",
    "hfo80_150_mean_envelope_z",
    "hfo80_150_max_envelope_z",
    "hfo80_150_event_count",
)

ENTROPY_MORPHOLOGY_FEATURE_NAMES: Sequence[str] = (
    "teager_kaiser_energy_mean",
    "zero_crossing_rate",
    "kurtosis",
    "skewness",
    "permutation_entropy_order3",
)

CONNECTIVITY_STATIC_FEATURE_NAMES: Sequence[str] = (
    "early_adj_strength",
    "post_adj_strength",
    "post_minus_pre_adj_strength",
    "early_adj_strength_rank",
    "connectivity_drop_score",
    "connectivity_gain_score",
)

_GROUP_TO_NAMES: dict[str, Sequence[str]] = {
    "clinical_onset_core": CLINICAL_ONSET_CORE_FEATURE_NAMES,
    "burstness": BURSTNESS_FEATURE_NAMES,
    "hfo_lite": HFO_LITE_FEATURE_NAMES,
    "entropy_morphology": ENTROPY_MORPHOLOGY_FEATURE_NAMES,
    "connectivity_static": CONNECTIVITY_STATIC_FEATURE_NAMES,
}

_VALID_EXTRA_GROUP_SEQUENCES: set[tuple[str, ...]] = {
    (),
    ("clinical_onset_core",),
    ("clinical_onset_core", "burstness"),
    ("clinical_onset_core", "burstness", "hfo_lite"),
    ("clinical_onset_core", "burstness", "entropy_morphology"),
    ("clinical_onset_core", "burstness", "hfo_lite", "entropy_morphology"),
    ("clinical_onset_core", "burstness", "hfo_lite", "entropy_morphology", "connectivity_static"),
}


def parse_extra_window_feature_groups(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip().lower()
        if not raw or raw == "none":
            return []
        groups = [part.strip().lower() for part in raw.split(",") if part.strip()]
    else:
        groups = [str(part).strip().lower() for part in value if str(part).strip()]
    unknown = [group for group in groups if group not in _GROUP_TO_NAMES]
    if unknown:
        raise ValueError(f"Unknown extra window feature group(s): {unknown}")
    group_tuple = tuple(groups)
    if group_tuple not in _VALID_EXTRA_GROUP_SEQUENCES:
        valid = ["none", *[",".join(seq) for seq in sorted(_VALID_EXTRA_GROUP_SEQUENCES) if seq]]
        raise ValueError(
            f"Unsupported extra window feature group sequence: {groups}. "
            f"Accepted values: {valid}"
        )
    return list(groups)


def window_feature_names_for_groups(extra_window_feature_groups: Any = None) -> list[str]:
    names = list(WINDOW_NODE_FEATURE_NAMES)
    for group in parse_extra_window_feature_groups(extra_window_feature_groups):
        names.extend(_GROUP_TO_NAMES[group])
    if len(names) != len(set(names)):
        duplicates = sorted({name for name in names if names.count(name) > 1})
        raise RuntimeError(f"Duplicate window feature names: {duplicates}")
    return names


def window_feature_groups_metadata(extra_window_feature_groups: Any = None) -> dict[str, Any]:
    return {"base": True, "extra": parse_extra_window_feature_groups(extra_window_feature_groups)}


def _trapz(y: np.ndarray, x: np.ndarray, *, axis: int = -1) -> np.ndarray:
    return np.asarray(trapezoid(y, x=x, axis=axis), dtype=np.float32)


def _safe_float_array(values: np.ndarray) -> np.ndarray:
    return np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def _compute_hjorth_parameters(channel_data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    activity = np.var(channel_data, axis=-1)
    diff1 = np.diff(channel_data, axis=-1)
    diff2 = np.diff(diff1, axis=-1)
    diff1_var = np.var(diff1, axis=-1) + 1e-8
    diff2_var = np.var(diff2, axis=-1) + 1e-8
    mobility = np.sqrt(diff1_var / (activity + 1e-8))
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

    nperseg = int(min(num_samples, max(64, int(round(float(sfreq) * 2.0)))))
    noverlap = int(min(max(0, nperseg // 2), max(nperseg - 1, 0)))
    freqs, psd = welch(data, fs=float(sfreq), nperseg=nperseg, noverlap=noverlap, axis=-1, detrend=False)
    psd = np.asarray(psd, dtype=np.float32)

    restricted_mask = (freqs >= float(spectral_min_freq)) & (freqs <= float(spectral_max_freq))
    restricted_freqs = freqs[restricted_mask]
    restricted_psd = psd[:, restricted_mask]
    if restricted_psd.shape[-1] == 0:
        raise ValueError(f"No spectral bins available between {spectral_min_freq} and {spectral_max_freq} Hz.")

    band_features = []
    for _, low, high in SPECTRAL_BANDS:
        band_mask = (restricted_freqs >= float(low)) & (restricted_freqs < float(high))
        if np.any(band_mask):
            band_power = _trapz(restricted_psd[:, band_mask], restricted_freqs[band_mask], axis=-1)
        else:
            band_power = np.zeros(num_channels, dtype=np.float32)
        band_features.append(np.log1p(band_power).astype(np.float32, copy=False))

    total_power = _trapz(restricted_psd, restricted_freqs, axis=-1)
    log_total_power = np.log1p(total_power)
    rms = np.sqrt(np.mean(np.square(data), axis=-1) + 1e-8).astype(np.float32, copy=False)
    variance = np.var(data, axis=-1).astype(np.float32, copy=False)
    duration_sec = max(float(num_samples) / max(float(sfreq), 1e-8), 1e-6)
    line_length_per_sec = (np.sum(np.abs(np.diff(data, axis=-1)), axis=-1) / duration_sec).astype(np.float32, copy=False)
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
    )
    if features.shape[-1] != len(BASE_SPECTRAL_FEATURE_NAMES):
        raise RuntimeError(f"Expected {len(BASE_SPECTRAL_FEATURE_NAMES)} spectral features, got {features.shape[-1]}.")
    return _safe_float_array(features)


def compute_abs_pearson_connectivity(window_data: np.ndarray) -> np.ndarray:
    data = np.asarray(window_data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError("window_data must have shape [channels, time].")
    num_channels = int(data.shape[0])
    if num_channels == 0:
        return np.zeros((0, 0), dtype=np.float32)
    if num_channels == 1:
        return np.zeros((1, 1), dtype=np.float32)
    corr = np.corrcoef(data)
    corr = np.abs(np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)).astype(np.float32, copy=False)
    np.fill_diagonal(corr, 0.0)
    return corr


def compute_thresholded_abs_pearson_adjacency(
    window_data: np.ndarray,
    *,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> np.ndarray:
    corr = compute_abs_pearson_connectivity(window_data)
    num_channels = int(corr.shape[0])
    if num_channels <= 1:
        return np.zeros_like(corr, dtype=np.float32)
    upper = corr[np.triu_indices(num_channels, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return np.zeros_like(corr, dtype=np.float32)
    threshold = max(float(min_edge_weight), float(np.quantile(positive, float(np.clip(edge_quantile, 0.0, 1.0)))))
    edge_mask = np.triu(corr >= threshold, k=1)
    if not np.any(edge_mask):
        tri_i, tri_j = np.triu_indices(num_channels, k=1)
        max_idx = int(np.argmax(upper))
        edge_mask = np.zeros_like(corr, dtype=bool)
        edge_mask[int(tri_i[max_idx]), int(tri_j[max_idx])] = True
    weighted = np.zeros_like(corr, dtype=np.float32)
    weighted[edge_mask] = corr[edge_mask]
    return (weighted + weighted.T).astype(np.float32, copy=False)


def compute_graph_node_features(
    window_data: np.ndarray,
    *,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> np.ndarray:
    adjacency = compute_thresholded_abs_pearson_adjacency(
        window_data,
        edge_quantile=edge_quantile,
        min_edge_weight=min_edge_weight,
    )
    num_channels = int(adjacency.shape[0])
    if num_channels == 0:
        return np.zeros((0, len(GRAPH_FEATURE_NAMES)), dtype=np.float32)
    degree = (adjacency > 0.0).sum(axis=1).astype(np.float32) / max(num_channels - 1, 1)
    strength = adjacency.sum(axis=1).astype(np.float32) / max(num_channels - 1, 1)
    clustering = np.zeros(num_channels, dtype=np.float32)
    local_efficiency = np.zeros(num_channels, dtype=np.float32)
    for idx in range(num_channels):
        neighbors = np.flatnonzero(adjacency[idx] > 0.0)
        if neighbors.size >= 2:
            sub = adjacency[np.ix_(neighbors, neighbors)]
            possible = neighbors.size * (neighbors.size - 1)
            clustering[idx] = float(np.count_nonzero(sub) / max(possible, 1))
            local_efficiency[idx] = float(sub.sum() / max(possible, 1))
    if adjacency.sum() > 0.0:
        centrality = strength / max(float(strength.sum()), 1e-8)
    else:
        centrality = np.full(num_channels, 1.0 / max(num_channels, 1), dtype=np.float32)
    kcore = degree / max(float(degree.max()), 1e-8) if degree.max() > 0.0 else np.zeros_like(degree)
    features = np.stack([degree, strength, clustering, centrality, centrality, kcore, local_efficiency], axis=-1)
    return _safe_float_array(features)


def _feature_index(name: str) -> int:
    return list(WINDOW_NODE_FEATURE_NAMES).index(name)


def _baseline_mask(centers: np.ndarray, num_windows: int) -> np.ndarray:
    mask = np.asarray(centers, dtype=np.float32) < 0.0
    if mask.shape[0] != num_windows or not np.any(mask):
        mask = np.ones((num_windows,), dtype=bool)
    return mask


def _post_mask_or_all(centers: np.ndarray, num_windows: int) -> np.ndarray:
    mask = np.asarray(centers, dtype=np.float32) >= 0.0
    if mask.shape[0] != num_windows or not np.any(mask):
        mask = np.ones((num_windows,), dtype=bool)
    return mask


def _rank_earlier_is_better(latency: np.ndarray) -> np.ndarray:
    latency = np.asarray(latency, dtype=np.float32)
    if latency.size <= 1:
        return np.ones_like(latency, dtype=np.float32)
    order = np.argsort(latency, kind="mergesort")
    ranks = np.zeros_like(latency, dtype=np.float32)
    ranks[order] = 1.0 - (np.arange(latency.size, dtype=np.float32) / float(latency.size - 1))
    return ranks


def _least_squares_slope(series: np.ndarray, centers: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if int(np.sum(mask)) < 2:
        return np.zeros((series.shape[1],), dtype=np.float32)
    x = centers[mask].astype(np.float32)
    y = series[mask].astype(np.float32)
    x_centered = x - float(x.mean())
    denom = float(np.sum(np.square(x_centered))) + 1e-8
    return _safe_float_array(np.sum((y - y.mean(axis=0, keepdims=True)) * x_centered[:, None], axis=0) / denom)


def _latency_features(series: np.ndarray, centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pre_mask = _baseline_mask(centers, series.shape[0])
    post_mask = centers >= 0.0
    pre_mean = series[pre_mask].mean(axis=0)
    pre_std = series[pre_mask].std(axis=0)
    threshold = pre_mean + 2.0 * pre_std
    if np.any(post_mask):
        post_values = series[post_mask]
        post_centers = centers[post_mask]
        fallback = float(post_centers.max()) + 1.0
    else:
        post_values = series
        post_centers = centers
        fallback = float(np.max(np.abs(centers))) + 1.0
    latency = np.full((series.shape[1],), fallback, dtype=np.float32)
    for channel_idx in range(series.shape[1]):
        crossings = np.flatnonzero(post_values[:, channel_idx] >= threshold[channel_idx])
        if crossings.size:
            latency[channel_idx] = float(post_centers[int(crossings[0])])
    norm = max(float(np.max(np.abs(centers))), 1e-6)
    latency_norm = latency / norm
    return _safe_float_array(latency_norm), _rank_earlier_is_better(latency)


def _clinical_onset_core_features(base_series: np.ndarray, centers: np.ndarray) -> np.ndarray:
    idx_delta = _feature_index("log_bp_delta")
    idx_theta = _feature_index("log_bp_theta")
    idx_alpha = _feature_index("log_bp_alpha")
    idx_beta = _feature_index("log_bp_beta")
    idx_high = _feature_index("log_bp_high_gamma")
    idx_total = _feature_index("log_total_power")
    idx_line = _feature_index("line_length_per_sec")

    low_power = np.expm1(base_series[:, :, idx_delta]) + np.expm1(base_series[:, :, idx_theta])
    low_power = low_power + np.expm1(base_series[:, :, idx_alpha]) + np.expm1(base_series[:, :, idx_beta])
    high_power = np.expm1(base_series[:, :, idx_high])
    low_freq_current = np.log1p(np.clip(low_power, 0.0, None))
    fast_slow_ratio = np.log1p(np.clip(high_power, 0.0, None)) - low_freq_current

    pre_mask = _baseline_mask(centers, base_series.shape[0])
    pre_low = low_freq_current[pre_mask].mean(axis=0, keepdims=True)
    low_freq_suppression = pre_low - low_freq_current
    total = base_series[:, :, idx_total]
    broadband_electrodecrement = total[pre_mask].mean(axis=0, keepdims=True) - total

    early_mask = (centers >= 0.0) & (centers <= 10.0)
    if int(np.sum(early_mask)) < 2:
        early_mask = centers >= 0.0
    high_slope = _least_squares_slope(base_series[:, :, idx_high], centers, early_mask)
    line_slope = _least_squares_slope(base_series[:, :, idx_line], centers, early_mask)
    high_latency, high_rank = _latency_features(base_series[:, :, idx_high], centers)
    line_latency, line_rank = _latency_features(base_series[:, :, idx_line], centers)

    repeated = [
        np.broadcast_to(high_slope[None, :], fast_slow_ratio.shape),
        np.broadcast_to(line_slope[None, :], fast_slow_ratio.shape),
        np.broadcast_to(high_latency[None, :], fast_slow_ratio.shape),
        np.broadcast_to(line_latency[None, :], fast_slow_ratio.shape),
        np.broadcast_to(high_rank[None, :], fast_slow_ratio.shape),
        np.broadcast_to(line_rank[None, :], fast_slow_ratio.shape),
    ]
    return _safe_float_array(
        np.stack(
            [
                fast_slow_ratio,
                low_freq_suppression,
                broadband_electrodecrement,
                *repeated,
            ],
            axis=-1,
        )
    )


def _top_pct_mean(values: np.ndarray, pct: float = 0.20) -> np.ndarray:
    k = max(1, int(np.ceil(values.shape[0] * float(pct))))
    sorted_values = np.sort(values, axis=0)
    return sorted_values[-k:].mean(axis=0)


def _burstness_features(base_series: np.ndarray, centers: np.ndarray) -> np.ndarray:
    idx_high = _feature_index("log_bp_high_gamma")
    idx_line = _feature_index("line_length_per_sec")
    mask = _post_mask_or_all(centers, base_series.shape[0])
    norm = max(float(np.max(np.abs(centers))), 1e-6)
    outputs = []
    for idx in (idx_high, idx_line):
        values = base_series[mask, :, idx]
        outputs.append(_top_pct_mean(values))
    for idx in (idx_high, idx_line):
        values = base_series[mask, :, idx]
        ratio = np.log1p(np.clip(values.max(axis=0) / (values.mean(axis=0) + 1e-8), 0.0, 1e6))
        outputs.append(ratio)
    selected_centers = centers[mask]
    for idx in (idx_high, idx_line):
        values = base_series[mask, :, idx]
        peak_idx = np.argmax(values, axis=0)
        outputs.append(selected_centers[peak_idx] / norm)
    channel_features = np.stack(outputs, axis=-1).astype(np.float32, copy=False)
    return _safe_float_array(np.broadcast_to(channel_features[None, :, :], (base_series.shape[0], *channel_features.shape)))


def _count_true_runs(mask: np.ndarray, min_len: int) -> tuple[int, int]:
    count = 0
    duration = 0
    start: int | None = None
    for idx, flag in enumerate(np.asarray(mask, dtype=bool).tolist() + [False]):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            run_len = idx - start
            if run_len >= min_len:
                count += 1
                duration += run_len
            start = None
    return count, duration


def _hfo_lite_window_features(window_data: np.ndarray, sfreq: float) -> np.ndarray:
    data = np.asarray(window_data, dtype=np.float32)
    num_channels, num_samples = data.shape
    out = np.zeros((num_channels, len(HFO_LITE_FEATURE_NAMES)), dtype=np.float32)
    nyquist = float(sfreq) / 2.0
    if sfreq < 320.0 or nyquist <= 150.0 or num_samples < 4:
        return out
    try:
        sos = signal.butter(4, [80.0, 150.0], btype="bandpass", fs=float(sfreq), output="sos")
    except Exception:
        return out
    min_len = max(1, int(round(0.010 * float(sfreq))))
    duration_sec = max(float(num_samples) / max(float(sfreq), 1e-8), 1e-6)
    for channel_idx in range(num_channels):
        try:
            filtered = signal.sosfiltfilt(sos, data[channel_idx])
            env = np.abs(signal.hilbert(filtered))
            median = float(np.median(env))
            mad = float(np.median(np.abs(env - median)))
            z = (env - median) / (1.4826 * mad + 1e-8)
            mask = z > 3.0
            event_count, event_samples = _count_true_runs(mask, min_len)
            out[channel_idx, 0] = float(event_count) / duration_sec
            out[channel_idx, 1] = float(event_samples) / max(float(num_samples), 1.0)
            out[channel_idx, 2] = float(np.mean(z[mask])) if np.any(mask) else 0.0
            out[channel_idx, 3] = float(np.max(z)) if z.size else 0.0
            out[channel_idx, 4] = float(event_count)
        except Exception:
            out[channel_idx] = 0.0
    return _safe_float_array(out)


def _permutation_entropy_order3(values: np.ndarray) -> float:
    x = np.asarray(values, dtype=np.float32)
    if x.size < 3:
        return 0.0
    counts: dict[tuple[int, int, int], int] = {}
    for idx in range(x.size - 2):
        pattern = tuple(np.argsort(x[idx : idx + 3], kind="mergesort").tolist())
        counts[pattern] = counts.get(pattern, 0) + 1
    probs = np.asarray(list(counts.values()), dtype=np.float64)
    probs = probs / max(float(probs.sum()), 1e-8)
    entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
    return float(entropy / np.log(float(factorial(3))))


def _entropy_morphology_window_features(window_data: np.ndarray, sfreq: float) -> np.ndarray:
    data = np.asarray(window_data, dtype=np.float32)
    num_channels, num_samples = data.shape
    out = np.zeros((num_channels, len(ENTROPY_MORPHOLOGY_FEATURE_NAMES)), dtype=np.float32)
    duration_sec = max(float(num_samples) / max(float(sfreq), 1e-8), 1e-6)
    for channel_idx in range(num_channels):
        x = data[channel_idx]
        if x.size >= 3:
            psi = np.square(x[1:-1]) - x[:-2] * x[2:]
            out[channel_idx, 0] = float(np.mean(np.abs(psi)))
        out[channel_idx, 1] = float(np.count_nonzero(np.diff(np.signbit(x)))) / duration_sec
        centered = x - float(np.mean(x))
        std = float(np.std(centered))
        if std > 1e-8:
            z = centered / std
            out[channel_idx, 2] = float(np.mean(np.power(z, 4)) - 3.0)
            out[channel_idx, 3] = float(np.mean(np.power(z, 3)))
        out[channel_idx, 4] = _permutation_entropy_order3(x)
    return _safe_float_array(out)


def _connectivity_static_features(adjacency_series: np.ndarray, centers: np.ndarray) -> np.ndarray:
    strength = adjacency_series.sum(axis=-1) / max(adjacency_series.shape[-1] - 1, 1)
    pre_mask = _baseline_mask(centers, strength.shape[0])
    post_mask = _post_mask_or_all(centers, strength.shape[0])
    early_mask = (centers >= 0.0) & (centers <= 10.0)
    if not np.any(early_mask):
        early_mask = post_mask
    pre_strength = strength[pre_mask].mean(axis=0)
    post_strength = strength[post_mask].mean(axis=0)
    early_strength = strength[early_mask].mean(axis=0)
    early_rank = _rank_earlier_is_better(-early_strength)
    channel_features = np.stack(
        [
            early_strength,
            post_strength,
            post_strength - pre_strength,
            early_rank,
            np.maximum(pre_strength - post_strength, 0.0),
            np.maximum(post_strength - pre_strength, 0.0),
        ],
        axis=-1,
    )
    return _safe_float_array(np.broadcast_to(channel_features[None, :, :], (strength.shape[0], *channel_features.shape)))


def compute_window_feature_tensors_with_names(
    data: np.ndarray,
    window_segments: Sequence[dict[str, Any]],
    sfreq: float,
    *,
    spectral_min_freq: float = 1.0,
    spectral_max_freq: float = 150.0,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
    extra_window_feature_groups: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    signal_data = np.asarray(data, dtype=np.float32)
    if signal_data.ndim != 2:
        raise ValueError("data must have shape [channels, time].")
    if not window_segments:
        raise ValueError("window_segments must contain at least one sliding window.")

    groups = parse_extra_window_feature_groups(extra_window_feature_groups)
    base_series = []
    adjacency_series = []
    hfo_series = []
    entropy_series = []
    centers = []
    for window_meta in window_segments:
        start = int(window_meta["start_sample"])
        end = int(window_meta["end_sample"])
        window_data = signal_data[:, start:end]
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
        base_series.append(np.concatenate([spectral, graph_nodes], axis=-1).astype(np.float32, copy=False))
        adjacency_series.append(adjacency.astype(np.float32, copy=False))
        if "hfo_lite" in groups:
            hfo_series.append(_hfo_lite_window_features(window_data, sfreq=float(sfreq)))
        if "entropy_morphology" in groups:
            entropy_series.append(_entropy_morphology_window_features(window_data, sfreq=float(sfreq)))
        centers.append(float(window_meta["relative_center_sec"]))

    if not base_series:
        raise ValueError("No valid sliding windows were available for tensor feature extraction.")

    window_features = np.stack(base_series, axis=0).astype(np.float32, copy=False)
    window_adjacency = np.stack(adjacency_series, axis=0).astype(np.float32, copy=False)
    relative_centers_sec = np.asarray(centers, dtype=np.float32)
    extra_arrays = []
    if "clinical_onset_core" in groups:
        extra_arrays.append(_clinical_onset_core_features(window_features, relative_centers_sec))
    if "burstness" in groups:
        extra_arrays.append(_burstness_features(window_features, relative_centers_sec))
    if "hfo_lite" in groups:
        extra_arrays.append(_safe_float_array(np.stack(hfo_series, axis=0)))
    if "entropy_morphology" in groups:
        extra_arrays.append(_safe_float_array(np.stack(entropy_series, axis=0)))
    if "connectivity_static" in groups:
        extra_arrays.append(_connectivity_static_features(window_adjacency, relative_centers_sec))
    if extra_arrays:
        window_features = np.concatenate([window_features, *extra_arrays], axis=-1).astype(np.float32, copy=False)

    feature_names = window_feature_names_for_groups(groups)
    if window_features.shape[-1] != len(feature_names):
        raise RuntimeError(f"window feature dim={window_features.shape[-1]} but names={len(feature_names)}.")
    return _safe_float_array(window_features), window_adjacency, relative_centers_sec, feature_names


def compute_window_feature_tensors(
    data: np.ndarray,
    window_segments: Sequence[dict[str, Any]],
    sfreq: float,
    *,
    spectral_min_freq: float = 1.0,
    spectral_max_freq: float = 150.0,
    edge_quantile: float = 0.70,
    min_edge_weight: float = 0.10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    window_features, window_adjacency, relative_centers_sec, _ = compute_window_feature_tensors_with_names(
        data,
        window_segments,
        sfreq,
        spectral_min_freq=spectral_min_freq,
        spectral_max_freq=spectral_max_freq,
        edge_quantile=edge_quantile,
        min_edge_weight=min_edge_weight,
        extra_window_feature_groups=None,
    )
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


def _build_centered_raw_waveform(
    segment_data: np.ndarray,
    *,
    used_duration_sec: float,
    target_duration_sec: float,
    target_sfreq: float,
    segment_start_sec: float,
    center_sec: float,
) -> tuple[np.ndarray, int, int]:
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


__all__ = [
    "BASE_SPECTRAL_FEATURE_NAMES",
    "BURSTNESS_FEATURE_NAMES",
    "CLINICAL_ONSET_CORE_FEATURE_NAMES",
    "CONNECTIVITY_STATIC_FEATURE_NAMES",
    "ENTROPY_MORPHOLOGY_FEATURE_NAMES",
    "GRAPH_FEATURE_NAMES",
    "HFO_LITE_FEATURE_NAMES",
    "SPECTRAL_BANDS",
    "WINDOW_NODE_FEATURE_NAMES",
    "_build_centered_raw_waveform",
    "compute_abs_pearson_connectivity",
    "compute_graph_node_features",
    "compute_spectral_channel_features",
    "compute_thresholded_abs_pearson_adjacency",
    "compute_window_feature_tensors",
    "compute_window_feature_tensors_with_names",
    "parse_extra_window_feature_groups",
    "window_feature_groups_metadata",
    "window_feature_names_for_groups",
]
