from __future__ import annotations

import numpy as np
from scipy.signal import welch


SHORT_FEATURE_NAMES = [
    "rms",
    "variance",
    "line_length",
    "peak_to_peak",
    "max_abs",
    "zero_crossing_rate",
    "hjorth_mobility",
    "hjorth_complexity",
    "spectral_entropy",
    "low_gamma_power",
    "high_gamma_power",
    "broadband_power",
]

LOWFREQ_FEATURE_NAMES = [
    "delta_power",
    "theta_power",
    "alpha_power",
    "beta_power",
    "low_gamma_power",
    "relative_delta",
    "relative_theta",
    "relative_alpha",
    "relative_beta",
    "relative_low_gamma",
    "low_high_ratio",
    "spectral_entropy_long",
]

BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta": (13.0, 30.0),
    "low_gamma": (30.0, 80.0),
    "high_gamma": (80.0, 150.0),
    "broadband": (1.0, 150.0),
}


def compute_short_window_channel_features(
    window_data: np.ndarray,
    sfreq: float,
) -> tuple[np.ndarray, list[str]]:
    data = _as_2d_float(window_data)
    if data.shape[-1] < 2:
        return np.zeros((data.shape[0], len(SHORT_FEATURE_NAMES)), dtype=np.float32), list(SHORT_FEATURE_NAMES)

    duration = max(data.shape[-1] / float(sfreq), 1e-6)
    rms = np.sqrt(np.mean(np.square(data), axis=-1) + 1e-8)
    variance = np.var(data, axis=-1)
    line_length = np.sum(np.abs(np.diff(data, axis=-1)), axis=-1) / duration
    peak_to_peak = np.ptp(data, axis=-1)
    max_abs = np.max(np.abs(data), axis=-1)
    zero_crossing_rate = np.mean(np.diff(np.signbit(data), axis=-1), axis=-1)
    hjorth_mobility, hjorth_complexity = _hjorth(data)

    freqs, psd = _safe_welch(data, sfreq)
    spectral_entropy = _spectral_entropy(psd)
    low_gamma = _band_power(freqs, psd, *BANDS["low_gamma"])
    high_gamma = _band_power(freqs, psd, *BANDS["high_gamma"])
    broadband = _band_power(freqs, psd, *BANDS["broadband"])

    features = np.stack(
        [
            rms,
            variance,
            line_length,
            peak_to_peak,
            max_abs,
            zero_crossing_rate,
            hjorth_mobility,
            hjorth_complexity,
            spectral_entropy,
            np.log1p(low_gamma),
            np.log1p(high_gamma),
            np.log1p(broadband),
        ],
        axis=-1,
    )
    return _finite(features), list(SHORT_FEATURE_NAMES)


def compute_lowfreq_features_aligned_to_short_grid(
    signal_segment: np.ndarray,
    sfreq: float,
    short_window_centers_sec: np.ndarray,
    lowfreq_window_sec: float = 1.0,
) -> tuple[np.ndarray, list[str]]:
    data = _as_2d_float(signal_segment)
    centers = np.asarray(short_window_centers_sec, dtype=np.float32)
    num_channels, num_samples = data.shape
    window_samples = max(2, int(round(float(lowfreq_window_sec) * float(sfreq))))
    half_window = window_samples // 2
    segment_duration = num_samples / float(sfreq)
    onset_in_segment_sec = segment_duration / 2.0

    feature_blocks = []
    for center_sec in centers:
        center_sample = int(round((float(center_sec) + onset_in_segment_sec) * float(sfreq)))
        start = center_sample - half_window
        end = start + window_samples
        if start < 0:
            start = 0
            end = min(window_samples, num_samples)
        if end > num_samples:
            end = num_samples
            start = max(0, end - window_samples)
        window = data[:, start:end]
        if window.shape[-1] < 2:
            feature_blocks.append(np.zeros((num_channels, len(LOWFREQ_FEATURE_NAMES)), dtype=np.float32))
        else:
            feature_blocks.append(_compute_lowfreq_window_features(window, sfreq))

    features = np.stack(feature_blocks, axis=0) if feature_blocks else np.zeros((0, num_channels, 0), dtype=np.float32)
    return _finite(features), list(LOWFREQ_FEATURE_NAMES)


def _compute_lowfreq_window_features(window_data: np.ndarray, sfreq: float) -> np.ndarray:
    freqs, psd = _safe_welch(window_data, sfreq)
    delta = _band_power(freqs, psd, *BANDS["delta"])
    theta = _band_power(freqs, psd, *BANDS["theta"])
    alpha = _band_power(freqs, psd, *BANDS["alpha"])
    beta = _band_power(freqs, psd, *BANDS["beta"])
    low_gamma = _band_power(freqs, psd, *BANDS["low_gamma"])
    total = np.clip(delta + theta + alpha + beta + low_gamma, 1e-8, None)
    low_high_ratio = (delta + theta + alpha) / np.clip(beta + low_gamma, 1e-8, None)
    entropy = _spectral_entropy(psd)
    return np.stack(
        [
            np.log1p(delta),
            np.log1p(theta),
            np.log1p(alpha),
            np.log1p(beta),
            np.log1p(low_gamma),
            delta / total,
            theta / total,
            alpha / total,
            beta / total,
            low_gamma / total,
            low_high_ratio,
            entropy,
        ],
        axis=-1,
    ).astype(np.float32, copy=False)


def _as_2d_float(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data, dtype=np.float32)
    if array.ndim != 2:
        raise ValueError("window_data must have shape [channels, time].")
    return array


def _safe_welch(data: np.ndarray, sfreq: float) -> tuple[np.ndarray, np.ndarray]:
    nperseg = max(2, min(data.shape[-1], int(round(float(sfreq) * 2.0))))
    noverlap = min(nperseg // 2, nperseg - 1)
    freqs, psd = welch(data, fs=float(sfreq), nperseg=nperseg, noverlap=noverlap, axis=-1, detrend=False)
    return np.asarray(freqs, dtype=np.float32), np.asarray(psd, dtype=np.float32)


def _band_power(freqs: np.ndarray, psd: np.ndarray, low: float, high: float) -> np.ndarray:
    high = min(float(high), float(freqs[-1]) if freqs.size else float(high))
    mask = (freqs >= float(low)) & (freqs < high)
    if not np.any(mask):
        return np.zeros(psd.shape[0], dtype=np.float32)
    return _integrate_trapezoid(psd[:, mask], freqs[mask], axis=-1).astype(np.float32, copy=False)


def _integrate_trapezoid(y: np.ndarray, x: np.ndarray, axis: int = -1) -> np.ndarray:
    trapezoid = getattr(np, "trapezoid", None)
    if trapezoid is not None:
        return trapezoid(y, x, axis=axis)
    trapz = getattr(np, "trapz", None)
    if trapz is not None:
        return trapz(y, x, axis=axis)
    y = np.asarray(y)
    x = np.asarray(x)
    dx = np.diff(x)
    y_moved = np.moveaxis(y, axis, -1)
    area = 0.5 * (y_moved[..., 1:] + y_moved[..., :-1]) * dx
    return area.sum(axis=-1)


def _spectral_entropy(psd: np.ndarray) -> np.ndarray:
    denom = np.clip(psd.sum(axis=-1, keepdims=True), 1e-8, None)
    p = psd / denom
    entropy = -np.sum(p * np.log(p + 1e-8), axis=-1)
    return (entropy / np.log(max(psd.shape[-1], 2))).astype(np.float32, copy=False)


def _hjorth(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    activity = np.var(data, axis=-1) + 1e-8
    diff1 = np.diff(data, axis=-1)
    diff2 = np.diff(diff1, axis=-1)
    diff1_var = np.var(diff1, axis=-1) + 1e-8
    diff2_var = np.var(diff2, axis=-1) + 1e-8
    mobility = np.sqrt(diff1_var / activity)
    complexity = np.sqrt(diff2_var / diff1_var) / np.clip(mobility, 1e-8, None)
    return mobility.astype(np.float32, copy=False), complexity.astype(np.float32, copy=False)


def _finite(array: np.ndarray) -> np.ndarray:
    return np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


__all__ = [
    "LOWFREQ_FEATURE_NAMES",
    "SHORT_FEATURE_NAMES",
    "compute_lowfreq_features_aligned_to_short_grid",
    "compute_short_window_channel_features",
]
