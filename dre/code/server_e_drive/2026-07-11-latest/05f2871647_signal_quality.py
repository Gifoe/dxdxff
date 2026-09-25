from __future__ import annotations

import numpy as np


def robust_scale_channels(signal: np.ndarray, valid_mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(signal, dtype=float)
    if data.ndim != 2:
        raise ValueError("signal must be [C,T]")
    median = np.nanmedian(data, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(data - median), axis=1, keepdims=True)
    scale = 1.4826 * mad
    valid = np.isfinite(data).sum(axis=1) >= 4
    valid &= np.isfinite(scale[:, 0]) & (scale[:, 0] > 1e-8)
    if valid_mask is not None:
        valid &= np.asarray(valid_mask, dtype=bool)
    output = np.full_like(data, np.nan, dtype=float)
    output[valid] = np.clip((data[valid] - median[valid]) / scale[valid], -10.0, 10.0)
    return output, valid


def robust_z_from_baseline(values: np.ndarray, baseline_mask: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=float); base = data[:, np.asarray(baseline_mask, dtype=bool)]
    median = np.nanmedian(base, axis=1, keepdims=True)
    mad = np.nanmedian(np.abs(base - median), axis=1, keepdims=True)
    scale = 1.4826 * mad
    fallback = np.nanstd(base, axis=1, keepdims=True)
    scale = np.where(scale > 1e-8, scale, fallback)
    scale = np.where(scale > 1e-8, scale, np.nan)
    return (data - median) / scale


def time_slice(onset_sample: int, sampling_rate: float, n_samples: int, start_sec: float, end_sec: float) -> slice:
    start = int(np.clip(round(onset_sample + start_sec * sampling_rate), 0, n_samples))
    end = int(np.clip(round(onset_sample + end_sec * sampling_rate), 0, n_samples))
    return slice(start, max(start, end))


__all__ = ["robust_scale_channels", "robust_z_from_baseline", "time_slice"]
