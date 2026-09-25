from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Any

import numpy as np
from scipy.signal import butter, iirnotch, resample_poly, sosfiltfilt, filtfilt


@dataclass(frozen=True)
class RawPreprocessingConfig:
    target_sfreq: float = 200.0
    window_sec: float = 4.0
    bandpass: tuple[float, float] | None = (0.5, 80.0)
    notch: float | None = 50.0
    normalization: str = "robust_median_iqr"
    clipping: float = 8.0


def _source_windows(sample: dict[str, Any], sfreq: float, duration: float) -> tuple[np.ndarray, str]:
    if "raw_window_waveforms" in sample:
        windows = np.asarray(sample["raw_window_waveforms"], dtype=np.float32)
        if windows.ndim != 3:
            raise ValueError(f"raw_window_waveforms must be [window,channel,time], got {windows.shape}.")
        return windows, "explicit_raw_window_waveforms"
    required = ("raw_waveform", "raw_valid_start_sample", "raw_valid_samples", "window_relative_centers_sec")
    missing = [key for key in required if key not in sample]
    if missing:
        raise ValueError(f"Continuous raw extraction requires audited raw_valid fields; missing {missing}.")
    raw = np.asarray(sample["raw_waveform"], dtype=np.float32)
    centers = np.asarray(sample["window_relative_centers_sec"], dtype=float).reshape(-1)
    if raw.ndim != 2 or not centers.size:
        raise ValueError("raw_waveform must be [channel,time] and window centers must be non-empty.")
    valid_start = int(sample["raw_valid_start_sample"])
    valid_end = min(raw.shape[1], valid_start + int(sample["raw_valid_samples"]))
    if valid_start < 0 or valid_end <= valid_start:
        raise ValueError("Audited raw_valid interval is invalid.")
    length = max(1, int(round(sfreq * duration)))
    onset = raw.shape[1] / 2.0
    windows = np.zeros((len(centers), raw.shape[0], length), dtype=np.float32)
    for index, relative_center in enumerate(centers):
        requested_start = int(round(onset + relative_center * sfreq - length / 2.0))
        requested_end = requested_start + length
        copy_start = max(requested_start, valid_start, 0)
        copy_end = min(requested_end, valid_end, raw.shape[1])
        if copy_end <= copy_start:
            raise ValueError(f"Raw window {index} does not overlap the audited valid interval.")
        destination = copy_start - requested_start
        windows[index, :, destination : destination + copy_end - copy_start] = raw[:, copy_start:copy_end]
    return windows, "seizure_onset_at_raw_target_midpoint"


def _preprocess_1d(values: np.ndarray, sfreq: float, config: RawPreprocessingConfig) -> np.ndarray:
    signal = np.asarray(values, dtype=np.float64)
    if config.bandpass is not None:
        low, high = config.bandpass
        high = min(float(high), sfreq / 2.0 - 0.5)
        if 0 < low < high:
            sos = butter(4, [float(low), high], btype="bandpass", fs=sfreq, output="sos")
            signal = sosfiltfilt(sos, signal)
    if config.notch is not None and 0 < float(config.notch) < sfreq / 2.0:
        b, a = iirnotch(float(config.notch), 30.0, fs=sfreq)
        signal = filtfilt(b, a, signal)
    if not np.isclose(sfreq, config.target_sfreq):
        ratio = Fraction(float(config.target_sfreq) / float(sfreq)).limit_denominator(1000)
        signal = resample_poly(signal, ratio.numerator, ratio.denominator)
    target = int(round(config.target_sfreq * config.window_sec))
    if signal.size < target:
        signal = np.pad(signal, (0, target - signal.size))
    else:
        signal = signal[:target]
    if config.normalization == "robust_median_iqr":
        median = float(np.median(signal))
        scale = float(np.percentile(signal, 75) - np.percentile(signal, 25))
        signal = (signal - median) / max(scale, 1e-6)
    elif config.normalization == "zscore":
        signal = (signal - signal.mean()) / max(float(signal.std()), 1e-6)
    else:
        raise ValueError(f"Unsupported raw normalization: {config.normalization}")
    return np.clip(np.nan_to_num(signal), -config.clipping, config.clipping).astype(np.float32)


def extract_and_preprocess_windows(sample: dict[str, Any], *, original_sfreq: float, config: RawPreprocessingConfig) -> tuple[np.ndarray, dict[str, Any]]:
    if original_sfreq <= 0:
        raise ValueError("original_sfreq must be positive.")
    source, time_origin = _source_windows(sample, float(original_sfreq), config.window_sec)
    output = np.empty((source.shape[0], source.shape[1], int(round(config.target_sfreq * config.window_sec))), dtype=np.float32)
    for window in range(source.shape[0]):
        for channel in range(source.shape[1]):
            output[window, channel] = _preprocess_1d(source[window, channel], float(original_sfreq), config)
    audit = {
        **asdict(config),
        "original_sfreq": float(original_sfreq),
        "time_origin": time_origin,
        "source_shape": list(source.shape),
        "output_shape": list(output.shape),
        "valid_windows": int(output.shape[0]),
        "finite_ratio": float(np.isfinite(output).mean()),
    }
    return output, audit


__all__ = ["RawPreprocessingConfig", "extract_and_preprocess_windows"]
