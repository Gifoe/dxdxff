from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from .robust_rank import percentile_rank_channels
from .schema import RawRunRecord
from .signal_quality import robust_scale_channels, robust_z_from_baseline


ALGORITHM_VERSION = "ei_hfer_page_hinkley_v2"


def _smooth(values: np.ndarray, width: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, max(values.shape[1] - width + 1, 0), step, dtype=int)
    if not starts.size: return np.empty((values.shape[0], 0)), starts
    cumulative = np.pad(np.cumsum(values, axis=1), ((0, 0), (1, 0)))
    return (cumulative[:, starts + width] - cumulative[:, starts]) / width, starts


def detect_persistent(z: np.ndarray, times: np.ndarray, threshold: float = 3.0, persistence_sec: float = .25) -> np.ndarray:
    output = np.full(z.shape[0], np.nan); step = float(np.median(np.diff(times))) if len(times) > 1 else persistence_sec
    count = max(1, int(np.ceil(persistence_sec / max(step, 1e-6))))
    for channel in range(z.shape[0]):
        above = (z[channel] > threshold) & (times >= 0)
        if above.size < count: continue
        hits = np.convolve(above.astype(int), np.ones(count, dtype=int), mode="valid")
        index = np.where(hits >= count)[0]
        if index.size: output[channel] = times[index[0]]
    return output


def detect_page_hinkley(z: np.ndarray, times: np.ndarray, threshold: float = 5.0, delta: float = .05, persistence_sec: float = .25) -> np.ndarray:
    """Fixed one-sided Page-Hinkley/CUSUM detector with persistence."""
    statistic = np.zeros_like(z, dtype=float)
    for index in range(1, z.shape[1]):
        statistic[:, index] = np.maximum(0.0, statistic[:, index - 1] + np.nan_to_num(z[:, index], nan=0.0) - delta)
    return detect_persistent(statistic, times, threshold=threshold, persistence_sec=persistence_sec)


def compute_ei(record: RawRunRecord) -> tuple[np.ndarray, pd.DataFrame, dict]:
    fs = float(record.sampling_rate); upper = min(127.0, .40 * fs)
    invalid = np.full(len(record.channel_names), np.nan)
    if upper <= 30.0:
        return invalid, pd.DataFrame(), {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "ei_valid": False, "invalid_reason": "LOW_SAMPLING_RATE_EI_INVALID", "effective_high_upper": upper}
    data, valid = robust_scale_channels(record.signal, record.valid_channel_mask); indices = np.where(valid)[0]
    if indices.size < 4:
        return invalid, pd.DataFrame(), {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "ei_valid": False, "invalid_reason": "fewer_than_four_valid_channels"}
    selected = np.nan_to_num(data[indices]); low_sos = butter(4, [4, 12], btype="bandpass", fs=fs, output="sos"); high_sos = butter(4, [12, upper], btype="bandpass", fs=fs, output="sos")
    try:
        low = np.abs(hilbert(sosfiltfilt(low_sos, selected, axis=1), axis=1)) ** 2
        high = np.abs(hilbert(sosfiltfilt(high_sos, selected, axis=1), axis=1)) ** 2
    except ValueError:
        return invalid, pd.DataFrame(), {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "ei_valid": False, "invalid_reason": "signal_too_short_for_zero_phase_filter"}
    width, step = max(4, round(.25 * fs)), max(1, round(.05 * fs))
    ratio = np.log1p(high / (low + 1e-8)); trajectory, starts = _smooth(ratio, width, step)
    times = (starts + width / 2 - record.onset_sample) / fs; baseline = (times >= -10) & (times < -1)
    if baseline.sum() < 5:
        return invalid, pd.DataFrame(), {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "ei_valid": False, "invalid_reason": "insufficient_preictal_baseline"}
    z = robust_z_from_baseline(trajectory, baseline); recruitment = detect_page_hinkley(z, times)
    early = (times >= 0) & (times <= 10); amplitude = np.nanquantile(np.maximum(z[:, early], 0), .75, axis=1) if early.any() else np.zeros(indices.size)
    latency = np.where(np.isfinite(recruitment), np.exp(-np.maximum(recruitment, 0) / 5.0), 0.0)
    raw = amplitude * latency; rank = percentile_rank_channels(raw, np.isfinite(raw), high_is_abnormal=True); output = invalid.copy(); output[indices] = rank
    channel = pd.DataFrame({"patient_key": record.patient_key, "seizure_id": record.seizure_id, "channel": [record.channel_names[i] for i in indices], "detected": np.isfinite(recruitment), "recruitment_time": recruitment, "amplitude_score": amplitude, "latency_score": latency, "ei_raw": raw, "ei": rank, "ei_valid": True, "invalid_reason": ""})
    return output, channel, {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "ei_valid": True, "invalid_reason": "", "effective_high_upper": upper, "change_detector": "fixed_page_hinkley", "page_hinkley_threshold": 5.0, "page_hinkley_delta": .05, "algorithm_version": ALGORITHM_VERSION}


__all__ = ["ALGORITHM_VERSION", "compute_ei", "detect_page_hinkley", "detect_persistent"]
