from __future__ import annotations

import math

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from .robust_rank import percentile_rank_channels
from .schema import RawRunRecord
from .signal_quality import robust_scale_channels, robust_z_from_baseline


ALGORITHM_VERSION = "ll_hfer_recruitment_v1"


def _rolling(values: np.ndarray, width: int, step: int) -> tuple[np.ndarray, np.ndarray]:
    starts = np.arange(0, max(values.shape[1] - width + 1, 0), step, dtype=int)
    if not starts.size: return np.empty((values.shape[0], 0)), starts
    cumulative = np.pad(np.cumsum(values, axis=1), ((0, 0), (1, 0)))
    return (cumulative[:, starts + width] - cumulative[:, starts]) / width, starts


def recruitment_time_from_score(score: np.ndarray, times: np.ndarray, threshold: float = 3.0, persistence_sec: float = .25) -> np.ndarray:
    output = np.full(score.shape[0], np.nan); step = float(np.median(np.diff(times))) if len(times) > 1 else persistence_sec
    persistence = max(1, math.ceil(persistence_sec / max(step, 1e-6)))
    for channel in range(score.shape[0]):
        active = (score[channel] > threshold) & (times >= 0)
        if len(active) < persistence: continue
        hits = np.convolve(active.astype(int), np.ones(persistence, dtype=int), mode="valid")
        found = np.where(hits >= persistence)[0]
        if found.size: output[channel] = times[found[0]]
    return output


def propagation_scores_from_times(times: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    times = np.asarray(times, dtype=float); valid = np.asarray(valid_mask, dtype=bool)
    recruited = valid & np.isfinite(times); output = np.full(times.shape, np.nan); output[valid] = 0.0
    if recruited.sum() >= 4:
        ranked = percentile_rank_channels(times, recruited, high_is_abnormal=False)
        output[recruited] = ranked[recruited]
    elif recruited.any():
        order = np.argsort(times[recruited], kind="stable"); values = np.linspace(1, 0, recruited.sum()); temp = np.zeros(recruited.sum()); temp[order] = values; output[recruited] = temp
    return output


def _fraction(times: np.ndarray, mask: np.ndarray, limit: float) -> float:
    return float((np.isfinite(times[mask]) & (times[mask] <= limit)).mean()) if mask.any() else np.nan


def compute_propagation(record: RawRunRecord) -> tuple[np.ndarray, pd.DataFrame, dict]:
    fs = record.sampling_rate; data, valid = robust_scale_channels(record.signal, record.valid_channel_mask); indices = np.where(valid)[0]
    output = np.full(len(record.channel_names), np.nan)
    if indices.size < 4:
        return output, pd.DataFrame(), {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "propagation_valid": False, "invalid_reason": "fewer_than_four_valid_channels"}
    selected = np.nan_to_num(data[indices]); width, step = max(3, round(.25 * fs)), max(1, round(.05 * fs))
    ll, starts = _rolling(np.abs(np.diff(selected, axis=1)), max(2, width - 1), step)
    hfer_available = fs >= 200
    if hfer_available:
        try:
            high = np.abs(hilbert(sosfiltfilt(butter(4, [30, 80], btype="bandpass", fs=fs, output="sos"), selected, axis=1), axis=1)) ** 2
            low = np.abs(hilbert(sosfiltfilt(butter(4, [4, 12], btype="bandpass", fs=fs, output="sos"), selected, axis=1), axis=1)) ** 2
            hfer, hstarts = _rolling(np.log1p(high / (low + 1e-8)), width, step); n = min(ll.shape[1], hfer.shape[1]); ll, hfer, starts = ll[:, :n], hfer[:, :n], starts[:n]
        except ValueError:
            hfer_available = False
    times = (starts + width / 2 - record.onset_sample) / fs; baseline = (times >= -10) & (times < -1)
    if baseline.sum() < 5:
        return output, pd.DataFrame(), {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "propagation_valid": False, "invalid_reason": "insufficient_preictal_baseline"}
    score = robust_z_from_baseline(ll, baseline)
    if hfer_available: score = np.maximum(score, robust_z_from_baseline(hfer, baseline))
    recruited_local = recruitment_time_from_score(score, times); recruited = np.full(len(record.channel_names), np.nan); recruited[indices] = recruited_local
    output = propagation_scores_from_times(recruited, valid)
    target = record.clinical_target_mask & valid; outside = ~record.clinical_target_mask & valid
    tm = float(np.nanmedian(recruited[target])) if np.isfinite(recruited[target]).any() else np.nan
    om = float(np.nanmedian(recruited[outside])) if np.isfinite(recruited[outside]).any() else np.nan
    seizure = {
        "patient_key": record.patient_key, "center": record.center, "seizure_id": record.seizure_id,
        "target_recruitment_median": tm, "outside_recruitment_median": om,
        "target_to_outside_delay": om - tm if np.isfinite([om, tm]).all() else np.nan,
        "outside_recruited_1s_fraction": _fraction(recruited, outside, 1),
        "outside_recruited_3s_fraction": _fraction(recruited, outside, 3),
        "outside_recruited_5s_fraction": _fraction(recruited, outside, 5),
        "target_recruited_1s_fraction": _fraction(recruited, target, 1),
        "target_recruited_3s_fraction": _fraction(recruited, target, 3),
        "earliest_outside_recruitment_time": float(np.nanmin(recruited[outside])) if np.isfinite(recruited[outside]).any() else np.nan,
        "propagation_valid": True, "hfer_available": hfer_available,
        "threshold": 3.0, "persistence_sec": .25, "algorithm_version": ALGORITHM_VERSION,
    }
    channel = pd.DataFrame({"patient_key": record.patient_key, "seizure_id": record.seizure_id, "channel": record.channel_names, "clinical_target": record.clinical_target_mask.astype(int), "valid": valid.astype(int), "recruitment_time": recruited, "propagation": output})
    return output, channel, seizure


__all__ = ["ALGORITHM_VERSION", "compute_propagation", "propagation_scores_from_times", "recruitment_time_from_score"]
