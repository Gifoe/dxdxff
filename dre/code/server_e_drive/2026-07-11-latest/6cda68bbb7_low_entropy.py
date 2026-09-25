from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.signal import welch

from .robust_rank import percentile_rank_channels
from .schema import RawRunRecord
from .signal_quality import time_slice


ALGORITHM_VERSION = "preictal_spectral_entropy_v1"


def normalized_spectral_entropy(values: np.ndarray, sampling_rate: float) -> float:
    data = np.asarray(values, dtype=float)
    if data.size < 16 or not np.isfinite(data).all(): return np.nan
    frequency, power = welch(data, fs=sampling_rate, nperseg=min(data.size, max(16, int(round(sampling_rate / 2)))))
    upper = min(120.0, .40 * sampling_rate); keep = (frequency >= 1.0) & (frequency <= upper)
    power = np.maximum(power[keep], 0.0)
    if power.size < 2 or power.sum() <= 0: return np.nan
    probability = power / power.sum(); return float(-(probability * np.log(probability + 1e-12)).sum() / np.log(power.size))


def compute_low_entropy(record: RawRunRecord) -> tuple[np.ndarray, pd.DataFrame, dict]:
    fs = record.sampling_rate; segment = time_slice(record.onset_sample, fs, record.signal.shape[1], -10, -1); data = np.asarray(record.signal[:, segment], dtype=float)
    width = max(16, int(round(fs))); n_windows = data.shape[1] // width; scores = np.full(len(record.channel_names), np.nan); valid_counts = np.zeros(len(record.channel_names), dtype=int); artifact_counts = np.zeros(len(record.channel_names), dtype=int)
    if n_windows:
        absolute = np.abs(data); med = np.nanmedian(absolute, axis=1); mad = np.nanmedian(np.abs(absolute - med[:, None]), axis=1); threshold = med + 8.0 * mad
        for channel in range(data.shape[0]):
            if not record.valid_channel_mask[channel]: continue
            values = []
            for index in range(n_windows):
                local = data[channel, index * width:(index + 1) * width]
                if not np.isfinite(local).all() or np.nanstd(local) < 1e-8 or np.nanmax(np.abs(local)) > threshold[channel]:
                    artifact_counts[channel] += 1; continue
                entropy = normalized_spectral_entropy(local, fs)
                if np.isfinite(entropy): values.append(entropy)
            valid_counts[channel] = len(values)
            if len(values) >= 5: scores[channel] = float(np.median(values))
    rank = percentile_rank_channels(scores, np.isfinite(scores), high_is_abnormal=False)
    channel = pd.DataFrame({"patient_key": record.patient_key, "seizure_id": record.seizure_id, "channel": record.channel_names, "entropy": scores, "low_entropy": rank, "n_valid_subwindows": valid_counts, "artifact_fraction": artifact_counts / max(n_windows, 1), "entropy_valid": np.isfinite(rank), "invalid_reason": np.where(np.isfinite(rank), "", "fewer_than_five_clean_subwindows")})
    valid = np.isfinite(rank).sum() >= 4
    quality = {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "entropy_valid": valid, "n_preictal_subwindows": n_windows, "n_valid_channels": int(np.isfinite(rank).sum()), "artifact_fraction": float(artifact_counts.sum() / max(n_windows * len(record.channel_names), 1)), "invalid_reason": "" if valid else "fewer_than_four_valid_channels", "algorithm_version": ALGORITHM_VERSION}
    return rank, channel, quality


__all__ = ["ALGORITHM_VERSION", "compute_low_entropy", "normalized_spectral_entropy"]
