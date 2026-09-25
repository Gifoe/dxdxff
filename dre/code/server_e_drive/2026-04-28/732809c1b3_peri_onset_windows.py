from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data_interface import SeizureRecord


@dataclass
class PeriOnsetWindowSet:
    subject_id: str
    seizure_id: str
    signal_segment: np.ndarray
    sfreq: float
    channel_names: list[str]
    labels: np.ndarray
    window_starts: np.ndarray
    window_ends: np.ndarray
    window_centers_sec: np.ndarray
    baseline_mask: np.ndarray
    pretransition_mask: np.ndarray
    postonset_mask: np.ndarray


def create_peri_onset_windows(
    seizure: SeizureRecord,
    pre_sec: float = 10.0,
    post_sec: float = 10.0,
    window_sec: float = 0.250,
    stride_sec: float = 0.150,
    baseline_start_sec: float = -10.0,
    baseline_end_sec: float = -2.0,
) -> PeriOnsetWindowSet | None:
    signal = np.asarray(seizure.signal, dtype=np.float32)
    if signal.ndim != 2 or signal.shape[0] == 0:
        return None

    sfreq = float(seizure.sfreq)
    onset_sample = int(round(float(seizure.seizure_onset_sec) * sfreq))
    segment_start = onset_sample - int(round(float(pre_sec) * sfreq))
    segment_end = onset_sample + int(round(float(post_sec) * sfreq))
    if segment_start < 0 or segment_end > signal.shape[-1] or segment_end <= segment_start:
        return None

    signal_segment = signal[:, segment_start:segment_end].astype(np.float32, copy=False)
    window_samples = max(1, int(round(float(window_sec) * sfreq)))
    stride_samples = max(1, int(round(float(stride_sec) * sfreq)))
    if signal_segment.shape[-1] < window_samples:
        return None

    starts = np.arange(0, signal_segment.shape[-1] - window_samples + 1, stride_samples, dtype=np.int64)
    ends = starts + window_samples
    if starts.size == 0:
        return None

    centers_sec = ((starts + ends) / 2.0) / sfreq - float(pre_sec)
    baseline_mask = (centers_sec >= float(baseline_start_sec)) & (centers_sec <= float(baseline_end_sec))
    pretransition_mask = (centers_sec > float(baseline_end_sec)) & (centers_sec < 0.0)
    postonset_mask = (centers_sec >= 0.0) & (centers_sec <= float(post_sec))

    if not np.any(baseline_mask):
        return None

    return PeriOnsetWindowSet(
        subject_id=seizure.subject_id,
        seizure_id=seizure.seizure_id,
        signal_segment=signal_segment,
        sfreq=sfreq,
        channel_names=list(seizure.channel_names),
        labels=np.asarray(seizure.labels, dtype=np.float32),
        window_starts=starts,
        window_ends=ends,
        window_centers_sec=centers_sec.astype(np.float32),
        baseline_mask=baseline_mask.astype(bool),
        pretransition_mask=pretransition_mask.astype(bool),
        postonset_mask=postonset_mask.astype(bool),
    )


__all__ = ["PeriOnsetWindowSet", "create_peri_onset_windows"]

