from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from .hfo_network import hfo_hub_rank
from .robust_rank import percentile_rank_channels
from .schema import RawRunRecord
from .signal_quality import time_slice


ALGORITHM_VERSION = "conditional_hfo_fixed_detector_v4"


@dataclass
class HFOResult:
    score: np.ndarray
    hub: np.ndarray
    channel_table: pd.DataFrame
    event_table: pd.DataFrame
    quality: dict


def _preictal_clean_duration(record: RawRunRecord) -> float:
    fs = record.sampling_rate; preictal = np.asarray(record.signal[:, :record.onset_sample], dtype=float)
    if preictal.shape[1] < fs: return 0.0
    median = np.nanmedian(preictal, axis=1, keepdims=True); mad = np.nanmedian(np.abs(preictal - median), axis=1, keepdims=True); scaled = np.abs((preictal - median) / (1.4826 * mad + 1e-8))
    valid = np.asarray(record.valid_channel_mask, dtype=bool); global_artifact = (scaled[valid] > 20.0).mean(axis=0) > .5
    width = max(1, int(round(fs))); windows = preictal.shape[1] // width
    clean = sum(not global_artifact[index * width:(index + 1) * width].any() for index in range(windows))
    return float(clean * width / fs)


def hfo_eligibility(record: RawRunRecord) -> tuple[bool, str, bool]:
    preictal = record.onset_sample / record.sampling_rate
    valid_fraction = float(np.asarray(record.valid_channel_mask, dtype=bool).mean())
    ripple = record.sampling_rate >= 1000
    fast = record.sampling_rate >= 2000
    if not ripple: return False, "SAMPLING_RATE_BELOW_1000", fast
    if preictal < 30 or _preictal_clean_duration(record) < 30: return False, "PREICTAL_CLEAN_DURATION_BELOW_30S", fast
    if valid_fraction < .80: return False, "VALID_CHANNEL_FRACTION_BELOW_0P80", fast
    return True, "", fast


def _events(envelope: np.ndarray, fs: float, low: float, threshold: np.ndarray, minimum_cycles: int = 6, merge_gap_sec: float = .010) -> list[np.ndarray]:
    minimum = max(1, int(np.ceil(minimum_cycles * fs / low))); merge_gap = max(1, int(round(merge_gap_sec * fs))); output = []
    for channel in range(envelope.shape[0]):
        active = envelope[channel] > threshold[channel]
        edges = np.diff(np.pad(active.astype(int), (1, 1))); starts = np.where(edges == 1)[0]; ends = np.where(edges == -1)[0]
        intervals: list[list[int]] = []
        for start, end in zip(starts, ends):
            if end - start < minimum: continue
            if intervals and start - intervals[-1][1] <= merge_gap: intervals[-1][1] = end
            else: intervals.append([int(start), int(end)])
        output.append(np.asarray([start / fs for start, _ in intervals], dtype=float))
    return output


def _band_events(data: np.ndarray, fs: float, low: float, high: float) -> tuple[list[np.ndarray], np.ndarray]:
    upper = min(high, .45 * fs)
    if upper <= low: return [np.array([]) for _ in range(data.shape[0])], np.full_like(data, np.nan)
    filtered = sosfiltfilt(butter(4, [low, upper], btype="bandpass", fs=fs, output="sos"), data, axis=1)
    envelope = np.abs(hilbert(filtered, axis=1)); median = np.median(envelope, axis=1); mad = np.median(np.abs(envelope - median[:, None]), axis=1)
    threshold = median + 5.0 * 1.4826 * mad
    return _events(envelope, fs, low, threshold), envelope


def compute_hfo(record: RawRunRecord, *, disabled: bool = False) -> HFOResult:
    n = len(record.channel_names); invalid = np.full(n, np.nan); eligible, reason, fast_eligible = hfo_eligibility(record)
    if disabled: eligible, reason = False, "HFO_DISABLED_BY_CLI"
    if not eligible:
        quality = {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "hfo_valid": False, "eligible": False, "invalid_reason": reason, "sampling_rate": record.sampling_rate, "fast_ripple_eligible": fast_eligible}
        return HFOResult(invalid, invalid.copy(), pd.DataFrame(), pd.DataFrame(), quality)
    segment = time_slice(record.onset_sample, record.sampling_rate, record.signal.shape[1], -30, 0); data = np.asarray(record.signal[:, segment], dtype=float)
    median = np.nanmedian(data, axis=1, keepdims=True); mad = np.nanmedian(np.abs(data - median), axis=1, keepdims=True); scaled = (data - median) / (1.4826 * mad + 1e-8)
    ripple, ripple_env = _band_events(scaled, record.sampling_rate, 80, 250)
    fast = [np.array([]) for _ in range(n)]
    if fast_eligible: fast, _ = _band_events(scaled, record.sampling_rate, 250, 500)
    # Fixed spike detector: simultaneous high amplitude and high first derivative.
    amplitude = np.abs(scaled); derivative = np.abs(np.diff(scaled, axis=1, prepend=scaled[:, :1])); spike_mask = (amplitude > 6.0) & (derivative > 4.0)
    spike_events = [np.where(np.diff(np.pad(row.astype(int), (1, 0))) == 1)[0] / record.sampling_rate for row in spike_mask]
    spike_ripple = [np.asarray([event for event in ripple[c] if np.any(np.abs(spike_events[c] - event) <= .05)]) for c in range(n)]
    # Reject broadband artifacts: events coincident with >50% of valid channels within 10 ms.
    valid_count = max(int(record.valid_channel_mask.sum()), 1)
    for family in (ripple, fast, spike_ripple):
        original = [events.copy() for events in family]
        for c in range(n):
            family[c] = np.asarray([event for event in original[c] if sum(np.any(np.abs(other - event) <= .05) for other in original) / valid_count <= .5])
    duration = max(data.shape[1] / record.sampling_rate, 1e-6)
    ripple_rate = np.asarray([len(x) / duration for x in ripple]); fast_rate = np.asarray([len(x) / duration for x in fast]); sr_rate = np.asarray([len(x) / duration for x in spike_ripple])
    hub = hfo_hub_rank([np.sort(np.concatenate((ripple[c], fast[c], spike_ripple[c]))) for c in range(n)], record.valid_channel_mask)
    fast_rank = percentile_rank_channels(fast_rate, record.valid_channel_mask, high_is_abnormal=True) if fast_eligible else np.full(n, np.nan)
    ranks = [fast_rank, percentile_rank_channels(sr_rate, record.valid_channel_mask, high_is_abnormal=True), hub]
    stacked = np.stack(ranks); score = np.full(n, np.nan)
    for channel_index in range(n):
        available = stacked[:, channel_index][np.isfinite(stacked[:, channel_index])]
        if available.size: score[channel_index] = float(np.median(available))
    channel = pd.DataFrame({"patient_key": record.patient_key, "seizure_id": record.seizure_id, "channel": record.channel_names, "ripple_rate": ripple_rate, "fast_ripple_rate": fast_rate, "spike_ripple_rate": sr_rate, "hfo_hub": hub, "hfo": score})
    event_rows = []
    for c, name in enumerate(record.channel_names):
        for family, events in (("ripple", ripple[c]), ("fast_ripple", fast[c]), ("spike_ripple", spike_ripple[c])):
            event_rows.extend({"patient_key": record.patient_key, "seizure_id": record.seizure_id, "channel": name, "event_type": family, "event_onset_sec": float(value - 30.0)} for value in events)
    quality = {"patient_key": record.patient_key, "seizure_id": record.seizure_id, "hfo_valid": np.isfinite(score).sum() >= 4, "eligible": True, "invalid_reason": "", "sampling_rate": record.sampling_rate, "fast_ripple_eligible": fast_eligible, "preictal_clean_duration_sec": _preictal_clean_duration(record), "valid_channel_fraction": float(record.valid_channel_mask.mean()), "algorithm_version": ALGORITHM_VERSION}
    return HFOResult(score, hub, channel, pd.DataFrame(event_rows), quality)


__all__ = ["ALGORITHM_VERSION", "HFOResult", "compute_hfo", "hfo_eligibility"]
