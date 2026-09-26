"""Nine descriptors, same-seizure reference, and four-view expansion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.signal import welch

from .data import PatientRecord


DESCRIPTOR_NAMES = (
    "log_delta_power",
    "log_theta_power",
    "log_beta_power",
    "log_low_gamma_power",
    "log_high_gamma_power",
    "root_mean_square",
    "variance",
    "line_length_per_second",
    "spectral_entropy",
)

BANDS = ((1.0, 4.0), (4.0, 8.0), (13.0, 30.0), (30.0, 80.0), (80.0, 150.0))


def compute_descriptors(waveform: np.ndarray, sampling_rate: float) -> np.ndarray:
    """Compute the nine paper descriptors for [window, channel, sample] input."""
    signal = np.asarray(waveform, dtype=np.float64)
    if signal.ndim != 3:
        raise ValueError("waveform must be [window, channel, sample]")
    frequencies, density = welch(
        signal, fs=sampling_rate, axis=-1, nperseg=min(signal.shape[-1], 256)
    )
    powers = []
    for lower, upper in BANDS:
        mask = (frequencies >= lower) & (frequencies < upper)
        band = np.trapz(density[..., mask], frequencies[mask], axis=-1)
        powers.append(np.log1p(np.maximum(band, 0.0)))
    rms = np.sqrt(np.mean(np.square(signal), axis=-1) + 1e-8)
    variance = np.var(signal, axis=-1)
    duration = signal.shape[-1] / sampling_rate
    line_length = np.sum(np.abs(np.diff(signal, axis=-1)), axis=-1) / max(duration, 1e-8)
    probability = density / np.maximum(density.sum(axis=-1, keepdims=True), 1e-12)
    entropy = -np.sum(probability * np.log(probability + 1e-12), axis=-1)
    entropy /= np.log(max(probability.shape[-1], 2))
    return np.stack([*powers, rms, variance, line_length, entropy], axis=-1).astype(np.float32)


def four_view_expansion(
    descriptors: np.ndarray,
    window_times: np.ndarray,
    valid: np.ndarray,
    epsilon: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray]:
    """Build ABS, DIFF, Z-DIFF, and LOG-R using negative-time windows only."""
    x = np.asarray(descriptors, dtype=np.float32)
    times = np.asarray(window_times)
    mask = np.asarray(valid, dtype=bool)
    if x.ndim != 4 or x.shape[-1] != 9:
        raise ValueError("descriptors must be [seizure, window, channel, 9]")
    reference_mask = mask & (times[:, :, None] < 0)
    count = reference_mask.sum(axis=1, keepdims=True)
    if np.any((mask.any(axis=1, keepdims=True)) & (count == 0)):
        raise ValueError("Every retained seizure-channel needs a pre-onset reference")
    weighted = np.where(reference_mask[..., None], x, 0.0)
    mean = weighted.sum(axis=1, keepdims=True) / np.maximum(count[..., None], 1)
    centered_reference = np.where(reference_mask[..., None], x - mean, 0.0)
    variance = np.square(centered_reference).sum(axis=1, keepdims=True)
    variance /= np.maximum(count[..., None], 1)
    deviation = np.sqrt(variance)
    difference = x - mean
    z_difference = difference / (deviation + epsilon)
    log_ratio = np.log((np.abs(x) + epsilon) / (np.abs(mean) + epsilon))
    expanded = np.concatenate((x, difference, z_difference, log_ratio), axis=-1)
    expanded = np.where(mask[..., None] & np.isfinite(expanded), expanded, 0.0)
    return expanded.astype(np.float32), mask


@dataclass(frozen=True)
class Standardizer:
    mean: np.ndarray
    scale: np.ndarray

    def apply(self, values: np.ndarray, valid: np.ndarray) -> np.ndarray:
        normalized = (values - self.mean) / self.scale
        return np.where(valid[..., None], normalized, 0.0).astype(np.float32)


def fit_standardizer(records: Iterable[PatientRecord]) -> Standardizer:
    """Fit dataset-level statistics using outer-fit patients only."""
    rows = []
    for record in records:
        expanded, valid = four_view_expansion(
            record.descriptors, record.window_times, record.valid
        )
        rows.append(expanded[valid])
    matrix = np.concatenate(rows, axis=0)
    mean = np.nanmean(matrix, axis=0)
    scale = np.nanstd(matrix, axis=0)
    scale = np.where(scale > 1e-6, scale, 1.0)
    return Standardizer(mean.astype(np.float32), scale.astype(np.float32))


def feature_baseline_vector(record: PatientRecord) -> np.ndarray:
    """Return the code-aligned 88-dimensional channel representation."""
    expanded, valid = four_view_expansion(
        record.descriptors, record.window_times, record.valid
    )
    difference = expanded[..., 9:18]
    z_difference = expanded[..., 18:27]
    # The four amplitude/state descriptors receive two additional views.
    selected = np.asarray([4, 5, 6, 7])
    extra = np.concatenate(
        (z_difference[..., selected], np.abs(difference[..., selected])), axis=-1
    )
    window44 = np.concatenate((expanded, extra), axis=-1)
    seizure_channel = np.zeros(
        (window44.shape[0], window44.shape[2], window44.shape[3]), dtype=np.float32
    )
    seizure_valid = valid.any(axis=1)
    for seizure in range(window44.shape[0]):
        weights = valid[seizure, :, :, None]
        seizure_channel[seizure] = (
            (window44[seizure] * weights).sum(axis=0)
            / np.maximum(weights.sum(axis=0), 1)
        )
    mean = np.zeros_like(seizure_channel[0])
    std = np.zeros_like(seizure_channel[0])
    for channel in range(window44.shape[2]):
        values = seizure_channel[seizure_valid[:, channel], channel]
        mean[channel] = values.mean(axis=0)
        std[channel] = values.std(axis=0)
    return np.concatenate((mean, std), axis=-1).astype(np.float32)
