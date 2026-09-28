"""Fixed, label-free 2 s / 1 s STFT aligned to historical A1 windows."""

from __future__ import annotations

import numpy as np


N_BINS = 32
LOW_HZ = 1.0
HIGH_HZ = 120.0
WINDOW_SECONDS = 2
HOP_SECONDS = 1
EPS = 1e-8


def log_frequency_stft(waveform: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return [59,C,32] from a [C,60*sample_rate] physical-uV clip.

    Each row is the Hann-windowed STFT power at its 2-second, 1-second-hop
    frame, averaged into frozen physical-frequency log bins. No per-record or
    per-window variance normalization is performed.
    """
    x = np.asarray(waveform, dtype=np.float32)
    if x.ndim != 2 or x.shape[1] != 60 * sample_rate:
        raise ValueError(f"Expected [channels,{60 * sample_rate}], got {x.shape}")
    if sample_rate < 240:
        raise ValueError("The frozen 120 Hz upper bin requires Nyquist >=120 Hz")
    if not np.isfinite(x).all():
        raise ValueError("Nonfinite input signal")
    width, hop = 2 * sample_rate, sample_rate
    frames = np.lib.stride_tricks.sliding_window_view(x, width, axis=-1)[:, ::hop]
    if frames.shape[1] != 59:
        raise RuntimeError(f"A1 window grid mismatch: {frames.shape}")
    window = np.hanning(width).astype(np.float32)
    power = np.abs(np.fft.rfft(frames * window, axis=-1)) ** 2
    frequencies = np.fft.rfftfreq(width, 1.0 / sample_rate)
    centers = np.geomspace(LOW_HZ, HIGH_HZ, N_BINS)
    upper = np.searchsorted(frequencies, centers, side="left")
    lower = np.maximum(0, upper - 1)
    span = frequencies[upper] - frequencies[lower]
    interpolation = np.divide(centers - frequencies[lower], span,
                              out=np.zeros_like(centers), where=span > 0)
    sampled_power = ((1.0 - interpolation) * power[:, :, lower] +
                     interpolation * power[:, :, upper])
    output = np.log(sampled_power.transpose(1, 0, 2) + EPS).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("Nonfinite log STFT output")
    return output


def fit_train_frequency_normalizer(train_arrays: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Train-only global per-frequency moments; never fit on validation/test."""
    total = np.zeros(N_BINS, dtype=np.float64)
    total_sq = np.zeros(N_BINS, dtype=np.float64)
    count = 0
    for array in train_arrays:
        flat = np.asarray(array, dtype=np.float32).reshape(-1, N_BINS)
        total += flat.sum(axis=0, dtype=np.float64)
        total_sq += np.square(flat.astype(np.float64)).sum(axis=0)
        count += len(flat)
    if not count:
        raise ValueError("No train STFT frames")
    mean = total / count
    std = np.sqrt(np.maximum(total_sq / count - mean * mean, 1e-8))
    return mean.astype(np.float32), std.astype(np.float32)
