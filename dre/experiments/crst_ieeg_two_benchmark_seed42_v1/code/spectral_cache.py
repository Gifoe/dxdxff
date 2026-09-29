"""Label-free deterministic physical-frequency transform of 60-s raw records.

This is a cacheable front end, not a learned or dataset-specific branch. The
same function handles ictal 250 Hz and Omni 1000 Hz physical-microvolt clips.
It never inspects labels, patient IDs, center IDs, or train/test membership.
"""

from __future__ import annotations

import math

import numpy as np
from scipy import signal


FREQUENCIES_HZ = np.geomspace(1.0, 300.0, 64).astype(np.float32)
BANDS_HZ = ((4, 8), (8, 13), (13, 30), (30, 55), (65, 100))


def physical_frequency_mask(fs: float) -> np.ndarray:
    if not 100 <= fs <= 2000:
        raise ValueError(f"Unsupported physical sampling rate: {fs}")
    return (FREQUENCIES_HZ <= 0.45 * fs).astype(np.float32)


def spectral_patches(raw: np.ndarray, fs: float, valid_samples: int | None = None,
                     valid_start: int = 0):
    """Return [C,59,64,8] log-amplitude patches and valid-window mask.

    A 2-s Hann STFT has 0.5-Hz resolution and 0.25-s hops. Eight within-window
    time positions are retained, not averaged. The 64-bin interpolation is
    physical-Hz based and masks bins above 0.45*fs.
    """
    x = np.asarray(raw, dtype=np.float32)
    if x.ndim != 2 or x.shape[0] == 0 or not np.isfinite(x).all():
        raise ValueError("Expected finite nonempty [channel,sample] raw EEG")
    fs = float(fs)
    window = int(round(2 * fs))
    hop = int(round(fs))
    if x.shape[1] != int(round(60 * fs)) or window < 2 or hop < 1:
        raise ValueError("Expected exact 60-s sample tensor")
    valid_samples = x.shape[1] if valid_samples is None else int(valid_samples)
    valid_start = int(valid_start)
    if not (0 <= valid_start and window <= valid_samples and
            valid_start + valid_samples <= x.shape[1]):
        raise ValueError("Invalid original signal duration")
    f_native, t_native, stft = signal.stft(
        x, fs=fs, window="hann", nperseg=window, noverlap=window - int(round(fs / 4)),
        nfft=window, boundary="zeros", padded=True, axis=-1)
    magnitude = np.log1p(np.abs(stft)).astype(np.float32)
    target_t = (np.arange(59)[:, None] + (np.arange(8)[None, :] + 0.5) / 4).ravel()
    time_idx = np.clip(np.searchsorted(t_native, target_t), 1, len(t_native) - 1)
    left_t, right_t = t_native[time_idx - 1], t_native[time_idx]
    time_weight = ((target_t - left_t) / np.maximum(right_t - left_t, 1e-12)).astype(np.float32)
    at_time = magnitude[:, :, time_idx - 1] * (1 - time_weight) + magnitude[:, :, time_idx] * time_weight
    freq_idx = np.clip(np.searchsorted(f_native, FREQUENCIES_HZ), 1, len(f_native) - 1)
    left_f, right_f = f_native[freq_idx - 1], f_native[freq_idx]
    freq_weight = ((FREQUENCIES_HZ - left_f) / np.maximum(right_f - left_f, 1e-12)).astype(np.float32)
    patches = at_time[:, freq_idx - 1, :] * (1 - freq_weight)[None, :, None]
    patches += at_time[:, freq_idx, :] * freq_weight[None, :, None]
    patches = patches.reshape(x.shape[0], 64, 59, 8).transpose(0, 2, 1, 3)
    fmask = physical_frequency_mask(fs)
    patches *= fmask[None, None, :, None]
    starts = np.arange(59) * hop
    wmask = (starts >= valid_start) & (starts + window <= valid_start + valid_samples)
    patches[:, ~wmask] = 0
    return patches.astype(np.float32), fmask, np.broadcast_to(wmask[None], (x.shape[0], 59)).copy()


def connectivity_edges(raw: np.ndarray, fs: float, valid_samples: int | None = None,
                       valid_start: int = 0):
    """Return label-free [C,C,15] AEC/coherence/PLI and five-band availability.

    The coherence statistic uses normalized complex analytic covariance across
    the valid 60-s record; PLI uses at most 256 evenly spaced analytic samples.
    Undefined bands are zero with an explicit returned availability mask.
    """
    x = np.asarray(raw, dtype=np.float64)
    fs = float(fs)
    valid = x.shape[1] if valid_samples is None else int(valid_samples)
    valid_start = int(valid_start)
    if x.ndim != 2 or valid < int(round(2 * fs)) or valid_start < 0 or valid_start + valid > x.shape[1]:
        raise ValueError("Invalid raw shape or valid-sample range")
    x = x[:, valid_start:valid_start + valid]
    c = x.shape[0]
    out = np.zeros((c, c, 15), dtype=np.float32)
    available = np.zeros(5, dtype=np.float32)
    for band_idx, (lo, hi) in enumerate(BANDS_HZ):
        if hi > 0.45 * fs:
            continue
        sos = signal.butter(3, (lo, hi), btype="bandpass", fs=fs, output="sos")
        analytic = signal.hilbert(signal.sosfiltfilt(sos, x, axis=-1), axis=-1)
        stride = max(1, math.ceil(analytic.shape[1] / 256))
        analytic = analytic[:, ::stride]
        envelope = np.abs(analytic)
        env = envelope - envelope.mean(axis=1, keepdims=True)
        env /= np.sqrt(np.sum(env * env, axis=1, keepdims=True)).clip(min=1e-12)
        aec = env @ env.T
        cov = analytic @ analytic.conj().T
        power = np.real(np.diag(cov)).clip(min=1e-12)
        coherence = np.abs(cov) ** 2 / np.outer(power, power)
        phase = analytic / np.abs(analytic).clip(min=1e-12)
        pli = np.zeros((c, c), dtype=np.float64)
        for sample in range(phase.shape[1]):
            z = phase[:, sample]
            pli += np.sign(np.imag(z[:, None] * z.conj()[None, :]))
        pli = np.abs(pli / phase.shape[1])
        out[:, :, 3 * band_idx:3 * band_idx + 3] = np.stack(
            (aec, coherence, pli), axis=-1).astype(np.float32)
        available[band_idx] = 1
    if not np.isfinite(out).all():
        raise RuntimeError("Non-finite label-free connectivity")
    return out, available
