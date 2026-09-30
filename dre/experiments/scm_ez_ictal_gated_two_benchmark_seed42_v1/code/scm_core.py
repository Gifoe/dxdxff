"""Core deterministic transforms and the <15K-parameter SCM-EZ network."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
import torch
from scipy.signal.windows import hann
from torch import nn


SEED = 42
N_STATES = 6
N_BINS = 16
BIN_EDGES_HZ = np.geomspace(1.0, 300.0, N_BINS + 1).astype(np.float64)
STATE_NAMES = ("PRE_EARLY", "PRE_LATE", "ONSET_EARLY", "ONSET_LATE", "SPREAD", "LATE")


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def visible_bins(fs: float) -> np.ndarray:
    """A bin is visible only when its complete physical interval is observable."""
    return BIN_EDGES_HZ[1:] <= 0.45 * float(fs) + 1e-12


def state_membership(centers: np.ndarray) -> np.ndarray:
    """Map explicit onset-relative centers to the six frozen physiological states."""
    t = np.asarray(centers, dtype=np.float64)
    if t.ndim != 1 or not len(t) or not np.isfinite(t).all() or np.any(np.diff(t) <= 0):
        raise ValueError("window centers must be finite, one-dimensional and strictly increasing")
    result = np.full(len(t), -1, dtype=np.int8)
    pre = t < 0
    if pre.any():
        midpoint = (float(t[pre].min()) + float(t[pre].max())) / 2.0
        result[pre & (t <= midpoint)] = 0
        result[pre & (t > midpoint)] = 1
    result[(t >= 0) & (t < 5)] = 2
    result[(t >= 5) & (t < 10)] = 3
    result[(t >= 10) & (t < 30)] = 4
    result[t >= 30] = 5
    if np.any(result < 0):
        raise RuntimeError("a valid center was not assigned to a state")
    return result


def window_spectra(raw: np.ndarray, fs: float, centers: np.ndarray, onset_sample: int,
                   valid_start: int, valid_samples: int) -> tuple[np.ndarray, np.ndarray]:
    """2-s Hann/rFFT physical-bin log-power at explicit real time centers.

    Returns [channel,window,16] and a [window] validity mask.  Window locations
    are derived from each real onset-relative center, not array ordinals.
    """
    x = np.asarray(raw, dtype=np.float32)
    t = np.asarray(centers, dtype=np.float64)
    fs = float(fs)
    n = int(round(2.0 * fs))
    if x.ndim != 2 or x.shape[1] != int(round(60.0 * fs)) or n < 2:
        raise ValueError("expected [channel,60 seconds] raw signal")
    if valid_start < 0 or valid_samples <= 0 or valid_start + valid_samples > x.shape[1]:
        raise ValueError("invalid source signal span")
    starts = np.rint(int(onset_sample) + (t - 1.0) * fs).astype(np.int64)
    valid = (starts >= int(valid_start)) & (starts + n <= int(valid_start + valid_samples))
    if not valid.all():
        raise RuntimeError("explicit cache centers do not align to the audited valid raw span")
    taper = hann(n, sym=False).astype(np.float32)
    segments = np.stack([x[:, start:start + n] for start in starts], axis=1)
    fft = np.fft.rfft(segments * taper[None, None, :], axis=-1)
    power = (fft.real * fft.real + fft.imag * fft.imag) / float(np.square(taper).sum())
    frequencies = np.fft.rfftfreq(n, d=1.0 / fs)
    output = np.zeros((x.shape[0], len(t), N_BINS), dtype=np.float32)
    availability = visible_bins(fs)
    for index, (lo, hi) in enumerate(zip(BIN_EDGES_HZ[:-1], BIN_EDGES_HZ[1:])):
        if not availability[index]:
            continue
        # Last edge is inclusive only for the final bin; adjacent bins never overlap.
        native = (frequencies >= lo) & (frequencies < hi)
        if not native.any():
            raise RuntimeError(f"no native FFT support in visible bin {index}")
        output[:, :, index] = np.log1p(power[:, :, native].sum(axis=-1)).astype(np.float32)
    if not np.isfinite(output).all():
        raise RuntimeError("non-finite window spectrum")
    return output, valid


def aggregate_states(spectra: np.ndarray, centers: np.ndarray,
                     window_valid: np.ndarray, frequency_valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Median valid windows into [channel,6,16] with [channel,6] state mask."""
    x = np.asarray(spectra, dtype=np.float32)
    states = state_membership(centers)
    wmask = np.asarray(window_valid, dtype=bool)
    fmask = np.asarray(frequency_valid, dtype=bool)
    if x.ndim != 3 or x.shape[1] != len(states) or x.shape[2] != N_BINS:
        raise ValueError("invalid spectral window tensor")
    if wmask.shape not in ((len(states),), x.shape[:2]):
        raise ValueError("invalid window-valid mask")
    if wmask.ndim == 1:
        wmask = np.broadcast_to(wmask[None], x.shape[:2])
    result = np.zeros((x.shape[0], N_STATES, N_BINS), dtype=np.float32)
    valid = np.zeros((x.shape[0], N_STATES), dtype=bool)
    for state in range(N_STATES):
        members = states == state
        if not members.any():
            continue
        for channel in range(x.shape[0]):
            selected = members & wmask[channel]
            if selected.any():
                result[channel, state] = np.median(x[channel, selected], axis=0)
                result[channel, state, ~fmask] = 0.0
                valid[channel, state] = True
    return result, valid


@dataclass
class RobustScaler:
    median: np.ndarray
    iqr: np.ndarray
    frequency_valid: np.ndarray

    def transform(self, x: np.ndarray) -> np.ndarray:
        value = (np.asarray(x, dtype=np.float32) - self.median) / self.iqr
        value = np.clip(value, -8.0, 8.0)
        value[..., ~self.frequency_valid] = 0.0
        return value.astype(np.float32)


def fit_robust_scaler(values_by_patient: list[np.ndarray], masks_by_patient: list[np.ndarray],
                      frequency_valid: np.ndarray) -> RobustScaler:
    med = np.zeros(N_BINS, dtype=np.float32)
    iqr = np.ones(N_BINS, dtype=np.float32)
    for frequency in np.flatnonzero(frequency_valid):
        pieces = []
        for values, mask in zip(values_by_patient, masks_by_patient):
            v = np.asarray(values, dtype=np.float32)
            m = np.asarray(mask, dtype=bool)
            pieces.append(v[..., frequency][m])
        joined = np.concatenate(pieces)
        if not len(joined) or not np.isfinite(joined).all():
            raise RuntimeError("empty/nonfinite FIT-only scaler bin")
        q25, q50, q75 = np.quantile(joined, (0.25, 0.5, 0.75))
        med[frequency] = q50
        iqr[frequency] = max(float(q75 - q25), 1e-6)
    return RobustScaler(med, iqr, np.asarray(frequency_valid, dtype=bool))


def patient_references(states: np.ndarray, state_valid: np.ndarray,
                       *, shuffled_reference: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Label-free current-channel-excluded patient reference per record/state."""
    x = np.asarray(states, dtype=np.float32)
    valid = np.asarray(state_valid, dtype=bool)
    channels = x.shape[0]
    reference = np.zeros_like(x)
    reference_valid = np.zeros_like(valid)
    fallback = 0
    for current in range(channels):
        source_current = current if shuffled_reference is None else int(shuffled_reference[current])
        for state in range(N_STATES):
            candidates = np.flatnonzero(valid[:, state])
            if len(candidates) >= 3:
                candidates = candidates[candidates != source_current]
            elif len(candidates):
                fallback += 1
            if len(candidates):
                reference[current, state] = np.median(x[candidates, state], axis=0)
                reference_valid[current, state] = True
    return reference, reference_valid, fallback


def scm_matrices(states: np.ndarray, state_valid: np.ndarray,
                 *, reference_permutation: np.ndarray | None = None) -> tuple[np.ndarray, int]:
    """Return exact [channel,34,6,6] Self+Patient comparison matrices."""
    x = np.asarray(states, dtype=np.float32)
    valid = np.asarray(state_valid, dtype=bool)
    references, reference_valid, fallback = patient_references(
        x, valid, shuffled_reference=reference_permutation)
    current_j = x[:, :, None, :]
    current_k = x[:, None, :, :]
    self_delta = current_j - current_k
    patient_delta = current_j - references[:, None, :, :]
    self_mask = valid[:, :, None] & valid[:, None, :]
    patient_mask = valid[:, :, None] & reference_valid[:, None, :]
    self_delta *= self_mask[..., None]
    patient_delta *= patient_mask[..., None]
    # [C,J,K,F] -> [C,F,J,K], plus the two explicit state-pair masks.
    result = np.concatenate((self_delta.transpose(0, 3, 1, 2),
                             patient_delta.transpose(0, 3, 1, 2),
                             self_mask[:, None].astype(np.float32),
                             patient_mask[:, None].astype(np.float32)), axis=1)
    if result.shape[1:] != (34, 6, 6) or not np.isfinite(result).all():
        raise RuntimeError("invalid SCM tensor")
    return result.astype(np.float32), fallback


class SCMEZ(nn.Module):
    """The frozen one-model SCM-EZ architecture."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(34, 24, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=3, padding=1, groups=24, bias=True),
            nn.GELU(),
            nn.Conv2d(24, 24, kernel_size=1, bias=True),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(144, 32), nn.GELU(), nn.Dropout(0.1), nn.Linear(32, 1)
        )

    def forward(self, matrices: torch.Tensor, channel_index: torch.Tensor,
                n_channels: int) -> torch.Tensor:
        h = self.encoder(matrices)
        embedding = torch.cat((h.mean(dim=(-2, -1)), h.amax(dim=(-2, -1))), dim=1)
        pooled = []
        for channel in range(int(n_channels)):
            current = embedding[channel_index == channel]
            if not len(current):
                raise RuntimeError("canonical channel has no synchronized record")
            pooled.append(torch.cat((current.mean(0), current.std(0, unbiased=False), current.amax(0))))
        return self.classifier(torch.stack(pooled)).squeeze(-1)


def parameter_count() -> int:
    return sum(parameter.numel() for parameter in SCMEZ().parameters() if parameter.requires_grad)
