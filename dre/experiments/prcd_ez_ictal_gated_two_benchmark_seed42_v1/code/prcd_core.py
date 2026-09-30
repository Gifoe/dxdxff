"""Fixed, CPU-vectorized PRCD-EZ feature operators.

The module is deliberately model-free until the final small classifier.  All
dictionary functions accept signal features and masks, never labels.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np
from scipy.signal import hilbert

SEED = 42
WINDOWS = 59
INPUT_DIM = 52
GROUPS = 48
KERNELS = 6
LENGTH = 9
VALID_DILATIONS = (1, 2, 4)
SUBSET_SIZES = (1, 2, 4)
BIAS_QUANTILES = (0.10, 0.25, 0.50, 0.75, 0.90)
VIEW_DIM = GROUPS * 14
RECORD_DIM = VIEW_DIM * 2
POOLED_DIM = RECORD_DIM * 4


def stable_seed(*parts: object) -> int:
    payload = "|".join(map(str, parts)).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


@dataclass(frozen=True)
class Kernel:
    group: int
    index: int
    dilation: int
    dimensions: tuple[int, ...]
    weights: np.ndarray
    quantile_index: int


def make_dictionary(seed: int = SEED) -> list[list[Kernel]]:
    result: list[list[Kernel]] = []
    for group in range(GROUPS):
        current = []
        for index in range(KERNELS):
            kernel_id = group * KERNELS + index
            rng = np.random.default_rng(stable_seed(seed, "kernel", kernel_id))
            dilation = int(rng.choice(VALID_DILATIONS))
            size = int(rng.choice(SUBSET_SIZES))
            dimensions = tuple(sorted(map(int, rng.choice(INPUT_DIM, size=size, replace=False))))
            rows = []
            for _ in dimensions:
                pattern = np.asarray([2.0] * 3 + [-1.0] * 6, dtype=np.float64)
                rng.shuffle(pattern)
                rows.append(pattern)
            weights = np.stack(rows)
            weights /= max(float(np.linalg.norm(weights)), 1e-12)
            q = stable_seed(seed, "bias_quantile", kernel_id) % len(BIAS_QUANTILES)
            current.append(Kernel(group, index, dilation, dimensions, weights.astype(np.float32), int(q)))
        result.append(current)
    return result


def fine_spectrum(waveforms: np.ndarray, fs: float, window_mask: np.ndarray,
                  dimensions: int = 16, chunk: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized 2-s/1-s Hann periodogram in physical log-spaced bands."""
    x = np.asarray(waveforms, dtype=np.float32)
    valid = np.asarray(window_mask, dtype=bool)
    n = int(round(2 * fs)); hop = int(round(fs))
    if x.ndim != 2 or valid.shape[0] != x.shape[0] or valid.shape[1] != WINDOWS:
        raise ValueError("fine-spectrum input shape mismatch")
    upper = min(300.0, 0.45 * float(fs))
    required = n + hop * (WINDOWS - 1)
    if x.shape[1] < required:
        x = np.pad(x, ((0, 0), (0, required - x.shape[1])))
    edges = np.geomspace(1.0, upper, dimensions + 1)
    frequencies = np.fft.rfftfreq(n, 1.0 / fs)
    df = float(frequencies[1] - frequencies[0])
    bins = [np.flatnonzero((frequencies >= lo) & (frequencies < hi)) for lo, hi in zip(edges[:-1], edges[1:])]
    taper = np.hanning(n).astype(np.float32)
    energy = max(float(np.square(taper).sum()), 1e-12)
    out = np.zeros((len(x), WINDOWS, dimensions), dtype=np.float32)
    mask = np.zeros_like(out, dtype=bool)
    for begin_channel in range(0, len(x), chunk):
        block = x[begin_channel:begin_channel + chunk]
        views = np.lib.stride_tricks.sliding_window_view(block, n, axis=1)[:, ::hop, :]
        views = views[:, :WINDOWS, :] * taper[None, None, :]
        power = np.square(np.abs(np.fft.rfft(views, axis=-1))) / energy
        for feature, ids in enumerate(bins):
            if len(ids):
                out[begin_channel:begin_channel + len(block), :, feature] = np.log1p(
                    power[..., ids].sum(axis=-1) * df).astype(np.float32)
                mask[begin_channel:begin_channel + len(block), :, feature] = True
    mask &= valid[..., None]
    out[~mask] = 0.0
    if not np.isfinite(out).all():
        raise RuntimeError("nonfinite fine spectrum")
    return out, mask


def first_difference(x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    out[:, 1:] = x[:, 1:] - x[:, :-1]
    return out


def kernel_activation(x: np.ndarray, mask: np.ndarray, kernel: Kernel,
                      centers: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Activation for N sequences, common centers; vectorized over N/time."""
    positions = centers[:, None] + (np.arange(LENGTH) - LENGTH // 2)[None, :] * kernel.dilation
    activation = np.zeros((len(x), len(centers)), dtype=np.float32)
    used = np.zeros_like(activation, dtype=np.int16)
    for row, dimension in enumerate(kernel.dimensions):
        observed = mask[:, positions, dimension].all(axis=-1)
        values = x[:, positions, dimension]
        contribution = np.einsum("ntj,j->nt", values, kernel.weights[row], optimize=True)
        activation += np.where(observed, contribution, 0.0)
        used += observed
    valid = used > 0
    activation[valid] /= np.sqrt(used[valid].astype(np.float32))
    activation[~valid] = np.nan
    return activation, valid


def fit_biases(sample_x: np.ndarray, sample_mask: np.ndarray,
               dictionary: list[list[Kernel]]) -> np.ndarray:
    biases = np.zeros((2, GROUPS, KERNELS), dtype=np.float32)
    for view, current in enumerate((sample_x, first_difference(sample_x))):
        for group, kernels in enumerate(dictionary):
            radius = (LENGTH // 2) * max(kernel.dilation for kernel in kernels)
            centers = np.arange(radius, current.shape[1] - radius)
            for index, kernel in enumerate(kernels):
                activation, valid = kernel_activation(current, sample_mask, kernel, centers)
                values = activation[valid]
                if not len(values):
                    raise RuntimeError("FIT bias sample has no valid activation")
                biases[view, group, index] = float(np.quantile(values, BIAS_QUANTILES[kernel.quantile_index]))
    return biases


def _longest_run_ratio(values: np.ndarray) -> np.ndarray:
    result = np.zeros(len(values), dtype=np.float32)
    for row, bits in enumerate(values):
        padded = np.pad(bits.astype(np.int8), (1, 1))
        changes = np.flatnonzero(np.diff(padded))
        result[row] = float(np.max(changes[1::2] - changes[::2], initial=0) / max(len(bits), 1))
    return result


def dictionary_view(x: np.ndarray, mask: np.ndarray, dictionary: list[list[Kernel]],
                    biases: np.ndarray) -> np.ndarray:
    """Return [sequence,672] competitive group statistics."""
    rows = []
    for group, kernels in enumerate(dictionary):
        radius = (LENGTH // 2) * max(kernel.dilation for kernel in kernels)
        centers = np.arange(radius, x.shape[1] - radius)
        activation = []
        for kernel in kernels:
            value, valid = kernel_activation(x, mask, kernel, centers)
            activation.append(np.where(valid, value, -np.inf))
        adjusted = np.stack(activation, axis=-1) - biases[group][None, None, :]
        # A competitive winner and top1-top2 margin are defined only when all
        # six kernels see valid source windows.  This matters for shortened
        # records whose frozen 59-position mask has an invalid tail.
        finite = np.isfinite(adjusted).all(axis=-1)
        winner = np.argmax(adjusted, axis=-1)
        top = np.max(adjusted, axis=-1)
        top2 = np.partition(adjusted, -2, axis=-1)[..., -2]
        denom = np.maximum(finite.sum(axis=1), 1)
        occupancy = np.stack([((winner == k) & finite).sum(axis=1) / denom for k in range(KERNELS)], axis=1)
        positive = np.stack([((winner == k) & (top > 0) & finite).sum(axis=1) / denom for k in range(KERNELS)], axis=1)
        margin_value = np.zeros_like(top, dtype=np.float32)
        np.subtract(top, top2, out=margin_value, where=finite)
        margin = margin_value.sum(axis=1) / denom
        longest = _longest_run_ratio((top > 0) & finite)
        rows.append(np.concatenate((occupancy, positive, margin[:, None], longest[:, None]), axis=1))
    out = np.concatenate(rows, axis=1).astype(np.float32)
    if out.shape != (len(x), VIEW_DIM) or not np.isfinite(out).all():
        raise RuntimeError("invalid competitive dictionary output")
    return out


def dictionary_record(x: np.ndarray, mask: np.ndarray, dictionary: list[list[Kernel]],
                      biases: np.ndarray) -> np.ndarray:
    return np.concatenate((dictionary_view(x, mask, dictionary, biases[0]),
                           dictionary_view(first_difference(x), mask, dictionary, biases[1])), axis=1)


def pool_records(records: np.ndarray, presence: np.ndarray) -> np.ndarray:
    """Pool [record,channel,1344] into [channel,5376]."""
    channels = records.shape[1]
    pooled = np.zeros((channels, POOLED_DIM), dtype=np.float32)
    for channel in range(channels):
        values = records[presence[:, channel], channel]
        if len(values):
            pooled[channel] = np.concatenate((values.mean(0), np.median(values, axis=0),
                                              np.quantile(values, .75, axis=0), values.max(0)))
    return pooled


def average_tie_rank(values: np.ndarray) -> np.ndarray:
    """Columnwise average-tie percentile mapped to [-1,1]."""
    x = np.asarray(values)
    if x.ndim != 2:
        raise ValueError("rank values must be 2D")
    n = len(x)
    if n == 1:
        return np.zeros_like(x, dtype=np.float32)
    order = np.argsort(x, axis=0, kind="stable")
    ranks = np.empty_like(x, dtype=np.float32)
    for column in range(x.shape[1]):
        sorted_values = x[order[:, column], column]
        starts = np.r_[0, 1 + np.flatnonzero(np.diff(sorted_values))]
        ends = np.r_[starts[1:], n]
        for start, end in zip(starts, ends):
            ranks[order[start:end, column], column] = (start + end - 1) / 2
    return (2.0 * ranks / (n - 1) - 1.0).astype(np.float32)


def patient_coordinates(absolute: np.ndarray, valid_channels: np.ndarray) -> np.ndarray:
    result = np.zeros((len(absolute), POOLED_DIM * 3), dtype=np.float32)
    current = absolute[valid_channels]
    if not len(current):
        raise RuntimeError("patient has no signal-valid channel")
    deviation = current - np.median(current, axis=0)
    rank = average_tie_rank(current)
    result[valid_channels] = np.concatenate((current, deviation, rank), axis=1)
    return result


def fixed_band_features(waveforms: np.ndarray, fs: float, present: np.ndarray,
                        window_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """12 burden, 8 PAC/availability, 10 AEC features for one record."""
    x = np.asarray(waveforms, dtype=np.float32)
    present = np.asarray(present, dtype=bool)
    bands = ((1, 4), (4, 8), (8, 13), (13, 30), (30, 55), (65, 100))
    n = int(round(2 * fs)); hop = int(round(fs)); taper = np.hanning(n).astype(np.float32)
    required = n + hop * (WINDOWS - 1)
    if x.shape[1] < required:
        x = np.pad(x, ((0, 0), (0, required - x.shape[1])))
    energy = max(float(np.square(taper).sum()), 1e-12)
    freq = np.fft.rfftfreq(n, 1 / fs); df = freq[1] - freq[0]
    views = np.lib.stride_tricks.sliding_window_view(x, n, axis=1)[:, ::hop, :][:, :WINDOWS] * taper
    power = np.square(np.abs(np.fft.rfft(views, axis=-1))) / energy
    burden = np.zeros((len(x), 12), dtype=np.float32)
    for band, (lo, hi) in enumerate(bands):
        ids = np.flatnonzero((freq >= lo) & (freq < hi) & (freq <= .45 * fs))
        if len(ids):
            value = np.log1p(power[..., ids].sum(-1) * df)
            for channel in np.flatnonzero(present):
                use = window_mask[channel]
                if use.any():
                    burden[channel, 2 * band:2 * band + 2] = (value[channel, use].mean(), np.quantile(value[channel, use], .75))
    # Fixed FFT bandpass + analytic signal.  This is label-free and shared by PAC/AEC.
    full_freq = np.fft.rfftfreq(x.shape[1], 1 / fs)
    spectrum = np.fft.rfft(x, axis=-1)
    analytic = {}
    for name, (lo, hi) in zip(("delta", "theta", "alpha", "beta", "gamma", "high"), bands):
        available = hi <= .45 * fs
        if available:
            keep = (full_freq >= lo) & (full_freq < hi)
            filtered = np.fft.irfft(spectrum * keep[None, :], n=x.shape[1], axis=-1)
            analytic[name] = hilbert(filtered, axis=-1)
        else:
            analytic[name] = None
    pac = np.zeros((len(x), 8), dtype=np.float32)
    combos = (("theta", "gamma"), ("theta", "high"), ("alpha", "gamma"), ("alpha", "high"))
    for index, (phase_band, amplitude_band) in enumerate(combos):
        ok = analytic[phase_band] is not None and analytic[amplitude_band] is not None
        pac[:, 4 + index] = float(ok)
        if ok:
            phase = np.angle(analytic[phase_band]); envelope = np.abs(analytic[amplitude_band])
            pac[:, index] = (np.abs(np.mean(envelope * np.exp(1j * phase), axis=-1)) /
                             np.maximum(envelope.mean(axis=-1), 1e-8)).astype(np.float32)
    aec = np.zeros((len(x), 10), dtype=np.float32)
    active = np.flatnonzero(present)
    stride = max(1, int(round(fs / 25.0)))
    for index, name in enumerate(("theta", "alpha", "beta", "gamma", "high")):
        if analytic[name] is None or len(active) < 2:
            continue
        envelope = np.abs(analytic[name][active, ::stride])
        correlation = np.nan_to_num(np.corrcoef(envelope), nan=0.0)
        strength = (np.abs(correlation).sum(axis=1) - 1.0) / max(len(active) - 1, 1)
        aec[active, index] = strength.astype(np.float32)
        aec[active, 5 + index] = average_tie_rank(strength[:, None])[:, 0]
    return burden, pac, aec


def dictionary_feature_name(index: int, coordinate: str = "A") -> dict[str, object]:
    pool, rem = divmod(index % POOLED_DIM, RECORD_DIM)
    view, rem = divmod(rem, VIEW_DIM)
    group, stat = divmod(rem, 14)
    if stat < 6: stat_name = f"winner_k{stat}"
    elif stat < 12: stat_name = f"positive_winner_k{stat - 6}"
    elif stat == 12: stat_name = "margin"
    else: stat_name = "longest_positive_run"
    return {"coordinate": coordinate, "pool": ("mean", "median", "q75", "max")[pool],
            "view": ("original", "difference")[view], "group": group, "statistic": stat_name}
