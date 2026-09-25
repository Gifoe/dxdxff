from __future__ import annotations

import numpy as np


def robust_baseline_zscore(
    features: np.ndarray,
    baseline_mask: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    mask = np.asarray(baseline_mask, dtype=bool)
    if x.ndim != 3:
        raise ValueError("features must have shape [T, C, F].")
    if mask.shape[0] != x.shape[0] or not np.any(mask):
        raise ValueError("baseline_mask must match T and contain at least one baseline window.")

    baseline = x[mask]
    median = np.nanmedian(baseline, axis=0, keepdims=True)
    q75 = np.nanpercentile(baseline, 75, axis=0, keepdims=True)
    q25 = np.nanpercentile(baseline, 25, axis=0, keepdims=True)
    iqr = q75 - q25
    z = (x - median) / (iqr + float(eps))
    return np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)


def baseline_correct_adjacency(
    adjacency_seq: np.ndarray,
    baseline_mask: np.ndarray,
    mode: str = "robust_z",
    eps: float = 1e-5,
) -> np.ndarray:
    adjacency = np.asarray(adjacency_seq, dtype=np.float32)
    mask = np.asarray(baseline_mask, dtype=bool)
    if adjacency.ndim != 3:
        raise ValueError("adjacency_seq must have shape [T, C, C].")
    if mask.shape[0] != adjacency.shape[0] or not np.any(mask):
        raise ValueError("baseline_mask must match T and contain at least one baseline window.")

    baseline = adjacency[mask]
    if mode == "delta":
        corrected = adjacency - np.nanmean(baseline, axis=0, keepdims=True)
    elif mode == "robust_z":
        median = np.nanmedian(baseline, axis=0, keepdims=True)
        q75 = np.nanpercentile(baseline, 75, axis=0, keepdims=True)
        q25 = np.nanpercentile(baseline, 25, axis=0, keepdims=True)
        corrected = (adjacency - median) / (q75 - q25 + float(eps))
    else:
        raise ValueError(f"Unsupported adjacency correction mode: {mode}")

    corrected = np.nan_to_num(corrected, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)
    for idx in range(corrected.shape[0]):
        np.fill_diagonal(corrected[idx], 0.0)
    return corrected


__all__ = ["baseline_correct_adjacency", "robust_baseline_zscore"]

