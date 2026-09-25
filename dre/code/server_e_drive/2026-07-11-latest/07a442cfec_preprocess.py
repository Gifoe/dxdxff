from __future__ import annotations

import math
from typing import Tuple

import numpy as np


def robust_normalize_1d(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr.astype(np.float32)
    centered = arr - np.median(arr)
    q75, q25 = np.percentile(centered, [75, 25])
    scale = float(q75 - q25)
    if abs(scale) < 1e-6:
        scale = float(np.std(centered))
    if abs(scale) < 1e-6:
        return np.zeros_like(centered, dtype=np.float32)
    return np.clip(centered / scale, -8.0, 8.0).astype(np.float32)


def resample_1d_if_needed(x: np.ndarray, sfreq: float, target_sfreq: float) -> Tuple[np.ndarray, str]:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    if arr.size == 0 or float(sfreq) <= 0 or float(target_sfreq) <= 0:
        return arr.astype(np.float32), "none"
    if abs(float(sfreq) - float(target_sfreq)) < 1e-6:
        return arr.astype(np.float32), "none"
    try:
        from scipy.signal import resample_poly

        gcd = math.gcd(int(round(target_sfreq)), int(round(sfreq)))
        up = int(round(target_sfreq)) // max(gcd, 1)
        down = int(round(sfreq)) // max(gcd, 1)
        return resample_poly(arr, up, down).astype(np.float32), "scipy.signal.resample_poly"
    except Exception:
        old_idx = np.arange(arr.size, dtype=np.float32)
        new_len = max(1, int(round(arr.size * float(target_sfreq) / float(sfreq))))
        new_idx = np.linspace(0, max(arr.size - 1, 0), new_len, dtype=np.float32)
        return np.interp(new_idx, old_idx, arr).astype(np.float32), "numpy.interp"


def ensure_fixed_length(x: np.ndarray, target_len: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float32).reshape(-1)
    target_len = int(target_len)
    if target_len <= 0:
        return np.zeros(0, dtype=np.float32)
    if arr.size == target_len:
        return arr.astype(np.float32)
    if arr.size > target_len:
        start = int((arr.size - target_len) // 2)
        return arr[start : start + target_len].astype(np.float32)
    out = np.zeros(target_len, dtype=np.float32)
    out[: arr.size] = arr
    return out

