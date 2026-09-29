"""Train-patient-only 36-D descriptor moments; no center/test fitting."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class DescriptorMoments:
    def __init__(self):
        self.count = np.zeros(36, dtype=np.float64)
        self.total = np.zeros(36, dtype=np.float64)
        self.square = np.zeros(36, dtype=np.float64)

    def update(self, values: np.ndarray, available: np.ndarray):
        x = np.asarray(values, dtype=np.float64)
        m = np.asarray(available, dtype=bool)
        if x.shape != m.shape or x.shape[-1] != 36:
            raise ValueError("Descriptor values/mask must end in 36 matching dimensions")
        if not np.isfinite(x[m]).all():
            raise RuntimeError("Nonfinite observed descriptor")
        flat, mask = x.reshape(-1, 36), m.reshape(-1, 36)
        self.count += mask.sum(axis=0)
        self.total += np.where(mask, flat, 0).sum(axis=0)
        self.square += np.where(mask, flat * flat, 0).sum(axis=0)

    def freeze(self, *, benchmark: str, fold: int, fit_patient_count: int,
               protocol_sha256: str) -> dict:
        if not (self.count > 0).any():
            raise RuntimeError("No observed train descriptors")
        safe_count = np.maximum(self.count, 1)
        mean = self.total / safe_count
        variance = np.maximum(self.square / safe_count - mean * mean, 0)
        scale = np.maximum(np.sqrt(variance), 1e-5)
        # A physically unavailable band has zero fit observations. Its mask is
        # always zero; use an identity normalization for that coordinate.
        scale[self.count == 0] = 1.0
        return {"benchmark": benchmark, "fold": fold,
                "fit_patient_count": fit_patient_count,
                "protocol_sha256": protocol_sha256,
                "mean": mean.tolist(), "std": scale.tolist(),
                "observed_counts": self.count.astype(int).tolist(),
                "validation_or_test_used_for_fit": False,
                "center_specific_normalization": False}


def apply(values: np.ndarray, available: np.ndarray, frozen: dict) -> np.ndarray:
    x = np.asarray(values, dtype=np.float32)
    mask = np.asarray(available, dtype=bool)
    if x.shape != mask.shape or x.shape[-1] != 36:
        raise ValueError("Descriptor/mask mismatch")
    mean = np.asarray(frozen["mean"], dtype=np.float32)
    std = np.asarray(frozen["std"], dtype=np.float32)
    normalized = (x - mean) / std
    normalized[~mask] = 0
    if not np.isfinite(normalized).all():
        raise RuntimeError("Nonfinite normalized descriptor")
    return normalized


def save(path: Path, frozen: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
