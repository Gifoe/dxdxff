from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class BootstrapInterval:
    estimate: float
    lower: float
    upper: float
    samples: int


def paired_patient_bootstrap(
    target: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    samples: int = 2000,
    seed: int = 42,
) -> BootstrapInterval:
    y = np.asarray(target)
    a = np.asarray(first)
    b = np.asarray(second)
    if not (y.shape == a.shape == b.shape) or y.ndim != 1:
        raise ValueError("Paired bootstrap inputs must be aligned one-dimensional arrays.")
    rng = np.random.default_rng(int(seed))
    differences = []
    for _ in range(int(samples)):
        indices = rng.integers(0, len(y), size=len(y))
        differences.append(float(metric(y[indices], a[indices]) - metric(y[indices], b[indices])))
    estimate = float(metric(y, a) - metric(y, b))
    lower, upper = np.percentile(np.asarray(differences), [2.5, 97.5]).tolist()
    return BootstrapInterval(estimate, float(lower), float(upper), int(samples))


__all__ = ["BootstrapInterval", "paired_patient_bootstrap"]
