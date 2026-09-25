from __future__ import annotations

from dataclasses import dataclass
import numpy as np

PHASES = {"preictal": (-10.0, -1.0, 5.0), "onset": (0.0, 5.0, 2.0),
          "spread": (5.0, 30.0, 3.0)}


@dataclass(frozen=True)
class PhaseInterval:
    name: str
    start_sec: float
    end_sec: float
    actual_duration_sec: float
    valid: bool


def phase_intervals(onset_sample: int, sampling_rate: float, n_samples: int) -> dict[str, PhaseInterval]:
    available_start = -float(onset_sample) / sampling_rate
    available_end = (n_samples - onset_sample) / sampling_rate
    result = {}
    for name, (start, end, minimum) in PHASES.items():
        lo, hi = max(start, available_start), min(end, available_end)
        duration = max(0.0, hi - lo)
        result[name] = PhaseInterval(name, lo, hi, duration, duration + 1e-9 >= minimum)
    return result


def phase_mask(times_sec: np.ndarray, phase: str) -> np.ndarray:
    start, end, _ = PHASES[phase]
    return (np.asarray(times_sec) >= start) & (np.asarray(times_sec) < end)


__all__ = ["PHASES", "PhaseInterval", "phase_intervals", "phase_mask"]
