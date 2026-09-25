from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DeviceResolution:
    cuda_available: bool
    requested_device: str
    resolved_device: str
    fallback_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def resolve_device(requested: str, *, strict: bool = False) -> DeviceResolution:
    import torch

    value = str(requested).strip().lower()
    if value not in {"cpu", "cuda"} and not value.startswith("cuda:"):
        raise ValueError(f"unsupported device: {requested}")
    available = bool(torch.cuda.is_available())
    if value.startswith("cuda") and not available:
        if strict:
            raise RuntimeError(f"requested device {value} is unavailable and strict_device is enabled")
        return DeviceResolution(available, value, "cpu", "requested CUDA device is unavailable")
    return DeviceResolution(available, value, value, None)


def grouped_train_validation_split(frame: pd.DataFrame, *, seed: int, validation_fraction: float = .20) -> tuple[np.ndarray, np.ndarray]:
    if "subject_id" not in frame:
        raise ValueError("grouped split requires subject_id")
    subjects = np.asarray(sorted(frame["subject_id"].astype(str).unique()))
    if len(subjects) < 2:
        return np.arange(len(frame), dtype=int), np.asarray([], dtype=int)
    rng = np.random.default_rng(seed)
    shuffled = subjects.copy(); rng.shuffle(shuffled)
    n_validation = min(len(subjects) - 1, max(1, int(np.ceil(len(subjects) * validation_fraction))))
    validation_subjects = set(shuffled[:n_validation])
    validation = np.flatnonzero(frame["subject_id"].astype(str).isin(validation_subjects).to_numpy())
    train = np.flatnonzero(~frame["subject_id"].astype(str).isin(validation_subjects).to_numpy())
    return train, validation


class PatientBalancedSampler:
    """Replacement sampler with equal total probability per patient."""

    def __init__(self, subject_ids, *, num_samples: int | None = None, seed: int = 0) -> None:
        values = pd.Series(subject_ids).astype(str).reset_index(drop=True)
        counts = values.value_counts()
        weights = values.map(lambda value: 1. / counts[value]).to_numpy(dtype=float)
        self.probabilities = weights / weights.sum()
        self.num_samples = int(num_samples or len(values))
        self.seed = int(seed)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        return iter(rng.choice(len(self.probabilities), size=self.num_samples, replace=True, p=self.probabilities).tolist())

    def __len__(self) -> int:
        return self.num_samples
