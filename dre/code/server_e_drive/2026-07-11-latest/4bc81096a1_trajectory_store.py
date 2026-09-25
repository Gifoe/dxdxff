from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .schemas import normalize_channel_name


@dataclass(frozen=True)
class RunTrajectory:
    """One cache run with explicit invalid/missing observation semantics."""

    run_id: str
    channel_names: tuple[str, ...]
    values: np.ndarray
    window_mask: np.ndarray | None = None
    finite_mask: np.ndarray | None = None

    def __post_init__(self) -> None:
        data = np.asarray(self.values, dtype=float)
        if data.ndim != 3 or data.shape[1] != len(self.channel_names) or min(data.shape) <= 0:
            raise ValueError("RunTrajectory values must be [window, channel, feature] with named channels")
        if self.window_mask is not None and np.asarray(self.window_mask).shape not in {data.shape[:1], data.shape[:2]}:
            raise ValueError("window_mask must be [window] or [window, channel]")
        if self.finite_mask is not None and np.asarray(self.finite_mask).shape != data.shape:
            raise ValueError("finite_mask must match [window, channel, feature]")

    def channel_mask(self, channel_index: int) -> np.ndarray:
        data = np.asarray(self.values, dtype=float)
        finite = np.isfinite(data[:, channel_index, :]).all(axis=1) if self.finite_mask is None else np.asarray(self.finite_mask, dtype=bool)[:, channel_index, :].all(axis=1)
        if self.window_mask is None:
            return finite
        supplied = np.asarray(self.window_mask, dtype=bool)
        return finite & (supplied if supplied.ndim == 1 else supplied[:, channel_index])


class PatientTrajectoryStore:
    """Read-only subject→run→window→channel store; invalid values remain invalid."""

    def __init__(self, *, feature_names: Iterable[str] = ()) -> None:
        self.feature_names = tuple(map(str, feature_names))
        self._runs: dict[str, list[RunTrajectory]] = {}
        self.duplicate_run_ids: list[tuple[str, str]] = []

    def add_run(self, subject_id: str, run: RunTrajectory) -> None:
        subject = str(subject_id)
        existing = self._runs.setdefault(subject, [])
        if any(item.run_id == run.run_id for item in existing):
            self.duplicate_run_ids.append((subject, run.run_id))
            raise ValueError(f"duplicate run_id for {subject}: {run.run_id}")
        if existing and np.asarray(existing[0].values).shape[2] != np.asarray(run.values).shape[2]:
            raise ValueError("all runs for a subject must have the same feature dimension")
        existing.append(run)

    def subjects(self) -> tuple[str, ...]: return tuple(sorted(self._runs))
    def runs(self, subject_id: str) -> tuple[RunTrajectory, ...]: return tuple(self._runs.get(str(subject_id), ()))

    def get_channel_trajectory(self, subject_id: str, channel_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        runs = self.runs(subject_id)
        if not runs: raise KeyError(f"unknown trajectory subject: {subject_id}")
        target = normalize_channel_name(channel_name)
        feature_dim = int(np.asarray(runs[0].values).shape[2]); max_windows = max(len(run.values) for run in runs)
        values = np.full((len(runs), max_windows, feature_dim), np.nan, dtype=float)
        mask = np.zeros((len(runs), max_windows), dtype=bool); seizure_mask = np.zeros(len(runs), dtype=bool)
        for run_index, run in enumerate(runs):
            names = [normalize_channel_name(name) for name in run.channel_names]
            if target not in names: continue
            channel_index = names.index(target); sequence = np.asarray(run.values, dtype=float)[:, channel_index, :]; valid = run.channel_mask(channel_index)
            values[run_index, :len(sequence)] = sequence
            mask[run_index, :len(sequence)] = valid
            seizure_mask[run_index] = bool(valid.any())
        return values, mask, seizure_mask

    def static_channel_features(self) -> dict[str, dict[str, list[float]]]:
        output: dict[str, dict[str, list[float]]] = {}
        for subject, runs in self._runs.items():
            output[subject] = {}
            for channel in sorted({normalize_channel_name(ch) for run in runs for ch in run.channel_names}):
                summaries = []
                for run in runs:
                    names = [normalize_channel_name(name) for name in run.channel_names]
                    if channel in names:
                        index = names.index(channel); valid = run.channel_mask(index)
                        if valid.any(): summaries.append(np.nanmean(np.asarray(run.values, dtype=float)[valid, index, :], axis=0))
                if summaries: output[subject][channel] = np.nanmean(np.vstack(summaries), axis=0).astype(float).tolist()
        return output
