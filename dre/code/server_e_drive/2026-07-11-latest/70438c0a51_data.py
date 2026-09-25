"""Task-2-only cache access and deterministic views.

This module deliberately never reads the value of ``channel_labels_nez``.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
from pathlib import Path
import pandas as pd
import torch

@dataclass(frozen=True)
class ViewConfig:
    n_views: int = 4
    max_seizures: int = 3
    max_channels: int = 64
    max_windows_per_phase: int = 6

def _hash(*parts: object) -> int:
    return int.from_bytes(hashlib.sha256("|".join(map(str, parts)).encode()).digest()[:8], "big")

def _pick(items: list, limit: int, offset: int) -> list:
    if len(items) <= limit:
        return list(items)
    return (items * 2)[offset % len(items):(offset % len(items)) + limit]

def validate_patient(patient: dict) -> None:
    if patient.get("outcome_success") not in (0, 1): raise ValueError("outcome_success must be 0/1")
    seizures = patient.get("seizures", [])
    if not seizures: raise ValueError("patient has no seizures")
    for seizure in seizures:
        w, side, mask = seizure["windows"], seizure["side_features"], seizure["window_mask"].bool()
        if w.ndim != 3 or w.shape[-1] != 500: raise ValueError("windows must be [C,T,500]")
        if side.shape != (*w.shape[:2], 12) or mask.shape != w.shape[:2]: raise ValueError("inconsistent seizure shapes")
        if not torch.isfinite(w).all() or not torch.isfinite(side).all(): raise ValueError("non-finite cache input")
        if len(seizure["phase_ids"]) != w.shape[1] or len(seizure["relative_times_sec"]) != w.shape[1]: raise ValueError("time dimensions mismatch")
        valid_by_time = mask.any(0)
        if not mask.any(1).any() or valid_by_time.sum() < 2: raise ValueError("insufficient valid data")
        for phase in range(3):
            if not (valid_by_time & (seizure["phase_ids"] == phase)).any(): raise ValueError(f"missing valid phase {phase}")

class Task2NativeViewBuilder:
    def __init__(self, seed: int, cfg: ViewConfig): self.seed, self.cfg = int(seed), cfg
    def view(self, patient: dict, view_id: int, include_outcome: bool = False) -> dict:
        # Do not inspect any prohibited cache fields; only native inputs below are read.
        ordered = sorted(patient["seizures"], key=lambda s: _hash(patient["patient_key"], self.seed, view_id, s["seizure_id"]))
        result = []
        for seizure in _pick(ordered, self.cfg.max_seizures, view_id * self.cfg.max_seizures):
            mask = seizure["window_mask"].bool()
            valid_channels = torch.where(mask.any(1))[0].tolist()
            channels = sorted(valid_channels, key=lambda i: _hash(patient["patient_key"], seizure["seizure_id"], self.seed, view_id, seizure["channel_names"][i]))
            channels = _pick(channels, self.cfg.max_channels, view_id * self.cfg.max_channels)
            times = []
            for phase in range(3):
                candidates = torch.where(mask[channels].any(0) & (seizure["phase_ids"] == phase))[0].tolist()
                times += _pick(candidates, self.cfg.max_windows_per_phase, _hash(patient["patient_key"], seizure["seizure_id"], phase, self.seed, view_id))
            c, t = torch.tensor(channels), torch.tensor(sorted(times))
            result.append({"seizure_id": seizure["seizure_id"], "windows": seizure["windows"][c][:, t].float(), "side_features": seizure["side_features"][c][:, t].float(), "window_mask": mask[c][:, t], "phase_ids": seizure["phase_ids"][t], "relative_times_sec": seizure["relative_times_sec"][t], "channel_names": [seizure["channel_names"][i] for i in channels]})
        view = {"patient_key": patient["patient_key"], "center": patient["center"], "view_id": view_id, "seizures": result}
        if include_outcome: view["outcome_success"] = int(patient["outcome_success"])
        return view

class Cohort:
    def __init__(self, root: str | Path, seed: int, cfg: ViewConfig):
        manifest = pd.read_csv(Path(root) / "cache_manifest.csv")
        if manifest.patient_key.duplicated().any(): raise ValueError("duplicate cache patient_key")
        self.paths = {r.patient_key: Path(root) / r.shard_path for r in manifest.itertuples()}
        self.builder = Task2NativeViewBuilder(seed, cfg)
    def load(self, key: str) -> dict: return torch.load(self.paths[key], map_location="cpu", weights_only=False)
    def view(self, key: str, view: int, outcome: bool = False) -> dict: return self.builder.view(self.load(key), view, outcome)
