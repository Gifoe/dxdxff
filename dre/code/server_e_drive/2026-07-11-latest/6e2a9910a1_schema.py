from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import torch

ALLOWED_SEIZURE_FIELDS = {"seizure_id","windows","window_mask","phase_ids","relative_times_sec","channel_names"}

@dataclass(frozen=True)
class ViewConfig:
    max_seizures_train: int = 3
    max_channels_train: int = 96
    max_windows_per_phase_train: int = 6

def validate_seizure(seizure: dict[str,Any]) -> None:
    windows=seizure["windows"];mask=seizure["window_mask"].bool();phase=seizure["phase_ids"];times=seizure["relative_times_sec"]
    if windows.ndim!=3 or windows.shape[-1]!=500:raise ValueError("windows must be [C,T,500]")
    if mask.shape!=windows.shape[:2]:raise ValueError("window_mask shape mismatch")
    if phase.shape not in (windows.shape[:2],(windows.shape[1],)):raise ValueError("phase_ids shape mismatch")
    if times.shape not in (windows.shape[:2],(windows.shape[1],)):raise ValueError("relative_times_sec shape mismatch")
    if len(seizure["channel_names"])!=windows.shape[0]:raise ValueError("channel_names shape mismatch")
    if not torch.isfinite(windows[mask]).all():raise ValueError("non-finite valid raw window")
    if not mask.any(1).any():raise ValueError("seizure has no valid channel")

def validate_patient(patient: dict[str,Any]) -> None:
    if patient.get("outcome_success") not in (0,1):raise ValueError("outcome_success must be 0/1")
    if not patient.get("seizures"):raise ValueError("patient has no seizure")
    for seizure in patient["seizures"]:validate_seizure(seizure)
