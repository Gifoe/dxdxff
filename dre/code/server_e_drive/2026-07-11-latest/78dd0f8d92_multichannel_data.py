from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from outcome_hifos.cache_schema import CachePayload
from task1_baselines.cache_io import task1_record_labels
from task1_baselines.raw_preprocessing import RawPreprocessingConfig, extract_and_preprocess_windows


@dataclass(frozen=True)
class Task1MultichannelWindow:
    waveform: np.ndarray
    label_nez: np.ndarray
    channel_names: tuple[str, ...]
    subject_id: str
    center: str
    seizure_id: str
    window_id: int
    modality: str
    label_source: str


@dataclass(frozen=True)
class Task1MultichannelBatch:
    """Padded batch. Metadata is audit-only and never supplied to the model."""

    waveform: torch.Tensor
    label_nez: torch.Tensor
    channel_mask: torch.Tensor
    channel_names: tuple[tuple[str, ...], ...]
    subject_ids: tuple[str, ...]
    centers: tuple[str, ...]
    seizure_ids: tuple[str, ...]
    window_ids: tuple[int, ...]
    modalities: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


class Task1MultichannelDataset(Dataset[Task1MultichannelWindow]):
    def __init__(self, items: Sequence[Task1MultichannelWindow], audits: Sequence[dict[str, Any]]) -> None:
        self.items = tuple(items)
        self.audits = tuple(audits)
        self.rows = pd.DataFrame([{
            "subject_id": item.subject_id, "center": item.center, "seizure_id": item.seizure_id,
            "window_id": item.window_id, "modality": item.modality, "n_channels": item.waveform.shape[0],
        } for item in self.items])

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> Task1MultichannelWindow:
        return self.items[index]


def _center(subject: str, meta: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    for source in (meta, record):
        for key in ("source_center", "center", "source_dataset"):
            value = str(source.get(key, "")).strip().lower()
            if value:
                return value
    return subject.split(":", 1)[0].lower() if ":" in subject else "unknown"


def _modality(record: Mapping[str, Any], sample: Mapping[str, Any]) -> str:
    return str(sample.get("modality", record.get("modality", "unknown"))).strip().lower() or "unknown"


def build_task1_multichannel_windows(cache: CachePayload, subject_ids: set[str], config: RawPreprocessingConfig) -> Task1MultichannelDataset:
    requested = {str(subject) for subject in subject_ids}
    missing_meta = requested - set(cache.patient_index)
    if missing_meta:
        raise ValueError(f"Task 1 raw patients missing from patient_index: {sorted(missing_meta)[:20]}")
    items: list[Task1MultichannelWindow] = []
    audits: list[dict[str, Any]] = []
    observed: set[str] = set()
    for record in cache.run_records:
        subject = str(record.get("subject_id", ""))
        if subject not in requested:
            continue
        meta = cache.patient_index[subject]
        sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
        channels, labels_nez, label_source = task1_record_labels(record, meta)
        if len(channels) != len(set(channels)):
            raise ValueError(f"Raw run for {subject} has duplicate channel names.")
        sfreq = float(sample.get("raw_temporal_sfreq", record.get("sfreq", 0.0)))
        windows, audit = extract_and_preprocess_windows(dict(sample), original_sfreq=sfreq, config=config)
        if windows.ndim != 3 or windows.shape[1] != len(channels) or labels_nez.shape != (len(channels),):
            raise ValueError(f"Raw channel axis/labels do not align for {subject} run {record.get('run_id')!r}.")
        seizure_id = str(record.get("run_id", sample.get("sample_id", "")))
        for window_id, waveform in enumerate(windows):
            if waveform.shape != (len(channels), int(round(config.target_sfreq * config.window_sec))) or not np.isfinite(waveform).all():
                raise ValueError(f"Invalid preprocessed multichannel window for {subject}.")
            items.append(Task1MultichannelWindow(
                waveform=waveform.astype(np.float32, copy=False), label_nez=labels_nez.astype(np.float32, copy=False),
                channel_names=tuple(channels), subject_id=subject, center=_center(subject, meta, record),
                seizure_id=seizure_id, window_id=int(window_id), modality=_modality(record, sample), label_source=label_source,
            ))
        audits.append({"subject_id": subject, "seizure_id": seizure_id, **audit})
        observed.add(subject)
    missing = requested - observed
    if missing:
        raise ValueError(f"Raw cache has no usable runs for manifest patients: {sorted(missing)[:20]}")
    if not items:
        raise ValueError("No multichannel raw windows were produced.")
    return Task1MultichannelDataset(items, audits)


def collate_multichannel_windows(batch: Sequence[Task1MultichannelWindow]) -> dict[str, Any]:
    if not batch:
        raise ValueError("Cannot collate an empty batch.")
    n_times = batch[0].waveform.shape[1]
    maximum = max(item.waveform.shape[0] for item in batch)
    waveform = torch.zeros((len(batch), maximum, n_times), dtype=torch.float32)
    labels = torch.zeros((len(batch), maximum), dtype=torch.float32)
    mask = torch.zeros((len(batch), maximum), dtype=torch.bool)
    for index, item in enumerate(batch):
        count = item.waveform.shape[0]
        waveform[index, :count] = torch.from_numpy(item.waveform)
        labels[index, :count] = torch.from_numpy(item.label_nez)
        mask[index, :count] = True
    return Task1MultichannelBatch(
        waveform=waveform, label_nez=labels, channel_mask=mask,
        channel_names=tuple(item.channel_names for item in batch),
        subject_ids=tuple(item.subject_id for item in batch), centers=tuple(item.center for item in batch),
        seizure_ids=tuple(item.seizure_id for item in batch), window_ids=tuple(item.window_id for item in batch),
        modalities=tuple(item.modality for item in batch),
    ).as_dict()
