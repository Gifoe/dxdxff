from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from outcome_hifos.cache_schema import CachePayload
from .cache_io import task1_record_labels
from .raw_preprocessing import RawPreprocessingConfig, extract_and_preprocess_windows


@dataclass(frozen=True)
class Task1TokenDataset:
    values: np.ndarray
    rows: pd.DataFrame
    preprocessing_audit: tuple[dict[str, Any], ...]


def _center(subject: str, meta: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    for source in (meta, record):
        for key in ("source_center", "center", "source_dataset"):
            value = str(source.get(key, "")).strip().lower()
            if value:
                return value
    return subject.split(":", 1)[0].lower() if ":" in subject else "unknown"


def build_task1_raw_tokens(cache: CachePayload, subject_ids: set[str], config: RawPreprocessingConfig) -> Task1TokenDataset:
    requested = {str(value) for value in subject_ids}
    values: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    audits: list[dict[str, Any]] = []
    observed: set[str] = set()
    for record in cache.run_records:
        subject = str(record.get("subject_id", ""))
        if subject not in requested:
            continue
        meta = cache.patient_index.get(subject)
        if meta is None:
            raise ValueError(f"Task 1 raw patient {subject} missing from patient_index.")
        sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
        channels, labels_nez, label_source = task1_record_labels(record, meta)
        sfreq = float(sample.get("raw_temporal_sfreq", record.get("sfreq", 0.0)))
        windows, audit = extract_and_preprocess_windows(dict(sample), original_sfreq=sfreq, config=config)
        if windows.shape[1] != len(channels):
            raise ValueError(f"Task 1 raw run {record.get('run_id')} channel axis does not match channel names.")
        seizure_id = str(record.get("run_id", sample.get("sample_id", "")))
        for window_index in range(windows.shape[0]):
            for channel_index, channel in enumerate(channels):
                values.append(windows[window_index, channel_index])
                rows.append(
                    {
                        "subject_id": subject, "center": _center(subject, meta, record),
                        "seizure_id": seizure_id, "channel_name": channel, "window_id": int(window_index),
                        "label_nez": int(labels_nez[channel_index]), "label_source": label_source,
                    }
                )
        audits.append({"subject_id": subject, "seizure_id": seizure_id, **audit})
        observed.add(subject)
    missing = sorted(requested - observed)
    if missing:
        raise ValueError(f"Task 1 raw cache missing old-90 patients: {missing[:20]}")
    if not values:
        raise ValueError("Task 1 raw cache yielded no channel-window tokens.")
    return Task1TokenDataset(np.stack(values).astype(np.float32), pd.DataFrame(rows), tuple(audits))


def build_task1_embedding_tokens(cache: CachePayload, subject_ids: set[str]) -> Task1TokenDataset:
    requested = {str(value) for value in subject_ids}
    values: list[np.ndarray] = []
    rows: list[dict[str, Any]] = []
    observed: set[str] = set()
    for record in cache.run_records:
        subject = str(record.get("subject_id", ""))
        if subject not in requested:
            continue
        meta = cache.patient_index.get(subject)
        if meta is None:
            raise ValueError(f"Task 1 FM patient {subject} missing from patient_index.")
        sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
        channels, labels_nez, label_source = task1_record_labels(record, meta)
        embeddings = np.asarray(sample.get("window_features"), dtype=np.float32)
        if embeddings.ndim != 3 or embeddings.shape[1] != len(channels):
            raise ValueError("Task 1 FM embeddings must be [window,channel,dimension] with aligned channels.")
        seizure_id = str(record.get("run_id", sample.get("sample_id", "")))
        for window_index in range(embeddings.shape[0]):
            for channel_index, channel in enumerate(channels):
                values.append(embeddings[window_index, channel_index])
                rows.append({"subject_id": subject, "center": _center(subject, meta, record), "seizure_id": seizure_id, "channel_name": channel, "window_id": window_index, "label_nez": int(labels_nez[channel_index]), "label_source": label_source})
        observed.add(subject)
    missing = sorted(requested - observed)
    if missing:
        raise ValueError(f"Task 1 FM cache missing old-90 patients: {missing[:20]}")
    if not values:
        raise ValueError("Task 1 FM cache yielded no embeddings.")
    return Task1TokenDataset(np.stack(values).astype(np.float32), pd.DataFrame(rows), ())


__all__ = ["Task1TokenDataset", "build_task1_embedding_tokens", "build_task1_raw_tokens"]
