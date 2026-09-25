from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from outcome_hifos.cache_schema import CachePayload


def _normalize(value: Any) -> str:
    return "".join(str(value).strip().upper().replace("-", "").replace("_", "").split())


def _center(subject_id: str, meta: Mapping[str, Any], record: Mapping[str, Any]) -> str:
    for source in (meta, record):
        for key in ("source_center", "center", "source_dataset"):
            value = str(source.get(key, "")).strip().lower()
            if value:
                return value
    return subject_id.split(":", 1)[0].lower() if ":" in subject_id else "unknown"


def _vector(value: Any) -> list[Any] | None:
    if isinstance(value, (str, bytes, Mapping)) or value is None:
        return None
    try:
        return list(value)
    except TypeError:
        return None


def task1_record_labels(record: Mapping[str, Any], meta: Mapping[str, Any]) -> tuple[list[str], np.ndarray, str]:
    sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
    local = record.get("channel_names_norm", sample.get("channel_names_norm", []))
    channels = [str(value) for value in local]
    canonical = _vector(meta.get("canonical_channels"))
    patient_labels = _vector(meta.get("labels"))
    record_labels = _vector(record.get("labels"))
    if canonical is not None and patient_labels is not None:
        if len(canonical) != len(patient_labels):
            raise ValueError("Task 1 canonical channel/label lengths differ.")
        label_lookup = {_normalize(channel): int(float(label)) for channel, label in zip(canonical, patient_labels)}
        missing_channels = [channel for channel in channels if _normalize(channel) not in label_lookup]
        if missing_channels:
            raise ValueError(f"Task 1 channels missing from canonical labels: {missing_channels[:10]}")
        labels = np.asarray([label_lookup[_normalize(channel)] for channel in channels], dtype=np.int64)
        label_source = "patient_index.canonical_channels+labels"
    elif record_labels is not None and len(record_labels) == len(channels):
        labels = np.asarray(record_labels, dtype=np.int64)
        label_source = "run_record.channel_names_norm+labels"
    else:
        raise ValueError("Task 1 record lacks explicit channel-aligned EZ labels.")
    if not set(labels.tolist()) <= {-1, 0, 1}:
        raise ValueError("Task 1 record has non-binary channel labels.")
    if np.any(labels < 0):
        raise ValueError("Task 1 record contains unknown channel labels in strict mode.")
    return channels, 1 - labels, label_source


def task1_feature_records(cache: CachePayload, subject_ids: set[str]) -> list[dict[str, Any]]:
    requested = {str(value) for value in subject_ids}
    missing_meta = sorted(requested - set(cache.patient_index))
    if missing_meta:
        raise ValueError(f"Task 1 old-90 patients missing from patient_index: {missing_meta[:20]}")
    output: list[dict[str, Any]] = []
    observed: set[str] = set()
    for record in cache.run_records:
        subject = str(record.get("subject_id", ""))
        if subject not in requested:
            continue
        meta = cache.patient_index[subject]
        sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
        channels, labels_nez, label_source = task1_record_labels(record, meta)
        features = np.asarray(sample.get("window_features"), dtype=np.float32)
        feature_names = sample.get("window_feature_names", cache.payload.get("window_feature_names", []))
        output.append(
            {
                "subject_id": subject,
                "run_id": str(record.get("run_id", sample.get("sample_id", ""))),
                "center": _center(subject, meta, record),
                "channel_names": channels,
                "features": features,
                "window_centers": np.asarray(sample.get("window_relative_centers_sec", np.arange(features.shape[0])), dtype=np.float32),
                "labels_nez": labels_nez,
                "label_source": label_source,
                "feature_names": [str(value) for value in feature_names],
            }
        )
        observed.add(subject)
    missing_runs = sorted(requested - observed)
    if missing_runs:
        raise ValueError(f"Task 1 old-90 patients have no feature runs: {missing_runs[:20]}")
    return output


__all__ = ["task1_feature_records", "task1_record_labels"]
