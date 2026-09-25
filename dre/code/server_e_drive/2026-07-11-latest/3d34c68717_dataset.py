from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .cache_schema import CachePayload
from .leakage_guard import assert_label_blind_tree
from .outcome_resolver import OutcomePolicy, resolve_patient_outcome


class DatasetContractError(ValueError):
    """Raised when cache records cannot form an unambiguous label-blind example."""


@dataclass(frozen=True)
class OutcomePatientExample:
    subject_id: str
    center: str
    target: float
    canonical_channels: tuple[str, ...]
    model_input: dict[str, Any]
    side_metadata: dict[str, Any]


def normalize_channel_name(value: Any) -> str:
    return "".join(str(value).strip().upper().replace("-", "").replace("_", "").split())


def _record_sample(record: Mapping[str, Any]) -> Mapping[str, Any]:
    sample = record.get("sample")
    return sample if isinstance(sample, Mapping) else {}


def _record_identity(record: Mapping[str, Any]) -> tuple[str, str, str]:
    sample = _record_sample(record)
    subject_id = str(record.get("subject_id", ""))
    run_id = str(record.get("run_id", ""))
    return subject_id, run_id, str(sample.get("sample_id", run_id))


def _duplicate_records_are_equal(records: Sequence[Mapping[str, Any]]) -> bool:
    first = records[0]
    first_sample = _record_sample(first)
    first_channels = _record_channels(first)
    first_features = np.asarray(first_sample.get("window_features"))
    first_centers = np.asarray(first_sample.get("window_relative_centers_sec"))
    return all(
        _record_channels(record) == first_channels
        and np.array_equal(np.asarray(_record_sample(record).get("window_features")), first_features)
        and np.array_equal(np.asarray(_record_sample(record).get("window_relative_centers_sec")), first_centers)
        for record in records[1:]
    )


def _record_channels(record: Mapping[str, Any]) -> list[str]:
    sample = _record_sample(record)
    raw = record.get("channel_names_norm", sample.get("channel_names_norm", sample.get("channel_names", [])))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raise DatasetContractError(f"Record {record.get('run_id')} has no usable channel name sequence.")
    channels = [normalize_channel_name(value) for value in raw]
    duplicates = sorted({name for name in channels if channels.count(name) > 1})
    if duplicates:
        raise DatasetContractError(f"Record {record.get('run_id')} has duplicate local channel name(s): {duplicates}")
    return channels


def _center(subject_id: str, meta: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> str:
    for source in (meta, *(records[:1])):
        for key in ("source_center", "center", "source_dataset"):
            value = str(source.get(key, "")).strip().lower()
            if value:
                return value
    return subject_id.split(":", 1)[0].lower() if ":" in subject_id else "unknown"


def _align_feature_run(
    record: Mapping[str, Any],
    canonical_channels: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    sample = _record_sample(record)
    features = np.asarray(sample.get("window_features"), dtype=np.float32)
    if features.ndim != 3:
        raise DatasetContractError(
            f"Record {record.get('run_id')} window_features must be [window,channel,feature], got {features.shape}."
        )
    local_channels = _record_channels(record)
    if features.shape[1] != len(local_channels):
        raise DatasetContractError(
            f"Record {record.get('run_id')} channel axis {features.shape[1]} does not match {len(local_channels)} names."
        )
    canonical_lookup = {name: index for index, name in enumerate(canonical_channels)}
    mapping: dict[str, int] = {}
    aligned = np.zeros((features.shape[0], len(canonical_channels), features.shape[2]), dtype=np.float32)
    present = np.zeros((len(canonical_channels),), dtype=bool)
    for local_index, channel in enumerate(local_channels):
        if channel not in canonical_lookup:
            continue
        canonical_index = canonical_lookup[channel]
        aligned[:, canonical_index, :] = features[:, local_index, :]
        present[canonical_index] = True
        mapping[str(local_index)] = canonical_index
    centers = np.asarray(sample.get("window_relative_centers_sec", []), dtype=np.float32).reshape(-1)
    if centers.size != aligned.shape[0]:
        raise DatasetContractError(
            f"Record {record.get('run_id')} has {aligned.shape[0]} feature windows but {centers.size} window centers."
        )
    order = np.argsort(centers, kind="stable")
    return aligned[order], centers[order], present, mapping


def _safe_channel_metadata(meta: Mapping[str, Any], canonical_channels: tuple[str, ...]) -> list[dict[str, Any]]:
    entries = meta.get("channel_meta", [])
    if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
        return []
    forbidden_fragments = ("label", "ez", "soz", "resect", "outcome", "engel", "success", "postop")
    safe: list[dict[str, Any]] = []
    for index, channel in enumerate(canonical_channels):
        source = entries[index] if index < len(entries) and isinstance(entries[index], Mapping) else {}
        clean = {
            str(key): value
            for key, value in source.items()
            if not any(fragment in str(key).lower() for fragment in forbidden_fragments)
            and isinstance(value, (str, int, float, bool, type(None)))
        }
        clean["canonical_channel"] = channel
        safe.append(clean)
    return safe


def build_outcome_patient_examples(
    feature_cache: CachePayload,
    *,
    subject_ids: Sequence[str] | None = None,
    outcome_policy: OutcomePolicy | None = None,
    exclusion_audit: list[dict[str, Any]] | None = None,
) -> list[OutcomePatientExample]:
    selected = None if subject_ids is None else {str(value) for value in subject_ids}
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    records_by_identity: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for record in feature_cache.run_records:
        records_by_identity[_record_identity(record)].append(record)
    for identity, records in sorted(records_by_identity.items()):
        subject_id, run_id, sample_id = identity
        if selected is not None and subject_id not in selected:
            continue
        if len(records) == 1:
            grouped[subject_id].append(records[0])
            continue
        equal = _duplicate_records_are_equal(records)
        if exclusion_audit is not None:
            exclusion_audit.append(
                {
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "reason": "duplicate_run_identical_deduplicated" if equal else "duplicate_run_conflict_excluded",
                    "duplicate_count": len(records),
                }
            )
        if equal:
            grouped[subject_id].append(records[0])

    examples: list[OutcomePatientExample] = []
    for subject_id in sorted(grouped):
        meta = feature_cache.patient_index.get(subject_id)
        if meta is None:
            raise DatasetContractError(f"Patient {subject_id} is present in run_records but missing from patient_index.")
        resolution = resolve_patient_outcome(subject_id, meta, grouped[subject_id], outcome_policy or OutcomePolicy())
        if resolution.normalized_label is None:
            continue
        raw_canonical = meta.get("canonical_channels")
        if not isinstance(raw_canonical, Sequence) or isinstance(raw_canonical, (str, bytes)) or not raw_canonical:
            raise DatasetContractError(f"Patient {subject_id} has no canonical_channels.")
        canonical_channels = tuple(normalize_channel_name(value) for value in raw_canonical)
        if len(set(canonical_channels)) != len(canonical_channels):
            raise DatasetContractError(f"Patient {subject_id} has duplicate canonical channel names.")

        feature_runs: list[np.ndarray] = []
        window_centers: list[np.ndarray] = []
        seizure_channel_masks: list[np.ndarray] = []
        mappings: list[dict[str, int]] = []
        run_ids: list[str] = []
        sample_ids: list[str] = []
        for record in sorted(grouped[subject_id], key=lambda item: (str(item.get("run_id", "")), str(_record_sample(item).get("sample_id", "")))):
            aligned, centers, present, mapping = _align_feature_run(record, canonical_channels)
            feature_runs.append(aligned)
            window_centers.append(centers)
            seizure_channel_masks.append(present)
            mappings.append(mapping)
            run_id = str(record.get("run_id", ""))
            run_ids.append(run_id)
            sample_ids.append(str(_record_sample(record).get("sample_id", run_id)))

        model_input = {
            "feature_runs": feature_runs,
            "window_centers": window_centers,
            "seizure_channel_mask": seizure_channel_masks,
            "canonical_index": np.arange(len(canonical_channels), dtype=np.int64),
        }
        assert_label_blind_tree(model_input, stage="patient_example")
        examples.append(
            OutcomePatientExample(
                subject_id=subject_id,
                center=_center(subject_id, meta, grouped[subject_id]),
                target=float(resolution.normalized_label),
                canonical_channels=canonical_channels,
                model_input=model_input,
                side_metadata={
                    "run_ids": tuple(run_ids),
                    "sample_ids": tuple(sample_ids),
                    "local_to_canonical": tuple(mappings),
                    "channel_meta_safe": tuple(_safe_channel_metadata(meta, canonical_channels)),
                    "outcome_source": resolution.source,
                    "outcome_confidence": resolution.confidence,
                },
            )
        )
    return examples


__all__ = [
    "DatasetContractError",
    "OutcomePatientExample",
    "build_outcome_patient_examples",
    "normalize_channel_name",
]
