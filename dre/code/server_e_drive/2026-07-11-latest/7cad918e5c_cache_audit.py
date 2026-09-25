from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .cache_schema import CachePayload, load_cache_contract
from .leakage_guard import assert_label_blind_tree, write_leakage_audit
from .manifest import json_safe, write_csv_atomic, write_json_atomic
from .outcome_resolver import OutcomeConflictError, OutcomePolicy, OutcomeResolution, resolve_patient_outcome


REQUIRED_AUDIT_FILES = (
    "outcome_cache_schema_feature.json",
    "outcome_cache_schema_raw.json",
    "outcome_patient_manifest.csv",
    "outcome_run_manifest.csv",
    "outcome_channel_manifest.csv",
    "outcome_raw_feature_alignment.csv",
    "outcome_outcome_field_audit.csv",
    "outcome_outcome_conflicts.csv",
    "outcome_center_distribution.csv",
    "outcome_quality_distribution.csv",
    "outcome_missingness_report.csv",
    "outcome_duplicate_subject_audit.csv",
    "outcome_leakage_audit.json",
)


@dataclass(frozen=True)
class AuditSummary:
    patient_count: int
    run_count_feature: int
    run_count_raw: int
    aligned_run_count: int
    outcome_counts: dict[str, int]
    conflict_subjects: tuple[str, ...]
    unknown_subjects: tuple[str, ...]


def _sample(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("sample")
    return value if isinstance(value, Mapping) else {}


def _record_key(record: Mapping[str, Any]) -> tuple[str, str, str]:
    sample = _sample(record)
    subject = str(record.get("subject_id", sample.get("subject_id", "")))
    run = str(record.get("run_id", sample.get("run_id", "")))
    sample_id = str(sample.get("sample_id", record.get("sample_id", run)))
    return subject, run, sample_id


def _channels(record: Mapping[str, Any]) -> list[str]:
    sample = _sample(record)
    values = record.get("channel_names_norm", sample.get("channel_names_norm", sample.get("channel_names", [])))
    return [str(value) for value in values] if isinstance(values, Sequence) and not isinstance(values, (str, bytes)) else []


def _window_centers(record: Mapping[str, Any]) -> np.ndarray:
    sample = _sample(record)
    return np.asarray(sample.get("window_relative_centers_sec", []), dtype=np.float64).reshape(-1)


def _center(subject_id: str, meta: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> str:
    for source in (meta, *(records[:1])):
        for key in ("source_center", "center", "source_dataset"):
            value = str(source.get(key, "")).strip().lower()
            if value:
                return value
    return subject_id.split(":", 1)[0].lower() if ":" in subject_id else "unknown"


def _schema(cache: CachePayload, kind: str) -> dict[str, Any]:
    sample_record = cache.run_records[0] if cache.run_records else {}
    sample = _sample(sample_record)
    arrays = {
        str(key): {"shape": list(value.shape), "dtype": str(value.dtype)}
        for key, value in sample.items()
        if isinstance(value, np.ndarray)
    }
    return {
        "cache_kind": kind,
        "source_path": str(cache.source_path) if cache.source_path else None,
        "top_level_keys": list(cache.top_level_keys),
        "n_run_records": len(cache.run_records),
        "n_patient_index": len(cache.patient_index),
        "sample_record_keys": sorted(str(key) for key in sample_record),
        "sample_keys": sorted(str(key) for key in sample),
        "sample_array_contract": arrays,
    }


def build_alignment_rows(feature: CachePayload, raw: CachePayload) -> list[dict[str, Any]]:
    feature_lookup = {_record_key(record): record for record in feature.run_records}
    raw_lookup = {_record_key(record): record for record in raw.run_records}
    rows: list[dict[str, Any]] = []
    for key in sorted(set(feature_lookup) | set(raw_lookup)):
        feature_record = feature_lookup.get(key)
        raw_record = raw_lookup.get(key)
        if feature_record is None:
            status = "raw_only"
        elif raw_record is None:
            status = "feature_only"
        else:
            status = "matched"
        feature_channels = _channels(feature_record or {})
        raw_channels = _channels(raw_record or {})
        shared_channels = set(feature_channels) & set(raw_channels)
        channel_ratio = len(shared_channels) / max(len(set(feature_channels)), 1) if feature_record is not None else 0.0
        feature_centers = _window_centers(feature_record or {})
        raw_centers = _window_centers(raw_record or {})
        if feature_centers.size and raw_centers.size:
            compared = min(feature_centers.size, raw_centers.size)
            center_ratio = float(np.isclose(feature_centers[:compared], raw_centers[:compared], atol=1e-4).sum() / max(feature_centers.size, 1))
        else:
            center_ratio = 0.0
        rows.append(
            {
                "subject_id": key[0],
                "run_id": key[1],
                "sample_id": key[2],
                "alignment_status": status,
                "feature_channel_count": len(feature_channels),
                "raw_channel_count": len(raw_channels),
                "shared_channel_count": len(shared_channels),
                "channel_alignment_ratio": float(channel_ratio),
                "feature_window_count": int(feature_centers.size),
                "raw_window_count": int(raw_centers.size),
                "window_center_alignment_ratio": center_ratio,
            }
        )
    return rows


def _quality_values(record: Mapping[str, Any]) -> list[tuple[str, str]]:
    values: list[tuple[str, str]] = []
    for source_name, source in (("record", record), ("sample", _sample(record))):
        for key, value in source.items():
            if "quality" in str(key).lower() and isinstance(value, (str, int, float, bool)):
                values.append((f"{source_name}.{key}", str(value)))
    return values


def audit_caches(
    feature_source: str | Path | Mapping[str, Any],
    raw_source: str | Path | Mapping[str, Any],
    output_dir: str | Path,
    *,
    fail_on_conflict: bool = False,
) -> AuditSummary:
    feature = load_cache_contract(feature_source)
    raw = load_cache_contract(raw_source)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output / "outcome_cache_schema_feature.json", _schema(feature, "feature"))
    write_json_atomic(output / "outcome_cache_schema_raw.json", _schema(raw, "raw"))

    feature_by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in feature.run_records:
        feature_by_subject[str(record.get("subject_id", ""))].append(record)
    raw_by_subject: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in raw.run_records:
        raw_by_subject[str(record.get("subject_id", ""))].append(record)

    subjects = sorted(set(feature.patient_index) | set(raw.patient_index) | set(feature_by_subject) | set(raw_by_subject))
    patient_rows: list[dict[str, Any]] = []
    outcome_rows: list[dict[str, Any]] = []
    conflict_rows: list[dict[str, Any]] = []
    resolutions: dict[str, OutcomeResolution] = {}
    policy = OutcomePolicy(fail_on_conflict=False)
    for subject_id in subjects:
        meta = feature.patient_index.get(subject_id, raw.patient_index.get(subject_id, {}))
        records = feature_by_subject.get(subject_id, []) + raw_by_subject.get(subject_id, [])
        resolution = resolve_patient_outcome(subject_id, meta, records, policy)
        resolutions[subject_id] = resolution
        center = _center(subject_id, meta, records)
        canonical = list(meta.get("canonical_channels", [])) if isinstance(meta, Mapping) else []
        patient_rows.append(
            {
                "subject_id": subject_id,
                "center": center,
                "outcome_group": resolution.group,
                "outcome_label": resolution.normalized_label,
                "outcome_source": resolution.source,
                "outcome_raw": resolution.raw_value,
                "outcome_confidence": resolution.confidence,
                "feature_run_count": len(feature_by_subject.get(subject_id, [])),
                "raw_run_count": len(raw_by_subject.get(subject_id, [])),
                "canonical_channel_count": len(canonical),
            }
        )
        for candidate in resolution.candidates:
            outcome_rows.append({"subject_id": subject_id, **asdict(candidate)})
        if resolution.group == "conflict":
            conflict_rows.extend({"subject_id": subject_id, **asdict(candidate)} for candidate in resolution.candidates if candidate.group in {"success", "failure"})

    if fail_on_conflict and conflict_rows:
        preview = sorted({row["subject_id"] for row in conflict_rows})[:20]
        raise OutcomeConflictError(f"Outcome conflicts detected for {preview}; inspect outcome_outcome_conflicts.csv")

    run_rows: list[dict[str, Any]] = []
    channel_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    missingness: Counter[str] = Counter()
    for cache_kind, cache in (("feature", feature), ("raw", raw)):
        for record in cache.run_records:
            subject_id, run_id, sample_id = _record_key(record)
            sample = _sample(record)
            channels = _channels(record)
            centers = _window_centers(record)
            features = np.asarray(sample.get("window_features", []))
            waveform = np.asarray(sample.get("raw_waveform", []))
            run_rows.append(
                {
                    "cache_kind": cache_kind,
                    "subject_id": subject_id,
                    "run_id": run_id,
                    "sample_id": sample_id,
                    "channel_count": len(channels),
                    "window_count": int(centers.size),
                    "feature_shape": str(tuple(features.shape)) if features.size else "",
                    "raw_shape": str(tuple(waveform.shape)) if waveform.size else "",
                    "sampling_frequency": sample.get("raw_temporal_sfreq", record.get("sfreq")),
                }
            )
            for index, channel in enumerate(channels):
                channel_rows.append({"cache_kind": cache_kind, "subject_id": subject_id, "run_id": run_id, "sample_id": sample_id, "local_channel_index": index, "channel_name": channel})
            for field in ("subject_id", "run_id"):
                if not str(record.get(field, "")).strip():
                    missingness[f"{cache_kind}.record.{field}"] += 1
            if not channels:
                missingness[f"{cache_kind}.channel_names"] += 1
            if not centers.size:
                missingness[f"{cache_kind}.window_relative_centers_sec"] += 1
            for field, value in _quality_values(record):
                quality_rows.append({"cache_kind": cache_kind, "subject_id": subject_id, "run_id": run_id, "field": field, "value": value})

    alignment_rows = build_alignment_rows(feature, raw)
    write_csv_atomic(output / "outcome_patient_manifest.csv", patient_rows)
    write_csv_atomic(output / "outcome_run_manifest.csv", run_rows)
    write_csv_atomic(output / "outcome_channel_manifest.csv", channel_rows)
    write_csv_atomic(output / "outcome_raw_feature_alignment.csv", alignment_rows)
    write_csv_atomic(output / "outcome_outcome_field_audit.csv", outcome_rows)
    write_csv_atomic(output / "outcome_outcome_conflicts.csv", conflict_rows, columns=("subject_id", "source", "raw_value", "group", "confidence", "priority"))

    center_counts = Counter((row["center"], row["outcome_group"]) for row in patient_rows)
    center_rows = [{"center": center, "outcome_group": outcome, "patient_count": count} for (center, outcome), count in sorted(center_counts.items())]
    write_csv_atomic(output / "outcome_center_distribution.csv", center_rows)
    quality_counts = Counter((row["cache_kind"], row["field"], row["value"]) for row in quality_rows)
    write_csv_atomic(output / "outcome_quality_distribution.csv", [{"cache_kind": kind, "field": field, "value": value, "run_count": count} for (kind, field, value), count in sorted(quality_counts.items())])
    write_csv_atomic(output / "outcome_missingness_report.csv", [{"field": field, "missing_count": count} for field, count in sorted(missingness.items())])

    duplicate_rows = []
    for cache_kind, cache in (("feature", feature), ("raw", raw)):
        grouped_records: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
        for record in cache.run_records:
            grouped_records[_record_key(record)].append(record)
        for key, records in sorted(grouped_records.items()):
            if len(records) < 2:
                continue
            first_sample = _sample(records[0])
            comparison_keys = ["window_features", "window_relative_centers_sec"]
            if cache_kind == "raw":
                comparison_keys.append("raw_waveform")
            identical = all(
                _channels(record) == _channels(records[0])
                and all(np.array_equal(np.asarray(_sample(record).get(field)), np.asarray(first_sample.get(field))) for field in comparison_keys)
                for record in records[1:]
            )
            duplicate_rows.append(
                {
                    "cache_kind": cache_kind,
                    "subject_id": key[0],
                    "run_id": key[1],
                    "sample_id": key[2],
                    "duplicate_count": len(records),
                    "duplicate_status": "identical_deduplicate" if identical else "conflict_exclude_run",
                }
            )
    write_csv_atomic(output / "outcome_duplicate_subject_audit.csv", duplicate_rows, columns=("cache_kind", "subject_id", "run_id", "sample_id", "duplicate_count", "duplicate_status"))

    safe_scan = assert_label_blind_tree(
        {
            "feature_x": [],
            "raw_x": [],
            "seizure_mask": [],
            "window_mask": [],
            "channel_mask": [],
            "window_centers": [],
            "canonical_index": [],
        },
        stage="audit_model_input_template",
    )
    write_leakage_audit([safe_scan], output / "outcome_leakage_audit.json")

    outcome_counts = Counter(resolution.group for resolution in resolutions.values())
    summary = AuditSummary(
        patient_count=len(subjects),
        run_count_feature=len(feature.run_records),
        run_count_raw=len(raw.run_records),
        aligned_run_count=sum(row["alignment_status"] == "matched" for row in alignment_rows),
        outcome_counts={key: int(outcome_counts.get(key, 0)) for key in ("success", "failure", "unknown", "conflict")},
        conflict_subjects=tuple(sorted(subject for subject, resolution in resolutions.items() if resolution.group == "conflict")),
        unknown_subjects=tuple(sorted(subject for subject, resolution in resolutions.items() if resolution.group == "unknown")),
    )
    write_json_atomic(output / "outcome_audit_summary.json", asdict(summary))
    return summary


__all__ = ["AuditSummary", "REQUIRED_AUDIT_FILES", "audit_caches", "build_alignment_rows"]
