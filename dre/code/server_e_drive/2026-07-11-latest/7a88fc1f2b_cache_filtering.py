from __future__ import annotations

import hashlib
from collections import Counter
from typing import Any

import numpy as np


def _hash(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[:16]


def resolve_channel_names(record: dict[str, Any], patient_index: dict[str, Any], width: int) -> list[str] | None:
    sample = record.get("sample") or {}
    meta = patient_index.get(str(record.get("subject_id", "")), {}) or {}
    for value in (record.get("canonical_channels"), sample.get("canonical_channels"),
                  record.get("channel_names"), sample.get("channel_names"),
                  meta.get("canonical_channels")):
        if isinstance(value, (list, tuple)) and len(value) == width:
            return [str(item) for item in value]
    return None


def _record_key(record: object, index: int) -> tuple[str, str]:
    if not isinstance(record, dict):
        return "", f"index:{index}"
    return str(record.get("subject_id", "")), str(record.get("run_id", record.get("record_id", f"index:{index}")))


def filter_valid_cache_records(
    cache: dict[str, Any], *, invalid_record_policy: str = "drop",
    strict_patient_coverage: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a filtered copy plus privacy-preserving record and coverage audits.

    The source mapping and all contained tensors are treated as read-only. Every
    member of a duplicated subject/run group is rejected.
    """
    if invalid_record_policy not in {"fail", "drop"}:
        raise ValueError("invalid_record_policy must be 'fail' or 'drop'")
    if not isinstance(cache, dict) or not isinstance(cache.get("run_records"), list) or not isinstance(cache.get("patient_index"), dict):
        raise ValueError("cache requires run_records list and patient_index mapping")
    records = cache["run_records"]
    patient_index = cache["patient_index"]
    keys = [_record_key(record, index) for index, record in enumerate(records)]
    duplicate_keys = {key for key, count in Counter(keys).items() if count > 1}
    declared_features = cache.get("feature_names") or cache.get("window_feature_names") or ()
    expected_dim = len(declared_features) or None
    if expected_dim is None:
        observed = [int(np.asarray((record.get("sample") or {}).get("window_features")).shape[2])
                    for record in records if isinstance(record, dict)
                    and np.asarray((record.get("sample") or {}).get("window_features")).ndim == 3]
        expected_dim = Counter(observed).most_common(1)[0][0] if observed else None

    retained: list[dict[str, Any]] = []
    retained_rows: list[dict[str, Any]] = []
    excluded_rows: list[dict[str, Any]] = []
    input_subjects = {key[0] for key in keys if key[0]}
    for index, record in enumerate(records):
        subject, run_id = keys[index]
        values = np.asarray((record.get("sample") or {}).get("window_features")) if isinstance(record, dict) else np.asarray(None)
        shape = tuple(int(item) for item in values.shape)
        width = shape[1] if values.ndim == 3 else 0
        names = resolve_channel_names(record, patient_index, width) if isinstance(record, dict) and values.ndim == 3 else None
        reason: str | None = None
        if keys[index] in duplicate_keys:
            reason = "duplicate_subject_run"
        elif not isinstance(record, dict) or not subject:
            reason = "invalid_record_mapping"
        elif values.ndim != 3 or min(values.shape, default=0) <= 0:
            reason = "invalid_shape"
        elif expected_dim is not None and int(values.shape[2]) != int(expected_dim):
            reason = "feature_dimension_mismatch"
        elif names is None:
            candidate_sources = []
            sample = record.get("sample") or {}
            meta = patient_index.get(subject, {}) or {}
            for candidate in (record.get("canonical_channels"), sample.get("canonical_channels"),
                              record.get("channel_names"), sample.get("channel_names"),
                              meta.get("canonical_channels")):
                if isinstance(candidate, (list, tuple)):
                    candidate_sources.append(candidate)
            reason = "channel_axis_mismatch" if candidate_sources else "missing_channel_source"
        elif not np.isfinite(values.astype(float, copy=False)).any():
            reason = "all_nonfinite"
        row = {
            "subject_id_hash": _hash(subject), "run_id_hash": _hash(run_id),
            "reason": reason or "retained", "window_shape": list(shape),
            "channel_count": int(width), "feature_dim": int(shape[2]) if values.ndim == 3 else 0,
        }
        if reason is not None:
            excluded_rows.append(row)
            if invalid_record_policy == "fail":
                raise ValueError(f"cache record {index} rejected: {reason}")
        else:
            retained.append(record)
            retained_rows.append(row)

    retained_subjects = {str(record.get("subject_id")) for record in retained}
    coverage_rows = [{"subject_id_hash": _hash(subject),
                      "input_run_count": int(sum(key[0] == subject for key in keys)),
                      "retained_run_count": int(sum(str(record.get("subject_id")) == subject for record in retained)),
                      "status": "retained" if subject in retained_subjects else "zero_valid_runs"}
                     for subject in sorted(input_subjects)]
    zero_valid = [row for row in coverage_rows if row["retained_run_count"] == 0]
    if strict_patient_coverage and zero_valid:
        raise ValueError(f"cache filtering leaves {len(zero_valid)} patients with zero valid runs")
    summary = {
        "input_records": len(records), "retained_records": len(retained),
        "excluded_records": len(excluded_rows), "expected_feature_dim": expected_dim,
        "reason_counts": dict(Counter(row["reason"] for row in excluded_rows)),
        "input_patients": len(input_subjects), "retained_patients": len(retained_subjects),
        "zero_valid_run_patients": len(zero_valid),
    }
    filtered = dict(cache)
    filtered["run_records"] = list(retained)
    audit = {"summary": summary, "excluded_records": excluded_rows,
             "retained_records": retained_rows, "patient_coverage": coverage_rows}
    return filtered, audit


def filter_cache_records(records: list[dict[str, Any]], patient_index: dict[str, Any], *, policy: str = "fail") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compatibility wrapper for callers that still pass cache components."""
    filtered, audit = filter_valid_cache_records(
        {"run_records": records, "patient_index": patient_index},
        invalid_record_policy=policy, strict_patient_coverage=False)
    summary = dict(audit["summary"])
    for reason, count in summary.pop("reason_counts", {}).items():
        summary[f"dropped_{reason}"] = count
    summary["excluded_records_detail"] = audit["excluded_records"]
    summary["retained_records_detail"] = audit["retained_records"]
    summary["patient_coverage"] = audit["patient_coverage"]
    return filtered["run_records"], summary
