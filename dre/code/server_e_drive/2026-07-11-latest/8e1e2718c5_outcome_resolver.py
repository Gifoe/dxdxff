from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

OutcomeGroup = Literal["success", "failure", "unknown", "conflict"]


class OutcomeConflictError(ValueError):
    """Raised when a patient has both success and failure outcome evidence."""


@dataclass(frozen=True)
class OutcomePolicy:
    fail_on_conflict: bool = True
    allow_channel_metadata_fallback: bool = True


@dataclass(frozen=True)
class OutcomeCandidate:
    source: str
    raw_value: Any
    group: Literal["success", "failure", "unknown"]
    confidence: Literal["high", "medium", "low", "none"]
    priority: int


@dataclass(frozen=True)
class OutcomeResolution:
    subject_id: str
    normalized_label: int | None
    group: OutcomeGroup
    source: str | None
    raw_value: Any
    confidence: Literal["high", "medium", "low", "none"]
    candidates: tuple[OutcomeCandidate, ...]


_PATIENT_FIELDS = (
    ("outcome_group", "generic", "high", 0),
    ("outcome_raw", "generic", "high", 1),
    ("outcome", "generic", "high", 2),
    ("lzu_outcome_raw", "generic", "high", 3),
    ("engel_score", "engel", "high", 4),
    ("engel", "engel", "high", 5),
    ("surgery_success", "generic", "medium", 6),
    ("surgery_result", "generic", "medium", 7),
    ("seizure_free", "generic", "medium", 8),
    ("success_used", "legacy", "low", 50),
)


def _finite_number(value: Any) -> float | None:
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_)):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    return None


def normalize_outcome_value(value: Any, *, mode: str = "generic") -> Literal["success", "failure", "unknown"]:
    if value is None:
        return "unknown"
    if mode == "legacy":
        return "success" if value is True or value == 1 or str(value).strip().lower() in {"true", "yes", "success"} else "unknown"
    if isinstance(value, (bool, np.bool_)):
        return "success" if bool(value) else "failure"
    numeric = _finite_number(value)
    if numeric is not None:
        if mode == "engel":
            if numeric == 1.0:
                return "success"
            if numeric in {2.0, 3.0, 4.0}:
                return "failure"
            return "unknown"
        if numeric == 1.0:
            return "success"
        if numeric == 0.0:
            return "failure"
        return "unknown"
    text = str(value).strip()
    if not text:
        return "unknown"
    normalized = text.lower().replace("_", " ").replace("-", " ").strip()
    compact = "".join(char for char in normalized if char.isalnum())
    if normalized in {"nan", "none", "null", "na", "n/a", "unknown", "not documented", "not recorded"} or text in {"未知", "不详", "未记录", "未記錄"}:
        return "unknown"
    if mode == "engel" or compact.startswith("engel"):
        suffix = compact[5:] if compact.startswith("engel") else compact
        if suffix in {"i", "1"}:
            return "success"
        if suffix in {"ii", "iii", "iv", "2", "3", "4"}:
            return "failure"
        return "unknown"
    success_values = {"success", "successful", "yes", "true", "seizure free", "engel i", "engel 1", "i"}
    failure_values = {"failure", "fail", "failed", "no", "false", "recurrence", "engel ii", "engel iii", "engel iv", "ii", "iii", "iv"}
    if normalized in success_values or any(token in text for token in ("成功", "无发作", "無發作")):
        return "success"
    if normalized in failure_values or any(token in text for token in ("失败", "失敗", "复发", "復發", "未控制", "仍发作", "仍發作")):
        return "failure"
    return "unknown"


def _candidates_from_mapping(mapping: Any, *, source_prefix: str, priority_offset: int = 0) -> list[OutcomeCandidate]:
    if not isinstance(mapping, Mapping):
        return []
    candidates: list[OutcomeCandidate] = []
    for field, mode, confidence, priority in _PATIENT_FIELDS:
        if field not in mapping:
            continue
        value = mapping.get(field)
        candidates.append(
            OutcomeCandidate(
                source=f"{source_prefix}.{field}",
                raw_value=value,
                group=normalize_outcome_value(value, mode=mode),
                confidence=confidence,
                priority=priority_offset + priority,
            )
        )
    return candidates


def _channel_candidates(value: Any, *, source_prefix: str, priority_offset: int) -> list[OutcomeCandidate]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        return []
    output: list[OutcomeCandidate] = []
    for index, item in enumerate(value):
        for candidate in _candidates_from_mapping(item, source_prefix=f"{source_prefix}[{index}]", priority_offset=priority_offset):
            output.append(
                OutcomeCandidate(
                    source=candidate.source,
                    raw_value=candidate.raw_value,
                    group=candidate.group,
                    confidence="low",
                    priority=max(candidate.priority, priority_offset),
                )
            )
    return output


def resolve_patient_outcome(
    subject_id: str,
    patient_meta: Mapping[str, Any],
    run_records: Iterable[Mapping[str, Any]],
    policy: OutcomePolicy | None = None,
) -> OutcomeResolution:
    active_policy = policy or OutcomePolicy()
    candidates = _candidates_from_mapping(patient_meta, source_prefix="patient", priority_offset=0)
    candidates.extend(_channel_candidates(patient_meta.get("channel_meta"), source_prefix="patient.channel_meta", priority_offset=300))
    for record_index, record in enumerate(run_records):
        if str(record.get("subject_id", subject_id)) != str(subject_id):
            continue
        candidates.extend(_candidates_from_mapping(record, source_prefix=f"run[{record_index}]", priority_offset=100))
        metadata = record.get("metadata")
        sample = record.get("sample")
        candidates.extend(_candidates_from_mapping(metadata, source_prefix=f"run[{record_index}].metadata", priority_offset=110))
        candidates.extend(_candidates_from_mapping(sample, source_prefix=f"run[{record_index}].sample", priority_offset=120))
        candidates.extend(_channel_candidates(record.get("channel_meta"), source_prefix=f"run[{record_index}].channel_meta", priority_offset=310))
        if isinstance(sample, Mapping):
            candidates.extend(_channel_candidates(sample.get("channel_meta"), source_prefix=f"run[{record_index}].sample.channel_meta", priority_offset=320))

    informative = [candidate for candidate in candidates if candidate.group in {"success", "failure"}]
    if not active_policy.allow_channel_metadata_fallback:
        informative = [candidate for candidate in informative if ".channel_meta" not in candidate.source]
    groups = {candidate.group for candidate in informative}
    if groups == {"success", "failure"}:
        resolution = OutcomeResolution(str(subject_id), None, "conflict", None, None, "none", tuple(sorted(candidates, key=lambda item: item.priority)))
        if active_policy.fail_on_conflict:
            details = ", ".join(f"{item.source}={item.group}" for item in informative)
            raise OutcomeConflictError(f"Conflicting outcome values for {subject_id}: {details}")
        return resolution
    if not informative:
        return OutcomeResolution(str(subject_id), None, "unknown", None, None, "none", tuple(sorted(candidates, key=lambda item: item.priority)))
    selected = min(informative, key=lambda item: item.priority)
    return OutcomeResolution(
        subject_id=str(subject_id),
        normalized_label=1 if selected.group == "success" else 0,
        group=selected.group,
        source=selected.source,
        raw_value=selected.raw_value,
        confidence=selected.confidence,
        candidates=tuple(sorted(candidates, key=lambda item: item.priority)),
    )


__all__ = [
    "OutcomeCandidate",
    "OutcomeConflictError",
    "OutcomePolicy",
    "OutcomeResolution",
    "normalize_outcome_value",
    "resolve_patient_outcome",
]
