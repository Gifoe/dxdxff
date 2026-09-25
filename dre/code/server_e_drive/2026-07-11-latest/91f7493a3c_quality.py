from __future__ import annotations

from collections.abc import Mapping
from typing import Any


QUALITY_LABELS = ("good", "review", "poor")
QUALITY_ALIASES = {
    "good": "good",
    "ok": "good",
    "clean": "good",
    "pass": "good",
    "usable": "good",
    "review": "review",
    "needs_review": "review",
    "manual_review": "review",
    "borderline": "review",
    "questionable": "review",
    "poor": "poor",
    "bad": "poor",
    "fail": "poor",
    "failed": "poor",
    "reject": "poor",
    "unusable": "poor",
}


def normalize_quality_label(value: Any) -> str:
    text = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    if text in QUALITY_ALIASES:
        return QUALITY_ALIASES[text]
    raise ValueError(f"Unsupported EDF quality label {value!r}; expected one of {QUALITY_LABELS}.")


def quality_weight(label: str, *, review_weight: float, poor_weight: float) -> float:
    normalized = normalize_quality_label(label)
    if normalized == "good":
        return 1.0
    if normalized == "review":
        return float(review_weight)
    return float(poor_weight)


def resolve_dotted(mapping: Mapping[str, Any], field: str) -> Any:
    current: Any = mapping
    for part in str(field).split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise KeyError(field)
    return current


def resolve_quality_label(sample: Mapping[str, Any], field: str) -> str:
    if not str(field).strip():
        raise ValueError("--edf_quality_field is required when --use_edf_quality_weighting is enabled.")
    sources = [sample]
    metadata = sample.get("quality_metadata") if isinstance(sample, Mapping) else None
    if isinstance(metadata, Mapping):
        sources.append(metadata)
    for source in sources:
        try:
            return normalize_quality_label(resolve_dotted(source, field))
        except KeyError:
            continue
    raise KeyError(f"Missing EDF quality field {field!r} for sample {sample.get('sample_id', sample.get('run_id', '<unknown>'))!r}.")


def is_simple_quality_metadata(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (str, int, float, bool)):
        return True
    if isinstance(value, Mapping):
        return all(isinstance(key, str) and is_simple_quality_metadata(child) for key, child in value.items())
    return False


__all__ = [
    "QUALITY_LABELS",
    "normalize_quality_label",
    "quality_weight",
    "resolve_quality_label",
    "is_simple_quality_metadata",
]
