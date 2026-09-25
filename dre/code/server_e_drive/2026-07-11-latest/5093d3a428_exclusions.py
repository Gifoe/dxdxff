from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd


REQUIRED_COLUMNS = ("patient_key", "seizure_id", "sample_id", "action", "reason")


def load_exclusion_manifest(path: str | Path | None) -> pd.DataFrame:
    if not path:
        return pd.DataFrame(columns=REQUIRED_COLUMNS)
    frame = pd.read_csv(Path(path).expanduser(), dtype=str).fillna("")
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Exclusion manifest missing columns: {missing}")
    frame = frame.loc[:, REQUIRED_COLUMNS].copy()
    unsupported = sorted(set(frame["action"]) - {"exclude_seizure"})
    if unsupported:
        raise ValueError(f"Unsupported exclusion actions: {unsupported}")
    return frame


def _record_values(record: Mapping[str, Any]) -> tuple[str, str, str]:
    sample = record.get("sample") if isinstance(record.get("sample"), Mapping) else {}
    patient = str(record.get("subject_id", record.get("patient_key", "")))
    seizure = str(record.get("run_id", record.get("seizure_id", sample.get("seizure_id", ""))))
    sample_id = str(sample.get("sample_id", record.get("sample_id", seizure)))
    return patient, seizure, sample_id


def exclusion_match(record: Mapping[str, Any], rule: Mapping[str, Any]) -> bool:
    patient, seizure, sample_id = _record_values(record)
    rule_sample = str(rule.get("sample_id", "*"))
    # Run identifiers may include BIDS prefixes/suffixes; the declared seizure token
    # must therefore match the full identifier as a substring.
    return (
        patient.lower() == str(rule.get("patient_key", "")).lower()
        and str(rule.get("seizure_id", "")).lower() in seizure.lower()
        and (rule_sample == "*" or rule_sample.lower() == sample_id.lower())
    )


def apply_exclusions(records: Sequence[Mapping[str, Any]], manifest: pd.DataFrame) -> tuple[list[Mapping[str, Any]], Counter]:
    kept: list[Mapping[str, Any]] = []
    removed: Counter = Counter()
    rules = manifest.to_dict("records")
    for record in records:
        matched = next((index for index, rule in enumerate(rules) if exclusion_match(record, rule)), None)
        if matched is None:
            kept.append(record)
        else:
            removed[matched] += 1
    return kept, removed


def build_exclusion_audit(
    manifest: pd.DataFrame,
    feature_records: Sequence[Mapping[str, Any]],
    raw_records: Sequence[Mapping[str, Any]],
) -> pd.DataFrame:
    feature_kept, feature_removed = apply_exclusions(feature_records, manifest)
    raw_kept, raw_removed = apply_exclusions(raw_records, manifest)
    rows = []
    for index, rule in enumerate(manifest.to_dict("records")):
        patient = str(rule["patient_key"])
        remaining = {
            _record_values(record)[1]
            for record in feature_kept
            if _record_values(record)[0].lower() == patient.lower()
        }
        rows.append({
            **rule,
            "feature_records_removed": int(feature_removed[index]),
            "raw_records_removed": int(raw_removed[index]),
            "remaining_seizures": len(remaining),
            "patient_retained": bool(remaining),
        })
    return pd.DataFrame(rows, columns=(*REQUIRED_COLUMNS, "feature_records_removed", "raw_records_removed", "remaining_seizures", "patient_retained"))


__all__ = ["apply_exclusions", "build_exclusion_audit", "exclusion_match", "load_exclusion_manifest"]
