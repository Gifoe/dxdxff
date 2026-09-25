from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .data_interface import PatientRecord


AUDIT_FIELDS = [
    "center_id",
    "patient_id",
    "global_patient_id",
    "n_edf",
    "n_seizures",
    "n_valid_channels",
    "n_bad_channels",
    "n_ez_raw",
    "n_ez_matched",
    "n_ez_unmatched",
    "ez_unmatched_names",
    "ez_ratio",
    "bad_ez_overlap",
    "bad_ez_overlap_names",
    "edf_channel_examples",
    "label_channel_examples",
    "skipped",
    "skip_reason",
    "warning",
]


def audit_patient_records(
    records: Sequence[PatientRecord],
    center_id: str,
    output_csv: str | Path,
) -> list[dict[str, Any]]:
    rows = [audit_patient_record(record, center_id) for record in records]
    write_audit_csv(rows, output_csv)
    return rows


def audit_patient_record(record: PatientRecord, center_id: str | None = None) -> dict[str, Any]:
    center = str(center_id or record.center_id or "unknown")
    patient_id = str(record.patient_id or record.subject_id)
    global_patient_id = str(record.subject_id)
    n_valid = int(len(record.canonical_channels))
    labels = np.asarray(record.labels, dtype=np.float32)
    n_ez_matched = int((labels > 0.5).sum())
    raw_ez = _raw_ez_channels(record)
    raw_bad = set(str(item) for item in record.bad_channels)
    matched_ez = set(record.ez_channels)
    unmatched = sorted(set(raw_ez) - matched_ez)
    bad_overlap = sorted(raw_bad & (set(raw_ez) | matched_ez))
    ez_ratio = float(n_ez_matched / max(n_valid, 1))
    warnings = []
    if ez_ratio > 0.45:
        warnings.append("ez_ratio_gt_0.45")
    if bad_overlap:
        warnings.append("bad_ez_overlap_bad_channel_masked")

    skipped = False
    skip_reasons = []
    if n_ez_matched == 0:
        skipped = True
        skip_reasons.append("n_ez_matched_eq_0")
    if n_valid < 8:
        skipped = True
        skip_reasons.append("n_valid_channels_lt_8")
    if len(record.seizures) == 0:
        skipped = True
        skip_reasons.append("n_seizures_eq_0")

    return {
        "center_id": center,
        "patient_id": patient_id,
        "global_patient_id": global_patient_id,
        "n_edf": len(record.edf_files),
        "n_seizures": len(record.seizures),
        "n_valid_channels": n_valid,
        "n_bad_channels": len(record.bad_channels),
        "n_ez_raw": len(raw_ez),
        "n_ez_matched": n_ez_matched,
        "n_ez_unmatched": len(unmatched),
        "ez_unmatched_names": ";".join(unmatched),
        "ez_ratio": ez_ratio,
        "bad_ez_overlap": len(bad_overlap),
        "bad_ez_overlap_names": ";".join(bad_overlap),
        "edf_channel_examples": ";".join(record.canonical_channels[:8]),
        "label_channel_examples": ";".join(sorted(raw_ez)[:8]),
        "skipped": int(skipped),
        "skip_reason": ";".join(skip_reasons),
        "warning": ";".join(warnings),
    }


def write_audit_csv(rows: Sequence[dict[str, Any]], output_csv: str | Path) -> None:
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=AUDIT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in AUDIT_FIELDS})


def write_audit_summary(
    center_rows: dict[str, Sequence[dict[str, Any]]],
    output_json: str | Path,
) -> dict[str, Any]:
    summary = {}
    for center_id, rows in center_rows.items():
        rows = list(rows)
        summary[center_id] = {
            "n_patients": len(rows),
            "n_skipped_patients": int(sum(int(row.get("skipped", 0)) for row in rows)),
            "n_trainable_patients": int(sum(1 - int(row.get("skipped", 0)) for row in rows)),
            "n_total_seizures": int(sum(int(row.get("n_seizures", 0)) for row in rows)),
            "n_zero_ez_patients": int(sum(row.get("skip_reason", "").find("n_ez_matched_eq_0") >= 0 for row in rows)),
            "n_warning_patients": int(sum(bool(row.get("warning")) for row in rows)),
        }
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as fout:
        json.dump(summary, fout, ensure_ascii=False, indent=2)
    return summary


def filter_trainable_patients(records: Sequence[PatientRecord], audit_rows: Sequence[dict[str, Any]]) -> list[PatientRecord]:
    keep = {row["global_patient_id"] for row in audit_rows if not int(row.get("skipped", 0))}
    return [record for record in records if record.subject_id in keep]


def _raw_ez_channels(record: PatientRecord) -> set[str]:
    raw: set[str] = set(record.ez_channels)
    for meta in record.channel_meta:
        text = str(meta.get("ez_channels_raw", "")).strip()
        if not text:
            continue
        for token in text.replace(",", ";").replace("，", ";").replace("、", ";").split(";"):
            token = token.strip()
            if token:
                raw.add(token.upper())
    return raw


__all__ = [
    "AUDIT_FIELDS",
    "audit_patient_record",
    "audit_patient_records",
    "filter_trainable_patients",
    "write_audit_csv",
    "write_audit_summary",
]
