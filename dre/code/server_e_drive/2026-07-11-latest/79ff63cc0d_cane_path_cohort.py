"""Strict post-hoc sensitivity cohort filtering without changing All90 folds."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence


EXPECTED_CENTER_COUNTS = {"hup": 36, "lzu": 21, "multicenter": 15, "pediatric": 8}
EXPECTED_EXCLUSION_STATUS = "unconfirmed_posthoc_sensitivity_exclusion"
EXPECTED_EXCLUSION_SOURCE = "posthoc_oracle_review"


def canonical_subject_id(value: Any) -> str:
    return str(value or "").strip().casefold()


def subject_center(subject_id: str) -> str:
    return str(subject_id).split(":", 1)[0].strip().lower()


def read_exclusion_manifest(path: str | Path) -> list[dict[str, str]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Sensitivity exclusion manifest not found: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [dict(row) for row in csv.DictReader(handle)]
    required = {"subject_id", "center", "reason", "status", "source"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"Exclusion manifest must contain columns {sorted(required)}")
    keys = [canonical_subject_id(row["subject_id"]) for row in rows]
    if len(rows) != 10 or len(set(keys)) != 10:
        raise ValueError("Sensitivity exclusion manifest must contain exactly 10 unique subject IDs")
    for row in rows:
        if row["status"] != EXPECTED_EXCLUSION_STATUS or row["source"] != EXPECTED_EXCLUSION_SOURCE:
            raise ValueError("Exclusion status/source does not identify an unconfirmed post-hoc sensitivity analysis")
        if subject_center(row["subject_id"]) != str(row["center"]).strip().lower():
            raise ValueError(f"Center mismatch in exclusion row: {row}")
    return rows


def build_sensitivity80_cohort(
    patient_index: Mapping[str, Any],
    outer_splits: Sequence[Mapping[str, Any]],
    exclusion_manifest: str | Path,
    *,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Filter the frozen All90 cohort and its existing folds, then audit coverage."""
    if len(patient_index) != 90:
        raise ValueError(f"Sensitivity80 filtering requires the canonical 90-patient input, got {len(patient_index)}")
    rows = read_exclusion_manifest(exclusion_manifest)
    canonical_to_actual: dict[str, str] = {}
    for subject_id in patient_index:
        key = canonical_subject_id(subject_id)
        if key in canonical_to_actual:
            raise ValueError(f"Case-insensitive duplicate subject ID in patient index: {subject_id}")
        canonical_to_actual[key] = str(subject_id)
    requested = [canonical_subject_id(row["subject_id"]) for row in rows]
    missing = sorted(key for key in requested if key not in canonical_to_actual)
    if missing:
        raise ValueError(f"Exclusion subjects not found by exact case-insensitive ID: {missing}")
    excluded = {canonical_to_actual[key] for key in requested}
    retained = {str(subject_id) for subject_id in patient_index} - excluded
    if len(retained) != 80:
        raise ValueError(f"Sensitivity cohort must contain 80 patients, got {len(retained)}")
    center_counts = Counter(subject_center(subject_id) for subject_id in retained)
    if dict(center_counts) != EXPECTED_CENTER_COUNTS:
        raise ValueError(f"Sensitivity80 center counts mismatch: {dict(center_counts)} != {EXPECTED_CENTER_COUNTS}")

    filtered_splits: list[dict[str, Any]] = []
    test_occurrences: Counter[str] = Counter()
    leakage: dict[int, list[str]] = {}
    for split in outer_splits:
        train = [str(value) for value in split["train_subjects"] if str(value) in retained]
        test = [str(value) for value in split["test_subjects"] if str(value) in retained]
        overlap = sorted(set(train) & set(test))
        if overlap:
            leakage[int(split["fold_idx"])] = overlap
        test_occurrences.update(test)
        filtered_splits.append({"fold_idx": int(split["fold_idx"]), "train_subjects": train, "test_subjects": test})
    if leakage:
        raise ValueError(f"Patient leakage after sensitivity filtering: {leakage}")
    wrong_occurrence = {key: count for key, count in test_occurrences.items() if count != 1}
    if set(test_occurrences) != retained or wrong_occurrence:
        raise ValueError("Filtered outer-test folds do not cover every retained subject exactly once")

    filtered_index = {subject_id: patient_index[subject_id] for subject_id in patient_index if subject_id in retained}
    audit = {
        "method": "N8F_CANE_PATH_CP_NEZ_80",
        "analysis_status": "posthoc_sensitivity_not_primary",
        "cohort_status": "posthoc_sensitivity",
        "n_input_patients": 90,
        "n_excluded_patients": 10,
        "n_patients": 80,
        "excluded_subjects": sorted(excluded),
        "retained_subjects": sorted(retained),
        "center_counts": dict(sorted(center_counts.items())),
        "requested_exclusions_all_matched_once": True,
        "outer_test_union_size": len(test_occurrences),
        "outer_test_each_subject_once": True,
        "fold_leakage": {},
        "canonical_old90_manifest_modified": False,
        "exclusions_confirmed_label_errors": False,
        "status": "passed",
    }
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "cane_path_cp_sensitivity80_cohort_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )
        with (destination / "cane_path_cp_sensitivity80_subjects.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["subject_id", "center", "status"])
            writer.writeheader()
            for subject_id in sorted(retained):
                writer.writerow({"subject_id": subject_id, "center": subject_center(subject_id), "status": "retained"})
    return filtered_index, filtered_splits, audit


__all__ = ["EXPECTED_CENTER_COUNTS", "build_sensitivity80_cohort", "read_exclusion_manifest"]
