"""Frozen Task 1 cohort and outer-fold contracts for RTC_SHIFT.

The cache may contain more patients than a formal analysis.  This module
verifies that the requested cohort is exactly the frozen All90 ledger (or its
predeclared sensitivity subset) and records a stable fold-ledger hash.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from .cane_path_cohort import canonical_subject_id, read_exclusion_manifest, subject_center


COHORTS = ("sensitivity80", "primary90")


def _sha256_lines(values: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(map(str, values)).encode("utf-8")).hexdigest()


def read_all90_ledger(path: str | Path) -> list[str]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"All90 allow-list not found: {source}")
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "subject_id" not in rows[0]:
        raise ValueError("All90 allow-list must contain a subject_id column")
    values = [str(row["subject_id"]).strip() for row in rows if str(row.get("subject_id", "")).strip()]
    keys = [canonical_subject_id(value) for value in values]
    if len(values) != 90 or len(set(keys)) != 90:
        raise ValueError("All90 allow-list must contain exactly 90 unique subject IDs")
    return values


def _fold_rows(outer_splits: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split in outer_splits:
        fold = int(split["fold_idx"])
        for partition, subjects in (("train", split["train_subjects"]), ("test", split["test_subjects"])):
            for subject_id in sorted(map(str, subjects)):
                rows.append({"fold_idx": fold, "partition": partition, "subject_id": subject_id})
    return rows


def validate_task1_cohort(
    patient_index: Mapping[str, Any],
    outer_splits: Sequence[Mapping[str, Any]],
    *,
    cohort_name: str,
    all90_ledger_path: str | Path,
    exclusion_path: str | Path | None,
    cache_subject_count: int | None = None,
) -> dict[str, Any]:
    """Fail closed unless the active patients and five folds match the contract."""
    cohort = str(cohort_name).lower()
    if cohort not in COHORTS:
        raise ValueError(f"Unsupported RTC cohort: {cohort_name!r}")
    all90 = read_all90_ledger(all90_ledger_path)
    all90_map = {canonical_subject_id(value): value for value in all90}
    active_map = {canonical_subject_id(str(value)): str(value) for value in patient_index}
    if len(active_map) != len(patient_index):
        raise ValueError("Active patient index has duplicate case-insensitive subject IDs")
    if cohort == "primary90":
        expected_keys = set(all90_map)
        expected_exclusions: list[str] = []
        expected_count = 90
        if exclusion_path and str(exclusion_path).strip():
            raise ValueError("primary90 must not apply the sensitivity80 exclusion manifest")
    else:
        if not exclusion_path:
            raise ValueError("sensitivity80 requires the predeclared ten-patient exclusion manifest")
        excluded_rows = read_exclusion_manifest(exclusion_path)
        excluded = [canonical_subject_id(row["subject_id"]) for row in excluded_rows]
        if not set(excluded).issubset(all90_map):
            raise ValueError("Sensitivity exclusions must all belong to the frozen All90 ledger")
        expected_keys = set(all90_map) - set(excluded)
        expected_exclusions = [all90_map[key] for key in excluded]
        expected_count = 80
    active_keys = set(active_map)
    missing = sorted(expected_keys - active_keys)
    unexpected = sorted(active_keys - expected_keys)
    if missing or unexpected or len(patient_index) != expected_count:
        raise ValueError(
            f"{cohort} cohort mismatch: expected={expected_count}, active={len(patient_index)}, "
            f"missing={missing}, unexpected={unexpected}"
        )
    if len(outer_splits) != 5:
        raise ValueError(f"{cohort} requires exactly five outer folds")
    occurrences: dict[str, int] = {key: 0 for key in expected_keys}
    fold_errors: list[str] = []
    for split in outer_splits:
        train = {canonical_subject_id(value) for value in split["train_subjects"]}
        test = {canonical_subject_id(value) for value in split["test_subjects"]}
        if train & test:
            fold_errors.append(f"fold {split['fold_idx']} has train/test overlap")
        if (train | test) != expected_keys:
            fold_errors.append(f"fold {split['fold_idx']} does not cover the active cohort")
        for subject in test:
            occurrences[subject] = occurrences.get(subject, 0) + 1
    duplicate_test = sorted(key for key, count in occurrences.items() if count != 1)
    if fold_errors or duplicate_test:
        raise ValueError(f"{cohort} outer-fold contract failed: {fold_errors}; test occurrence errors={duplicate_test}")
    ordered_active = [all90_map[key] for key in (canonical_subject_id(value) for value in all90) if key in expected_keys]
    rows = _fold_rows(outer_splits)
    fold_hash = _sha256_lines([f"{row['fold_idx']}|{row['partition']}|{row['subject_id']}" for row in rows])
    return {
        "cohort_name": cohort,
        "n_patients": expected_count,
        "ordered_subject_ids": ordered_active,
        "subject_set_hash": _sha256_lines(sorted(expected_keys)),
        "allow_list_path": str(Path(all90_ledger_path)),
        "exclusion_path": str(Path(exclusion_path)) if cohort == "sensitivity80" else "",
        "excluded_subject_ids": sorted(expected_exclusions),
        "cache_subject_count": int(cache_subject_count if cache_subject_count is not None else len(patient_index)),
        "missing_allowed_subjects": missing,
        "unexpected_selected_subjects": unexpected,
        "outer_fold_ledger_hash": fold_hash,
        "outer_test_each_subject_once": True,
        "outer_train_test_disjoint": True,
        "status": "passed",
    }


def write_task1_cohort_contract(output_dir: str | Path, audit: Mapping[str, Any], outer_splits: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "cohort_audit.json").write_text(json.dumps(dict(audit), indent=2, ensure_ascii=False), encoding="utf-8")
    with (destination / "cohort_subjects.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id", "center", "cohort_name"])
        writer.writeheader()
        for subject_id in audit["ordered_subject_ids"]:
            writer.writerow({"subject_id": subject_id, "center": subject_center(subject_id), "cohort_name": audit["cohort_name"]})
    rows = _fold_rows(outer_splits)
    with (destination / "outer_fold_ledger.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["fold_idx", "partition", "subject_id"])
        writer.writeheader(); writer.writerows(rows)


__all__ = ["COHORTS", "read_all90_ledger", "validate_task1_cohort", "write_task1_cohort_contract"]
