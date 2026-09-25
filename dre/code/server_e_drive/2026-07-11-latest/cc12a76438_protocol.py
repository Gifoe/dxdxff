from __future__ import annotations

import hashlib
from typing import Iterable

import pandas as pd


def build_task2_primary_cohort(outcome_manifest: pd.DataFrame, *, old90_subjects: Iterable[str]) -> tuple[pd.DataFrame, dict[str, int | str]]:
    required = {"subject_id", "center", "outcome_group", "outcome_label"}
    missing = required - set(outcome_manifest)
    if missing:
        raise ValueError(f"Outcome manifest is missing columns: {sorted(missing)}")
    if outcome_manifest["subject_id"].duplicated().any():
        raise ValueError("Outcome manifest contains duplicate subjects.")
    old90 = {str(value) for value in old90_subjects}
    success = outcome_manifest[(outcome_manifest["outcome_group"] == "success") & outcome_manifest["subject_id"].isin(old90)].copy()
    missing_old90 = sorted(old90 - set(success["subject_id"]))
    if missing_old90:
        raise ValueError(f"Old-90 subjects missing or not resolved as success: {missing_old90[:20]}")
    failures = outcome_manifest[outcome_manifest["outcome_group"] == "failure"].copy()
    cohort = pd.concat([success, failures], ignore_index=True).sort_values("subject_id", kind="stable").reset_index(drop=True)
    cohort["outcome_label"] = cohort["outcome_label"].astype(int)
    canonical = cohort[["subject_id", "center", "outcome_label"]].to_csv(index=False, lineterminator="\n")
    audit: dict[str, int | str] = {
        "old90_success": len(success),
        "failure": len(failures),
        "extra_success_excluded": int(((outcome_manifest["outcome_group"] == "success") & ~outcome_manifest["subject_id"].isin(old90)).sum()),
        "unknown_excluded": int((outcome_manifest["outcome_group"] == "unknown").sum()),
        "conflict_excluded": int((outcome_manifest["outcome_group"] == "conflict").sum()),
        "cohort_size": len(cohort),
        "cohort_hash": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    return cohort, audit


__all__ = ["build_task2_primary_cohort"]
