from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd


class Task1ProtocolError(ValueError):
    """Raised when the frozen old-90 protocol is ambiguous or incomplete."""


def freeze_v3_fold_manifest(
    ledger: pd.DataFrame | str | Path,
    *,
    expected_subjects: int = 90,
    expected_folds: tuple[int, ...] | None = (1, 2, 3, 4, 5),
) -> pd.DataFrame:
    table = pd.read_csv(ledger) if isinstance(ledger, (str, Path)) else ledger.copy()
    fold_column = "outer_fold" if "outer_fold" in table else "fold_idx" if "fold_idx" in table else None
    required = {"subject_id", "center"}
    missing = required - set(table)
    if missing or fold_column is None:
        raise Task1ProtocolError(f"V3 ledger is missing required columns: {sorted(missing | ({'fold_idx'} if fold_column is None else set()))}")
    compact = table[["subject_id", "center", fold_column]].drop_duplicates().rename(columns={fold_column: "outer_fold"})
    per_subject = compact.groupby("subject_id", sort=False)["outer_fold"].nunique()
    conflicts = per_subject[per_subject != 1]
    if not conflicts.empty:
        raise Task1ProtocolError(f"V3 ledger assigns patients to multiple folds: {conflicts.index.tolist()[:10]}")
    centers = compact.groupby("subject_id", sort=False)["center"].nunique()
    if (centers != 1).any():
        raise Task1ProtocolError("V3 ledger assigns at least one patient to multiple centers.")
    compact = compact.sort_values("subject_id", kind="stable").reset_index(drop=True)
    if len(compact) != int(expected_subjects):
        raise Task1ProtocolError(f"Expected {expected_subjects} old-90 subjects, found {len(compact)}.")
    folds = tuple(sorted(int(value) for value in compact["outer_fold"].unique()))
    if expected_folds is not None and folds != tuple(expected_folds):
        raise Task1ProtocolError(f"Expected folds {tuple(expected_folds)}, found {folds}.")
    compact.attrs["label_semantics"] = {"NEZ": 1, "EZ": 0}
    return compact


def fold_manifest_hash(manifest: pd.DataFrame) -> str:
    canonical = manifest[["subject_id", "center", "outer_fold"]].sort_values("subject_id").to_csv(index=False, lineterminator="\n")
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _column(table: pd.DataFrame, names: tuple[str, ...], label: str) -> str:
    found = next((name for name in names if name in table.columns), None)
    if found is None:
        raise Task1ProtocolError(f"Manifest is missing {label}; accepted columns: {names}.")
    return found


def load_task1_sensitivity_protocol(
    cohort_manifest: str | Path,
    fold_manifest: str | Path,
    split_manifest: str | Path,
    *,
    expected_subjects: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse supplied P2 manifests without generating any split or fold."""
    cohort_raw = pd.read_csv(cohort_manifest)
    cohort_subject = _column(cohort_raw, ("subject_id", "patient_key"), "subject id")
    cohort_center = _column(cohort_raw, ("center", "source_center"), "center")
    cohort = cohort_raw[[cohort_subject, cohort_center]].rename(columns={cohort_subject: "subject_id", cohort_center: "center"}).drop_duplicates()
    if len(cohort) != expected_subjects or not cohort["subject_id"].is_unique:
        raise Task1ProtocolError(f"Expected exactly {expected_subjects} unique cohort subjects, found {len(cohort)}.")

    folds_raw = pd.read_csv(fold_manifest)
    fold_subject = _column(folds_raw, ("subject_id", "patient_key"), "fold subject id")
    fold_value = _column(folds_raw, ("outer_fold", "fold_idx"), "outer fold")
    folds = folds_raw[[fold_subject, fold_value]].rename(columns={fold_subject: "subject_id", fold_value: "outer_fold"}).drop_duplicates()
    if folds.groupby("subject_id")["outer_fold"].nunique().ne(1).any():
        raise Task1ProtocolError("Fold manifest assigns a subject to multiple outer folds.")
    folds["outer_fold"] = pd.to_numeric(folds["outer_fold"], errors="raise").astype(int)
    if set(folds["subject_id"]) != set(cohort["subject_id"]) or sorted(folds["outer_fold"].unique()) != [1, 2, 3, 4, 5]:
        raise Task1ProtocolError("Fold manifest must exactly cover the cohort with outer folds 1..5.")
    frozen = cohort.merge(folds, on="subject_id", validate="one_to_one").sort_values("subject_id", kind="stable").reset_index(drop=True)

    split_raw = pd.read_csv(split_manifest)
    split_subject = _column(split_raw, ("subject_id", "patient_key"), "split subject id")
    split_fold = _column(split_raw, ("outer_fold", "fold_idx"), "split outer fold")
    split_part = _column(split_raw, ("partition", "split", "role", "split_role"), "split partition")
    aliases = {"fit": "fit", "train": "fit", "validation": "validation", "val": "validation", "test": "test", "outer_test": "test"}
    split = split_raw[[split_subject, split_fold, split_part]].rename(columns={split_subject: "subject_id", split_fold: "outer_fold", split_part: "partition"})
    split["outer_fold"] = pd.to_numeric(split["outer_fold"], errors="raise").astype(int)
    split["partition"] = split["partition"].astype(str).str.strip().str.lower().map(aliases)
    if split["partition"].isna().any():
        raise Task1ProtocolError("Split manifest has unsupported partition values.")
    if not set(split["subject_id"]).issubset(set(frozen["subject_id"])):
        raise Task1ProtocolError("Split manifest includes subjects outside the cohort.")
    for fold in range(1, 6):
        current = split[split["outer_fold"] == fold]
        if set(current["partition"]) != {"fit", "validation", "test"}:
            raise Task1ProtocolError(f"Outer fold {fold} must contain fit, validation, and test partitions.")
        if current["subject_id"].duplicated().any():
            raise Task1ProtocolError(f"Outer fold {fold} assigns a subject to multiple partitions.")
        if set(current["subject_id"]) != set(frozen["subject_id"]):
            raise Task1ProtocolError(f"Outer fold {fold} does not cover every cohort subject exactly once.")
        expected_test = set(frozen.loc[frozen["outer_fold"] == fold, "subject_id"])
        if set(current.loc[current["partition"] == "test", "subject_id"]) != expected_test:
            raise Task1ProtocolError(f"Outer fold {fold} test partition does not match fold manifest.")
    return frozen, split.sort_values(["outer_fold", "partition", "subject_id"], kind="stable").reset_index(drop=True)


__all__ = ["Task1ProtocolError", "fold_manifest_hash", "freeze_v3_fold_manifest", "load_task1_sensitivity_protocol"]
