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


__all__ = ["Task1ProtocolError", "fold_manifest_hash", "freeze_v3_fold_manifest"]
