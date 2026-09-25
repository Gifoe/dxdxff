from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd


class ProtocolError(ValueError):
    pass


def manifest_patient_keys(path: str | Path) -> set[str]:
    frame = pd.read_csv(Path(path).expanduser())
    patient_column = next((c for c in ("patient_key", "patient_id", "subject_id") if c in frame), None)
    if patient_column is None:
        raise ProtocolError("Fold manifest requires patient_key/patient_id/subject_id")
    if "partition" in frame:
        frame = frame[frame["partition"].astype(str).str.lower() == "test"]
    return set(frame[patient_column].astype(str))


def manifest_hash(frame: pd.DataFrame, columns: Sequence[str] = ("patient_key", "outer_fold")) -> str:
    normalized = frame.loc[:, list(columns)].sort_values(list(columns), kind="stable")
    return hashlib.sha256(normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def load_fold_manifest(path: str | Path, outcomes: pd.DataFrame, *, strict: bool = True) -> pd.DataFrame:
    frame = pd.read_csv(Path(path).expanduser())
    patient_column = next((c for c in ("patient_key", "patient_id", "subject_id") if c in frame), None)
    fold_column = next((c for c in ("outer_fold", "fold_idx") if c in frame), None)
    if patient_column is None or fold_column is None:
        raise ProtocolError("Fold manifest requires patient_key and outer_fold/fold_idx")
    if "partition" in frame:
        frame = frame[frame["partition"].astype(str).str.lower() == "test"]
    result = frame[[patient_column, fold_column]].rename(columns={patient_column: "patient_key", fold_column: "outer_fold"}).copy()
    result["patient_key"] = result["patient_key"].astype(str)
    result["outer_fold"] = pd.to_numeric(result["outer_fold"], errors="raise").astype(int)
    if result.empty:
        raise ProtocolError("Fold manifest has no outer-test patients")
    duplicate = result["patient_key"].duplicated(keep=False)
    if duplicate.any():
        raise ProtocolError(f"Each patient must occur in exactly one outer-test fold: {sorted(result.loc[duplicate, 'patient_key'].unique())[:20]}")
    eligible_frame = outcomes[outcomes["outcome_group"].isin(["success", "failure"])].copy()
    if eligible_frame["patient_key"].astype(str).duplicated().any():
        raise ProtocolError("Outcome cohort contains duplicate patient keys")
    labels = set(pd.to_numeric(eligible_frame["outcome_label"], errors="coerce").dropna().astype(int))
    if not labels or not labels.issubset({0, 1}):
        raise ProtocolError(f"Outcome labels must contain only 0 or 1; found {sorted(labels)}")
    eligible = set(eligible_frame["patient_key"].astype(str))
    manifest_subjects = set(result["patient_key"])
    missing, extra = sorted(eligible - manifest_subjects), sorted(manifest_subjects - eligible)
    if strict and (missing or extra):
        raise ProtocolError(f"Fold/outcome cohort mismatch; missing={missing[:10]}, extra={extra[:10]}")
    result = result[result["patient_key"].isin(eligible)].copy()
    counts = result.groupby("outer_fold")["patient_key"].size()
    if strict and (len(counts) != 5 or bool((counts <= 0).any())):
        raise ProtocolError(f"Strict outer_cv requires exactly five non-empty outer folds; found {counts.to_dict()}")
    return result.sort_values(["outer_fold", "patient_key"], kind="stable").reset_index(drop=True)


def checkpoint_training_subjects(manifest_path: str | Path, fold: int) -> set[str]:
    frame = pd.read_csv(Path(manifest_path).expanduser())
    patient = next((c for c in ("patient_key", "patient_id", "subject_id") if c in frame), None)
    fold_column = next((c for c in ("outer_fold", "fold_idx") if c in frame), None)
    if patient is None or fold_column is None:
        raise ProtocolError("P2 training manifest requires patient key and outer fold")
    current = frame[frame[fold_column].astype(int) == int(fold)]
    if "partition" in current:
        current = current[current["partition"].astype(str).str.lower().isin(["train", "fit"])]
        return set(current[patient].astype(str))
    all_subjects = set(frame[patient].astype(str))
    test = set(current[patient].astype(str))
    return all_subjects - test


def assert_checkpoint_safe(training_subjects: Iterable[str], test_subjects: Iterable[str]) -> None:
    overlap = sorted(set(map(str, training_subjects)) & set(map(str, test_subjects)))
    if overlap:
        raise ProtocolError(f"P2 checkpoint training subjects overlap outer-test: {overlap[:20]}")


__all__ = ["ProtocolError", "assert_checkpoint_safe", "checkpoint_training_subjects", "load_fold_manifest", "manifest_hash", "manifest_patient_keys"]
