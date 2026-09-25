from __future__ import annotations

import hashlib
from collections import Counter
from dataclasses import dataclass
from typing import Iterator

import pandas as pd


class FoldProtocolError(ValueError):
    """Raised when a patient-wise fold ledger cannot be constructed safely."""


@dataclass(frozen=True)
class OuterPartition:
    fold_idx: int
    train_subjects: tuple[str, ...]
    test_subjects: tuple[str, ...]
    train_manifest: pd.DataFrame
    test_manifest: pd.DataFrame


def _validate_manifest(manifest: pd.DataFrame, n_splits: int) -> pd.DataFrame:
    required = {"subject_id", "center", "outcome_label"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise FoldProtocolError(f"Patient manifest is missing columns: {missing}")
    frame = manifest.loc[:, ["subject_id", "center", "outcome_label"]].copy()
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["center"] = frame["center"].fillna("unknown").astype(str)
    if frame["subject_id"].duplicated().any():
        duplicates = sorted(frame.loc[frame["subject_id"].duplicated(keep=False), "subject_id"].unique().tolist())
        raise FoldProtocolError(f"Patient manifest contains duplicate subject rows: {duplicates[:20]}")
    if frame["outcome_label"].isna().any() or not set(frame["outcome_label"].astype(int).unique()).issubset({0, 1}):
        raise FoldProtocolError("Patient manifest outcome_label must contain only binary 0/1 values.")
    frame["outcome_label"] = frame["outcome_label"].astype(int)
    if int(n_splits) < 2:
        raise FoldProtocolError("n_splits must be at least 2.")
    if len(frame) < int(n_splits):
        raise FoldProtocolError(f"Cannot create {n_splits} folds from {len(frame)} patients.")
    return frame.sort_values("subject_id", kind="stable").reset_index(drop=True)


def _stable_tie(seed: int, subject_id: str) -> str:
    return hashlib.sha256(f"{int(seed)}|{subject_id}".encode("utf-8")).hexdigest()


def _assign_strata(frame: pd.DataFrame, n_splits: int) -> pd.Series:
    composite = frame["center"] + "|" + frame["outcome_label"].astype(str)
    composite_counts = composite.value_counts().to_dict()
    center_counts = frame["center"].value_counts().to_dict()
    outcome_counts = frame["outcome_label"].value_counts().to_dict()
    strata: list[str] = []
    for index, row in frame.iterrows():
        comp = composite.iloc[index]
        center = str(row["center"])
        outcome = int(row["outcome_label"])
        if int(composite_counts.get(comp, 0)) >= n_splits:
            strata.append(f"center_outcome:{comp}")
        elif int(center_counts.get(center, 0)) >= n_splits:
            strata.append(f"center:{center}")
        elif int(outcome_counts.get(outcome, 0)) >= n_splits:
            strata.append(f"outcome:{outcome}")
        else:
            strata.append("all")
    return pd.Series(strata, index=frame.index, dtype="object")


def build_composite_fold_ledger(
    manifest: pd.DataFrame,
    n_splits: int = 5,
    seed: int = 42,
    cohort: str = "feature",
) -> pd.DataFrame:
    frame = _validate_manifest(manifest, int(n_splits))
    frame["stratum"] = _assign_strata(frame, int(n_splits))
    stratum_totals = Counter(frame["stratum"].tolist())
    ordered = frame.assign(
        _rarity=frame["stratum"].map(stratum_totals),
        _tie=frame["subject_id"].map(lambda value: _stable_tie(seed, value)),
    ).sort_values(["_rarity", "_tie", "subject_id"], kind="stable")

    fold_sizes = Counter()
    fold_strata: dict[int, Counter] = {fold: Counter() for fold in range(1, int(n_splits) + 1)}
    fold_centers: dict[int, Counter] = {fold: Counter() for fold in range(1, int(n_splits) + 1)}
    fold_outcomes: dict[int, Counter] = {fold: Counter() for fold in range(1, int(n_splits) + 1)}
    assignments: dict[str, int] = {}
    for row in ordered.itertuples(index=False):
        scores: list[tuple[float, int]] = []
        for fold in range(1, int(n_splits) + 1):
            size_penalty = float((fold_sizes[fold] + 1) ** 2)
            stratum_penalty = float((fold_strata[fold][row.stratum] + 1) ** 2) / max(stratum_totals[row.stratum], 1)
            center_penalty = float((fold_centers[fold][row.center] + 1) ** 2)
            outcome_penalty = float((fold_outcomes[fold][int(row.outcome_label)] + 1) ** 2)
            scores.append((2.0 * size_penalty + 3.0 * stratum_penalty + 0.5 * center_penalty + 0.5 * outcome_penalty, fold))
        _, selected_fold = min(scores, key=lambda value: (value[0], value[1]))
        assignments[str(row.subject_id)] = selected_fold
        fold_sizes[selected_fold] += 1
        fold_strata[selected_fold][str(row.stratum)] += 1
        fold_centers[selected_fold][str(row.center)] += 1
        fold_outcomes[selected_fold][int(row.outcome_label)] += 1

    result = frame.copy()
    result["fold_idx"] = result["subject_id"].map(assignments).astype(int)
    result["cohort"] = str(cohort)
    result["split_seed"] = int(seed)
    result["n_splits"] = int(n_splits)
    return result.loc[:, ["subject_id", "center", "outcome_label", "stratum", "fold_idx", "cohort", "split_seed", "n_splits"]].sort_values("subject_id", kind="stable").reset_index(drop=True)


def ledger_hash(ledger: pd.DataFrame) -> str:
    columns = ["subject_id", "center", "outcome_label", "stratum", "fold_idx", "cohort", "split_seed", "n_splits"]
    normalized = ledger.loc[:, columns].sort_values("subject_id", kind="stable").reset_index(drop=True)
    return hashlib.sha256(normalized.to_csv(index=False, lineterminator="\n").encode("utf-8")).hexdigest()


def iter_outer_partitions(manifest: pd.DataFrame, ledger: pd.DataFrame) -> Iterator[OuterPartition]:
    if set(manifest["subject_id"].astype(str)) != set(ledger["subject_id"].astype(str)):
        raise FoldProtocolError("Fold ledger subjects do not exactly match the patient manifest.")
    indexed = manifest.copy()
    indexed["subject_id"] = indexed["subject_id"].astype(str)
    for fold_idx in sorted(ledger["fold_idx"].unique().tolist()):
        test_subjects = tuple(sorted(ledger.loc[ledger["fold_idx"] == fold_idx, "subject_id"].astype(str).tolist()))
        train_subjects = tuple(sorted(set(indexed["subject_id"]) - set(test_subjects)))
        train_manifest = indexed[indexed["subject_id"].isin(train_subjects)].copy().reset_index(drop=True)
        test_manifest = indexed[indexed["subject_id"].isin(test_subjects)].copy().reset_index(drop=True)
        if set(train_subjects) & set(test_subjects):
            raise FoldProtocolError(f"Fold {fold_idx} contains overlapping train/test patients.")
        yield OuterPartition(int(fold_idx), train_subjects, test_subjects, train_manifest, test_manifest)


def build_inner_ledgers(outer_train_manifest: pd.DataFrame, n_splits: int = 4, seed: int = 42) -> pd.DataFrame:
    return build_composite_fold_ledger(outer_train_manifest, n_splits=n_splits, seed=seed, cohort="inner")


__all__ = [
    "FoldProtocolError",
    "OuterPartition",
    "build_composite_fold_ledger",
    "build_inner_ledgers",
    "iter_outer_partitions",
    "ledger_hash",
]
