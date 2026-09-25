from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LOCOPartition:
    held_out_center: str
    train_manifest: pd.DataFrame
    test_manifest: pd.DataFrame


def build_loco_partitions(manifest: pd.DataFrame) -> list[LOCOPartition]:
    required = {"subject_id", "center", "outcome_label"}
    if not required.issubset(manifest.columns):
        raise ValueError(f"LOCO manifest is missing columns: {sorted(required - set(manifest.columns))}")
    output = []
    for center in sorted(manifest["center"].astype(str).unique().tolist()):
        test = manifest[manifest["center"].astype(str) == center].copy().reset_index(drop=True)
        train = manifest[manifest["center"].astype(str) != center].copy().reset_index(drop=True)
        if train.empty or test.empty:
            raise ValueError(f"LOCO center {center!r} has an empty train or test partition.")
        output.append(LOCOPartition(center, train, test))
    return output


def select_best_candidate(candidates: pd.DataFrame) -> dict[str, object]:
    required = {"variant", "macro_f1", "auroc", "balanced_accuracy", "brier", "parameter_count"}
    if not required.issubset(candidates.columns) or candidates.empty:
        raise ValueError(f"Candidate table must be non-empty with columns: {sorted(required)}")
    ranked = candidates.copy()
    ranked["_auroc"] = ranked["auroc"].fillna(-np.inf)
    ranked["_balanced"] = ranked["balanced_accuracy"].fillna(-np.inf)
    row = ranked.sort_values(
        ["macro_f1", "_auroc", "_balanced", "brier", "parameter_count", "variant"],
        ascending=[False, False, False, True, True, True],
        kind="stable",
    ).iloc[0]
    return {key: row[key] for key in candidates.columns}


def select_best_profile(candidates: pd.DataFrame) -> dict[str, object]:
    required = {"candidate_profile", "macro_f1", "auroc", "brier", "parameter_count"}
    if candidates.empty or not required.issubset(candidates.columns):
        raise ValueError(f"Candidate profile table must be non-empty with columns: {sorted(required)}")
    if "role" in candidates and set(candidates["role"].astype(str)) != {"inner_oof"}:
        raise ValueError("Candidate selection accepts only inner_oof metrics.")
    ranked = candidates.copy()
    ranked["_auroc"] = ranked["auroc"].fillna(-np.inf)
    row = ranked.sort_values(
        ["macro_f1", "_auroc", "brier", "parameter_count", "candidate_profile"],
        ascending=[False, False, True, True, True],
        kind="stable",
    ).iloc[0]
    return {key: row[key] for key in candidates.columns}


__all__ = ["LOCOPartition", "build_loco_partitions", "select_best_candidate", "select_best_profile"]
