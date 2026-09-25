from __future__ import annotations

from typing import Callable

import pandas as pd

from outcome_hifos.folds import build_inner_ledgers, iter_outer_partitions


def run_inner_crossfit(
    outer_train_manifest: pd.DataFrame,
    *,
    n_splits: int,
    seed: int,
    fit_predict: Callable[[tuple[str, ...], tuple[str, ...], int], pd.DataFrame],
) -> pd.DataFrame:
    ledger = build_inner_ledgers(outer_train_manifest, n_splits=n_splits, seed=seed)
    rows: list[pd.DataFrame] = []
    for partition in iter_outer_partitions(outer_train_manifest, ledger):
        if set(partition.train_subjects) & set(partition.test_subjects):
            raise ValueError(f"Inner fold {partition.fold_idx} has patient overlap.")
        prediction = fit_predict(partition.train_subjects, partition.test_subjects, partition.fold_idx).copy()
        if "subject_id" not in prediction:
            raise ValueError("Inner fit_predict output is missing subject_id.")
        predicted_subjects = prediction["subject_id"].astype(str).tolist()
        if sorted(predicted_subjects) != sorted(partition.test_subjects):
            raise ValueError(
                f"Inner fold {partition.fold_idx} predictions do not exactly match validation subjects."
            )
        prediction["subject_id"] = prediction["subject_id"].astype(str)
        prediction["inner_fold_idx"] = int(partition.fold_idx)
        prediction["role"] = "inner_oof"
        rows.append(prediction)
    combined = pd.concat(rows, ignore_index=True).sort_values("subject_id", kind="stable").reset_index(drop=True)
    if combined["subject_id"].duplicated().any():
        raise ValueError("Inner cross-fit produced duplicate patient predictions.")
    if set(combined["subject_id"]) != set(outer_train_manifest["subject_id"].astype(str)):
        raise ValueError("Inner cross-fit did not cover every outer-train patient exactly once.")
    return combined


__all__ = ["run_inner_crossfit"]
