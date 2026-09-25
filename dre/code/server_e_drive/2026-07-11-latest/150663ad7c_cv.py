from __future__ import annotations

from typing import Iterable

import numpy as np
from sklearn.model_selection import KFold


def inner_subject_splits(subjects: Iterable[str], n_splits: int, seed: int = 42) -> list[tuple[set[str], set[str]]]:
    unique = np.array(sorted(set(map(str, subjects))))
    if len(unique) < 2:
        raise ValueError("at least two train subjects are required for inner CV")
    k = min(int(n_splits), len(unique))
    result: list[tuple[set[str], set[str]]] = []
    for train_idx, test_idx in KFold(n_splits=k, shuffle=True, random_state=seed).split(unique):
        result.append((set(unique[train_idx].tolist()), set(unique[test_idx].tolist())))
    return result


def outer_subject_folds(ledger) -> list[tuple[int, set[str], set[str]]]:
    folds = sorted(int(item) for item in ledger["outer_fold"].unique())
    output = []
    for fold in folds:
        test = set(ledger.loc[ledger["outer_fold"].astype(int) == fold, "subject_id"].astype(str))
        train = set(ledger.loc[ledger["outer_fold"].astype(int) != fold, "subject_id"].astype(str))
        output.append((fold, train, test))
    return output
