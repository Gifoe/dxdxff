"""Outcome-blind deterministic split helpers."""

from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from sklearn.model_selection import train_test_split


def split_score(case_id: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_dev_holdout(case_id: str, seed: int, dev_fraction: float = 0.80) -> str:
    if not 0.0 < dev_fraction < 1.0:
        raise ValueError("dev_fraction must be between zero and one")
    return "DEV" if split_score(case_id, seed) < dev_fraction else "G3_HOLDOUT"


def make_g2_train_val(
    case_ids: Iterable[str],
    strata: Iterable[str],
    seed: int,
    train_fraction: float = 0.75,
) -> tuple[list[str], list[str]]:
    ids = np.asarray(list(case_ids), dtype=object)
    labels = np.asarray(list(strata), dtype=object)
    if len(ids) != len(labels) or len(ids) == 0:
        raise ValueError("case_ids and strata must be non-empty and equally sized")
    unique, counts = np.unique(labels, return_counts=True)
    stratify = labels if len(unique) > 1 and int(counts.min()) >= 2 else None
    train, val = train_test_split(
        ids,
        train_size=train_fraction,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    train_ids = sorted(str(value) for value in train)
    val_ids = sorted(str(value) for value in val)
    if set(train_ids) & set(val_ids):
        raise AssertionError("G2 train/val overlap")
    return train_ids, val_ids
