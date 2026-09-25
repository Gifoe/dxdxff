from __future__ import annotations

import pandas as pd
import pytest

from outcome_hifos.constants import MAIN_VARIANTS
from outcome_hifos.folds import (
    FoldProtocolError,
    build_composite_fold_ledger,
    build_inner_ledgers,
    iter_outer_partitions,
    ledger_hash,
)


def _manifest() -> pd.DataFrame:
    rows = []
    for index in range(20):
        rows.append(
            {
                "subject_id": f"p{index:02d}",
                "center": f"c{index % 2}",
                "outcome_label": index % 2,
            }
        )
    return pd.DataFrame(rows)


def test_composite_fold_ledger_is_deterministic_unique_and_balanced() -> None:
    first = build_composite_fold_ledger(_manifest(), n_splits=5, seed=42, cohort="feature")
    second = build_composite_fold_ledger(_manifest().sample(frac=1.0, random_state=9), n_splits=5, seed=42, cohort="feature")
    pd.testing.assert_frame_equal(first, second)
    assert first["subject_id"].is_unique
    assert sorted(first["fold_idx"].unique().tolist()) == [1, 2, 3, 4, 5]
    assert first.groupby("fold_idx").size().max() - first.groupby("fold_idx").size().min() <= 1


def test_all_variants_reuse_same_frozen_ledger_hash() -> None:
    ledger = build_composite_fold_ledger(_manifest(), n_splits=5, seed=42, cohort="feature")
    hashes = {variant: ledger_hash(ledger) for variant in MAIN_VARIANTS}
    assert len(set(hashes.values())) == 1


def test_outer_test_never_appears_in_inner_ledger() -> None:
    manifest = _manifest()
    ledger = build_composite_fold_ledger(manifest, n_splits=5, seed=42, cohort="feature")
    for partition in iter_outer_partitions(manifest, ledger):
        inner = build_inner_ledgers(partition.train_manifest, n_splits=4, seed=42)
        assert not set(partition.test_subjects) & set(inner["subject_id"])
        assert set(inner["subject_id"]) == set(partition.train_subjects)


def test_duplicate_subject_manifest_fails_closed() -> None:
    manifest = pd.concat([_manifest(), _manifest().iloc[[0]]], ignore_index=True)
    with pytest.raises(FoldProtocolError, match="duplicate subject"):
        build_composite_fold_ledger(manifest, n_splits=5, seed=42, cohort="feature")
