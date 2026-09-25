from __future__ import annotations

import numpy as np
import pandas as pd

from outcome_hifos.dataset import OutcomePatientExample
from outcome_hifos.folds import build_composite_fold_ledger
from outcome_hifos.baselines.training import run_hierarchical_token_oof


def _examples() -> list[OutcomePatientExample]:
    rng = np.random.default_rng(8)
    examples = []
    for index in range(10):
        target = index % 2
        examples.append(
            OutcomePatientExample(
                subject_id=f"p{index:02d}", center=f"c{index % 2}", target=float(target), canonical_channels=("A1", "A2"),
                model_input={"feature_runs": [rng.normal(target, 0.1, (2, 2, 4)).astype(np.float32)], "window_centers": [np.asarray([-1.0, 1.0])], "seizure_channel_mask": [np.asarray([True, True])], "canonical_index": np.arange(2)},
                side_metadata={},
            )
        )
    return examples


def test_fm_hierarchical_runner_uses_full_inner_oof_and_one_outer_row_per_patient(tmp_path) -> None:
    examples = _examples()
    manifest = pd.DataFrame([{"subject_id": item.subject_id, "center": item.center, "outcome_label": int(item.target)} for item in examples])
    folds = build_composite_fold_ledger(manifest, n_splits=2, seed=42, cohort="task2")
    oof = run_hierarchical_token_oof(
        examples, folds, variant="T2_CBRAMOD_FROZEN_HIERPOOL", seed=42, output_dir=tmp_path,
        device="cpu", inner_folds=2, epochs=1, max_outer_folds=1, model_config={"model_dim": 4, "dropout": 0.0},
    )
    assert oof["subject_id"].is_unique
    inner = pd.read_csv(next(tmp_path.rglob("inner_oof_predictions.csv")))
    assert inner["subject_id"].nunique() == 5
    assert set(inner["inner_fold_idx"]) == {1, 2}
