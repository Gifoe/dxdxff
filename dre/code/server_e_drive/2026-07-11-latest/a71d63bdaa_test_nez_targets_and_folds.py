from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TARGETS = ROOT / "reference" / "latent_core_targets_nez_s5_8"


def test_nez_target_semantics_and_all90() -> None:
    frame = pd.read_csv(TARGETS / "latent_core_targets_all_oof.csv")
    assert frame["subject_id"].nunique() == 90
    assert sorted(frame["fold_id"].astype(int).unique().tolist()) == [1, 2, 3, 4, 5]
    assert (frame["label_nez"] == 1.0 - frame["label_ez"]).all()
    assert (frame.loc[frame["label_ez"] > 0.5, "pseudo_core_q"] == 0.0).all()
    assert (frame.loc[frame["label_nez"] > 0.5, "pseudo_core_q"] > 0.0).any()
    assert np.allclose(frame["teacher_nez_score"], 1.0 - frame["a9v3_oof_score"], atol=1e-12)


def test_fold_targets_exclude_heldout_fold() -> None:
    for fold_id in range(1, 6):
        frame = pd.read_csv(TARGETS / f"latent_core_targets_fold{fold_id}_train.csv")
        assert not (frame["fold_id"].astype(int) == fold_id).any()
