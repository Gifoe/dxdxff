from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from outcome_hifos.calibration import ProtocolLeakageError
from outcome_hifos.training.inner_cv import run_inner_crossfit
from outcome_hifos.training.late_fusion import fit_cross_fitted_stacker
from outcome_hifos.training.outer_cv import build_loco_partitions, select_best_candidate


def _branch(prefix: str) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"p{i}" for i in range(8)],
            "role": ["inner_oof"] * 8,
            "outcome": [0, 0, 0, 0, 1, 1, 1, 1],
            "outer_fold_idx": [2] * 8,
            "inner_fold_idx": [index % 4 for index in range(8)],
            "seed": [42] * 8,
            "probability": [0.05, 0.2, 0.3, 0.45, 0.55, 0.7, 0.8, 0.95] if prefix == "feature" else [0.2, 0.4, 0.1, 0.35, 0.6, 0.9, 0.65, 0.85],
            "raw_logit": np.linspace(-2.0, 2.0, 8),
            "input_representation": ["raw_logit"] * 8,
        }
    )


def test_stacker_rejects_outer_rows_in_fit_table() -> None:
    contaminated = _branch("feature")
    contaminated.loc[0, "role"] = "outer_test"
    outer = pd.DataFrame({"subject_id": ["z1"], "feature_raw_logit": [-0.4], "fm_raw_logit": [0.7], "feature_input_representation": ["raw_logit"], "fm_input_representation": ["raw_logit"], "outer_fold_idx": [2], "seed": [42], "role": ["outer_test"]})
    with pytest.raises(ProtocolLeakageError, match="outer_test"):
        fit_cross_fitted_stacker(contaminated, _branch("fm"), outer)


def test_stacker_aligns_inner_oof_and_predicts_outer_once() -> None:
    outer = pd.DataFrame(
        {
            "subject_id": ["z1", "z2"],
            "feature_raw_logit": [-1.4, 1.4],
            "fm_raw_logit": [-0.8, 0.8],
            "feature_input_representation": ["raw_logit", "raw_logit"],
            "fm_input_representation": ["raw_logit", "raw_logit"],
            "outer_fold_idx": [2, 2],
            "seed": [42, 42],
            "role": ["outer_test", "outer_test"],
        }
    )
    result = fit_cross_fitted_stacker(_branch("feature"), _branch("fm").sample(frac=1.0, random_state=3), outer)
    assert result.inner_oof["subject_id"].tolist() == [f"p{i}" for i in range(8)]
    assert result.outer_predictions["subject_id"].tolist() == ["z1", "z2"]
    assert np.all((result.outer_predictions["fusion_probability"] >= 0.0) & (result.outer_predictions["fusion_probability"] <= 1.0))
    assert set(result.coefficients) == {"intercept", "feature", "fm"}


def test_inner_crossfit_never_trains_on_validation_subject() -> None:
    manifest = pd.DataFrame(
        {
            "subject_id": [f"p{i}" for i in range(8)],
            "center": [f"c{i % 2}" for i in range(8)],
            "outcome_label": [i % 2 for i in range(8)],
        }
    )
    calls: list[tuple[set[str], set[str]]] = []

    def callback(train_subjects: tuple[str, ...], validation_subjects: tuple[str, ...], fold_idx: int) -> pd.DataFrame:
        calls.append((set(train_subjects), set(validation_subjects)))
        outcomes = manifest.set_index("subject_id")["outcome_label"]
        return pd.DataFrame(
            {
                "subject_id": validation_subjects,
                "outcome": [int(outcomes.loc[subject]) for subject in validation_subjects],
                "probability": [0.8 if int(outcomes.loc[subject]) else 0.2 for subject in validation_subjects],
                "logit": [1.4 if int(outcomes.loc[subject]) else -1.4 for subject in validation_subjects],
            }
        )

    oof = run_inner_crossfit(manifest, n_splits=4, seed=42, fit_predict=callback)
    assert set(oof["subject_id"]) == set(manifest["subject_id"])
    assert set(oof["role"]) == {"inner_oof"}
    assert all(not train & validation for train, validation in calls)


def test_model_selection_uses_declared_tie_breaks() -> None:
    candidates = pd.DataFrame(
        [
            {"variant": "a", "macro_f1": 0.7, "auroc": 0.75, "balanced_accuracy": 0.7, "brier": 0.2, "parameter_count": 100},
            {"variant": "b", "macro_f1": 0.7, "auroc": 0.80, "balanced_accuracy": 0.6, "brier": 0.3, "parameter_count": 50},
        ]
    )
    assert select_best_candidate(candidates)["variant"] == "b"


def test_loco_partitions_hold_out_exactly_one_center() -> None:
    manifest = pd.DataFrame(
        {
            "subject_id": ["a", "b", "c", "d"],
            "center": ["c1", "c1", "c2", "c2"],
            "outcome_label": [0, 1, 0, 1],
        }
    )
    partitions = build_loco_partitions(manifest)
    assert [partition.held_out_center for partition in partitions] == ["c1", "c2"]
    assert all(set(partition.test_manifest["center"]) == {partition.held_out_center} for partition in partitions)
