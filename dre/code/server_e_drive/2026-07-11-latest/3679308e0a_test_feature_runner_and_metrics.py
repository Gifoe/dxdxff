from __future__ import annotations

import numpy as np
import pandas as pd

from task1_baselines.metrics import compute_task1_metrics
from task1_baselines.thresholds import select_patient_macro_threshold
from task1_baselines.training.feature_runner import run_feature_oof


def _table() -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    folds = []
    for index in range(20):
        subject = f"p{index:02d}"
        fold = index % 5 + 1
        folds.append({"subject_id": subject, "center": f"c{index % 2}", "outer_fold": fold})
        for channel in range(4):
            label = channel % 2
            rows.append(
                {
                    "subject_id": subject,
                    "center": f"c{index % 2}",
                    "channel_name": f"A{channel}",
                    "label_nez": label,
                    "x": float(label * 4 + index / 100),
                    "z": float(channel),
                    "valid_seizure_count": 1,
                    "valid_window_count": 2,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(folds)


def test_threshold_optimizes_patient_macro_not_pooled_channels() -> None:
    rows = pd.DataFrame(
        {
            "subject_id": ["p1", "p1", "p1", "p1", "p2", "p2"],
            "label_nez": [0, 0, 0, 1, 0, 1],
            "score_nez_probability": [0.1, 0.2, 0.3, 0.6, 0.45, 0.55],
        }
    )
    selected = select_patient_macro_threshold(rows)
    assert 0.05 <= selected.threshold <= 0.95
    assert selected.source == "inner_oof_patient_macro_f1"


def test_feature_runner_emits_one_oof_row_per_patient_channel() -> None:
    table, folds = _table()
    result = run_feature_oof(table, folds, model_name="rbf_svm", seed=42, inner_folds=2, compact=True)
    assert len(result.oof) == len(table)
    assert not result.oof.duplicated(["subject_id", "channel_name"]).any()
    assert set(result.oof["outer_fold"]) == {1, 2, 3, 4, 5}
    assert set(result.oof["threshold_source"]) == {"inner_oof_patient_macro_f1"}
    assert "x" not in result.oof
    assert "z" not in result.oof
    for fold, train_subjects in result.scaler_fit_subjects.items():
        test_subjects = set(folds.loc[folds["outer_fold"] == fold, "subject_id"])
        assert not test_subjects & set(train_subjects)


def test_fixed_baseline_skips_inner_tuning_and_uses_fixed_threshold() -> None:
    table, folds = _table()
    result = run_feature_oof(table, folds, model_name="rbf_svm", seed=42, inner_folds=4, fixed_baseline=True)
    assert set(result.oof["threshold_source"]) == {"fixed_0.5"}
    assert set(result.oof["selected_threshold"]) == {0.5}
    assert len(result.selected_parameters) == 5


def test_fixed_validation_uses_only_explicit_validation_for_threshold() -> None:
    table, folds = _table()
    rows = []
    for fold in range(1, 6):
        validation_subject = f"p{fold % 5:02d}"
        for subject in folds["subject_id"]:
            current = int(folds.loc[folds["subject_id"] == subject, "outer_fold"].iloc[0])
            rows.append({"subject_id": subject, "outer_fold": fold, "partition": "test" if current == fold else "validation" if subject == validation_subject else "fit"})
    result = run_feature_oof(table, folds, model_name="logistic_regression", seed=42, split_manifest=pd.DataFrame(rows), selection_protocol="fixed_validation")
    assert len(result.oof) == len(table)
    assert set(result.oof["threshold_source"]) == {"outer_validation_patient_macro_f1"}


def test_task1_metrics_handles_single_class_auc_as_nan() -> None:
    rows = pd.DataFrame(
        {
            "subject_id": ["p1", "p1"],
            "label_nez": [1, 1],
            "score_nez_probability": [0.7, 0.8],
            "predicted_nez": [1, 1],
        }
    )
    metrics = compute_task1_metrics(rows)
    assert np.isnan(metrics["AUROC_EZ"])
    assert np.isnan(metrics["AUROC_NEZ"])
    assert metrics["n_patients"] == 1
