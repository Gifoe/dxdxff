from __future__ import annotations

import pandas as pd
import pytest

from outcome_hifos.metrics import MetricBundle
from outcome_hifos.reports.summarize import comparison_cohort_status, summarize_experiment
from outcome_hifos.training.trainer import select_early_stop_score


def _prediction_rows() -> pd.DataFrame:
    rows = []
    for seed in (11, 12):
        rows.extend(
            [
                {
                    "subject_id": "failure",
                    "center": "c1",
                    "outcome": 0,
                    "variant": "H2_HIER_POOL",
                    "seed": seed,
                    "outer_fold_idx": 0,
                    "role": "outer_test",
                    "raw_logit": 0.4,
                    "calibrated_probability": 0.6,
                    "selected_threshold": 0.7,
                    "predicted": 0,
                    "calibration_method": "platt_inner_oof",
                    "threshold_source": "inner_oof",
                    "checkpoint_path": "checkpoint.pt",
                    "fold_ledger_hash": "fold-hash",
                    "config_hash": "config-hash",
                    "cohort_id": "feature_full",
                    "subject_set_hash": "subjects-hash",
                },
                {
                    "subject_id": "success",
                    "center": "c1",
                    "outcome": 1,
                    "variant": "H2_HIER_POOL",
                    "seed": seed,
                    "outer_fold_idx": 0,
                    "role": "outer_test",
                    "raw_logit": 1.4,
                    "calibrated_probability": 0.8,
                    "selected_threshold": 0.7,
                    "predicted": 1,
                    "calibration_method": "platt_inner_oof",
                    "threshold_source": "inner_oof",
                    "checkpoint_path": "checkpoint.pt",
                    "fold_ledger_hash": "fold-hash",
                    "config_hash": "config-hash",
                    "cohort_id": "feature_full",
                    "subject_set_hash": "subjects-hash",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_summary_primary_metrics_use_stored_selected_threshold_predictions(tmp_path) -> None:
    summarize_experiment(_prediction_rows(), pd.DataFrame(), tmp_path, run_manifest={"protocol": "screening"})
    by_seed = pd.read_csv(tmp_path / "outcome_metrics_by_seed.csv")
    assert (by_seed["macro_f1"] == 1.0).all()
    assert (by_seed["macro_f1_at_0_5"] < 1.0).all()
    summary = pd.read_csv(tmp_path / "outcome_metrics_summary.csv")
    assert summary.loc[0, "macro_f1"] == 1.0
    assert summary.loc[0, "seed_count"] == 2


def test_summary_rejects_non_inner_oof_threshold_source(tmp_path) -> None:
    predictions = _prediction_rows()
    predictions.loc[0, "threshold_source"] = "outer_test"
    with pytest.raises(ValueError, match="threshold_source"):
        summarize_experiment(predictions, pd.DataFrame(), tmp_path, run_manifest={"protocol": "screening"})


def test_summary_preserves_fold_seed_threshold_and_calibration_columns(tmp_path) -> None:
    predictions = _prediction_rows()
    summarize_experiment(predictions, pd.DataFrame(), tmp_path, run_manifest={"protocol": "screening"})
    saved = pd.read_csv(tmp_path / "outcome_oof_predictions.csv")
    required = {
        "seed",
        "outer_fold_idx",
        "selected_threshold",
        "threshold_source",
        "calibration_method",
        "raw_logit",
        "calibrated_probability",
        "predicted",
    }
    assert required.issubset(saved.columns)


def test_early_stop_uses_auroc_and_falls_back_to_validation_loss_for_single_class() -> None:
    predictions = pd.DataFrame({"outcome": [0.0, 1.0], "logit": [-0.2, 0.4]})
    metrics = MetricBundle({"auroc": 0.75, "auprc": 0.8, "macro_f1": 0.6}, ((1, 0), (0, 1)), {})
    score, effective, reason = select_early_stop_score(metrics, predictions, "auroc")
    assert score == 0.75
    assert effective == "auroc"
    assert reason is None

    single_class = MetricBundle(
        {"auroc": float("nan"), "auprc": float("nan"), "macro_f1": 0.5},
        ((2, 0), (0, 0)),
        {"auroc": "single_class_target"},
    )
    score, effective, reason = select_early_stop_score(single_class, pd.DataFrame({"outcome": [0.0, 0.0], "logit": [-1.0, -0.5]}), "auroc")
    assert score < 0.0
    assert effective == "validation_loss"
    assert reason == "single_class_target"


def test_bootstrap_reference_is_declared_not_alphabetical(tmp_path) -> None:
    reference = _prediction_rows()
    candidate = reference.copy()
    candidate["variant"] = "A_ALPHABETICALLY_FIRST"
    predictions = pd.concat([candidate, reference], ignore_index=True)
    summarize_experiment(
        predictions,
        pd.DataFrame(),
        tmp_path,
        run_manifest={"protocol": "screening", "bootstrap_reference": "H2_HIER_POOL", "bootstrap_samples": 10},
    )
    bootstrap = pd.read_csv(tmp_path / "outcome_paired_bootstrap.csv")
    assert set(bootstrap["reference"]) == {"H2_HIER_POOL"}


def test_go_no_go_rejects_cohort_mismatch() -> None:
    predictions = _prediction_rows()
    other = predictions.copy()
    other["variant"] = "H9_FM_RECURRENCE"
    other["cohort_id"] = "fm_full"
    other["subject_set_hash"] = "other-subjects"
    combined = pd.concat([predictions, other], ignore_index=True)
    assert comparison_cohort_status(combined, "H9_FM_RECURRENCE", "H2_HIER_POOL") == "INVALID_COHORT_MISMATCH"


def test_go_no_go_does_not_compare_full_feature_with_fusion_intersection(tmp_path) -> None:
    feature = _prediction_rows()
    feature["variant"] = "H8_RECURRENCE"
    fusion = feature.copy()
    fusion["variant"] = "H10_LATE_FUSION"
    fusion["cohort_id"] = "fusion_intersection"
    fusion["fold_ledger_hash"] = "fusion-fold-hash"
    fusion["subject_set_hash"] = "fusion-subjects-hash"
    fusion["calibrated_probability"] = fusion["outcome"].map({0: 0.8, 1: 0.6})
    fusion["predicted"] = (fusion["calibrated_probability"] >= fusion["selected_threshold"]).astype(int)
    summarize_experiment(
        pd.concat([feature, fusion], ignore_index=True),
        pd.DataFrame(),
        tmp_path,
        run_manifest={
            "protocol": "screening",
            "bootstrap_pairs": [
                ["H9_FM_INTERSECTION", "H8_FEATURE_INTERSECTION"],
                ["H10_LATE_FUSION", "H8_FEATURE_INTERSECTION"],
            ],
            "bootstrap_samples": 10,
        },
    )
    report = (tmp_path / "outcome_final_report.md").read_text(encoding="utf-8")
    assert "NO_VISIBLE_FUSION_GAIN" not in report
