import math

from neuroez_c.task2.metrics import bootstrap_metrics, compute_metrics, expected_calibration_error


def test_success_positive_and_derived_failure_metrics_and_calibration():
    metrics = compute_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9], 0.5)
    assert metrics["success_auprc"] == 1.0 and metrics["failure_auprc"] == 1.0
    assert metrics["auroc"] == 1.0
    assert metrics["success_recall"] == 1.0 and metrics["failure_recall"] == 1.0 and metrics["specificity_for_failure"] == 1.0
    assert 0 <= expected_calibration_error([0, 1], [0.1, 0.9]) <= 1


def test_single_class_and_bootstrap_handling():
    metrics = compute_metrics([0, 0], [0.1, 0.2])
    assert math.isnan(metrics["auroc"]) and math.isnan(metrics["success_auprc"])
    table, skipped = bootstrap_metrics([0, 0, 1], [0.1, 0.2, 0.8], 0.5, repeats=20, seed=42)
    assert len(table) + skipped == 20
    assert set(table["decision_threshold"]) == {0.5}
    assert set(table["inner_cv"]) == {False}
