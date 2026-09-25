import numpy as np

from exp_ez_hybrid import _summarize_prediction_records, select_best_decision_rule


def _record():
    return {"subject_id": "p1", "canonical_channels": ["a", "b"], "labels": np.array([1., 0.]), "labels_ez": np.array([1., 0.]), "labels_nez": np.array([0., 1.]), "scores": np.array([0.8, 0.2]), "score_ez": np.array([0.8, 0.2]), "score_nez": np.array([0.2, 0.8]), "channel_mask": np.array([True, True])}


def test_primary_summary_declares_validation_only_threshold() -> None:
    summary, _ = _summarize_prediction_records([_record()])
    rule, _, _ = select_best_decision_rule([_record()])
    assert summary["strategy"] == "validation_only_threshold"
    assert rule["primary_hard_label_rule"] == "validation_only_threshold"
    assert summary["true_k_usage"] == "ranking_diagnostic_only"
    assert summary["test_true_count_used_for_primary_labels"] is False

