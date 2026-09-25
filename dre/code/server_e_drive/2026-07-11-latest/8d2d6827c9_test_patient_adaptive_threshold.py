from __future__ import annotations

import inspect

import numpy as np
import pytest
import torch

from neuroez_c.cane_path_cp_trainer import build_inner_crossfit_splits, train_path_head
from neuroez_c.cane_path_threshold import (
    PATH_SUMMARY_FIELDS,
    PatientAdaptiveThresholdHead,
    apply_patient_adaptive_threshold,
    build_path_summary,
    oracle_standardized_threshold,
    path_training_loss,
    robust_standardize_nez_logits,
)
from tests.cane_path_test_helpers import cane_args, cane_model, synthetic_batch


def _path_inputs():
    batch = synthetic_batch()
    outputs = cane_model()(batch)
    return outputs, batch


def test_path_summary_has_exactly_23_fields() -> None:
    assert len(PATH_SUMMARY_FIELDS) == 23 and len(set(PATH_SUMMARY_FIELDS)) == 23


def test_path_summary_shape_is_b_by_23() -> None:
    summary, standardized = build_path_summary(*_path_inputs())
    assert summary.shape == (4, 23) and standardized.shape == (4, 6)


def test_path_summary_is_finite() -> None:
    summary, _ = build_path_summary(*_path_inputs())
    assert torch.isfinite(summary).all()


def test_path_summary_fields_exclude_center_and_label() -> None:
    names = " ".join(PATH_SUMMARY_FIELDS).lower()
    assert "center" not in names and "label" not in names and "true" not in names


def test_robust_standardization_uses_median_and_iqr() -> None:
    values = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    standardized, median, iqr = robust_standardize_nez_logits(values, torch.ones_like(values, dtype=torch.bool))
    assert median.item() == pytest.approx(1.5) and iqr.item() == pytest.approx(1.5)
    assert standardized[0, 0].item() == pytest.approx(-1.0)


def test_robust_standardization_constant_logits_safe() -> None:
    result, _, iqr = robust_standardize_nez_logits(torch.ones(2, 4), torch.ones(2, 4, dtype=torch.bool))
    assert not result.any() and not iqr.any() and torch.isfinite(result).all()


def test_robust_standardization_ignores_padding() -> None:
    values = torch.tensor([[1.0, 2.0, 999.0]])
    mask = torch.tensor([[1, 1, 0]], dtype=torch.bool)
    result, median, _ = robust_standardize_nez_logits(values, mask)
    assert median.item() == pytest.approx(1.5) and result[0, 2] == 0


def test_path_head_threshold_is_bounded() -> None:
    head = PatientAdaptiveThresholdHead(max_abs_threshold=2.5)
    threshold = head(torch.randn(32, 23) * 100)
    assert threshold.abs().max() <= 2.5


def test_path_head_rejects_wrong_summary_width() -> None:
    with pytest.raises(ValueError):
        PatientAdaptiveThresholdHead()(torch.zeros(2, 22))


def test_threshold_rule_is_score_comparison_not_topk() -> None:
    score = torch.tensor([[-1.0, 0.2, 1.5]])
    result = apply_patient_adaptive_threshold(score, torch.tensor([0.1]), torch.ones(1, 3, dtype=torch.bool))
    assert result["predicted_nez_mask"].tolist() == [[False, True, True]]


def test_threshold_rule_masks_padding() -> None:
    result = apply_patient_adaptive_threshold(torch.ones(1, 3), torch.zeros(1), torch.tensor([[1, 0, 0]], dtype=torch.bool))
    assert result["predicted_nez_mask"].tolist() == [[True, False, False]]
    assert result["predicted_ez_mask"].tolist() == [[False, False, False]]


def test_threshold_rule_does_not_accept_true_count() -> None:
    parameters = inspect.signature(apply_patient_adaptive_threshold).parameters
    assert not any("count" in name or "label" in name for name in parameters)


def test_oracle_threshold_is_deterministic() -> None:
    scores = np.array([-1.0, -0.2, 0.1, 2.0])
    labels = np.array([0, 0, 1, 1])
    assert oracle_standardized_threshold(scores, labels) == oracle_standardized_threshold(scores, labels)


def test_oracle_threshold_perfect_separation() -> None:
    row = oracle_standardized_threshold(np.array([-2.0, -1.0, 1.0, 2.0]), np.array([0, 0, 1, 1]))
    assert row["oracle_macro_f1"] == 1.0 and row["oracle_standardized_threshold"] == 0.0


def test_oracle_threshold_rejects_misalignment() -> None:
    with pytest.raises(ValueError):
        oracle_standardized_threshold(np.array([0.0]), np.array([0, 1]))


def test_path_training_loss_is_finite_and_differentiable() -> None:
    predicted = torch.tensor([0.1, -0.2], requires_grad=True)
    oracle = torch.tensor([0.0, -0.1])
    scores = [torch.tensor([-1.0, 1.0]), torch.tensor([-0.2, 0.4, 1.0])]
    labels = [torch.tensor([0, 1]), torch.tensor([0, 1, 1])]
    loss, parts = path_training_loss(predicted, oracle, scores, labels)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(predicted.grad).all() and set(parts) == {
        "path_pbce", "path_soft_f1_loss", "path_threshold_huber", "path_threshold_l2"
    }


def test_inner_crossfit_covers_each_subject_once() -> None:
    subjects = [f"hup:p{i:02d}" for i in range(17)]
    splits = build_inner_crossfit_splits(subjects, 4, 42)
    heldout = [subject for split in splits for subject in split["heldout_subjects"]]
    assert sorted(heldout) == sorted(subjects) and len(heldout) == len(set(heldout))


def test_inner_crossfit_has_no_fit_validation_leakage() -> None:
    splits = build_inner_crossfit_splits([f"lzu:p{i}" for i in range(20)], 4, 42)
    assert all(not (set(row["fit_subjects"]) & set(row["heldout_subjects"])) for row in splits)


def test_inner_crossfit_is_seed_deterministic() -> None:
    subjects = [f"pediatric:p{i}" for i in range(20)]
    assert build_inner_crossfit_splits(subjects, 4, 42) == build_inner_crossfit_splits(subjects, 4, 42)


def test_path_history_reports_legal_inner_validation_metrics() -> None:
    records = []
    centers = ("hup", "lzu", "multicenter", "pediatric")
    for index in range(20):
        records.append({
            "path_summary": np.linspace(-1, 1, 23, dtype=np.float32) + index / 100,
            "standardized_nez_logit": np.array([-1.0, -0.2, 0.4, 1.2], dtype=np.float32),
            "labels_nez": np.array([0, 0, 1, 1], dtype=np.float32),
            "channel_mask": np.ones(4, dtype=bool), "center": centers[index % 4],
        })
    args = cane_args(); args.path_epochs = 2; args.path_patience = 2
    _, history = train_path_head(records, args, torch.device("cpu"))
    assert len(history) == 2 and all("validation_legal_path_macro_f1" in row for row in history)
