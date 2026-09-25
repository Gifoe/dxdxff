from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from exp_ez_hybrid import compute_a9v8_lcbo_loss, select_best_decision_rule
from ez_features import WINDOW_NODE_FEATURE_NAMES
from patient_channel_ranker import PatientChannelClassifier
from scripts.build_task1_s5_8_cache import S5_8, _derive_s5_8
from scripts.build_nez_latent_core_targets import S5_8_SIGNS_EZ


def test_s5_8_names_and_formula_shape() -> None:
    assert tuple(S5_8) == tuple(S5_8_SIGNS_EZ)
    assert len(S5_8) == 8
    centers = np.asarray([-2.0, -1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    base = np.zeros((5, 3, len(WINDOW_NODE_FEATURE_NAMES)), dtype=np.float32)
    high = list(WINDOW_NODE_FEATURE_NAMES).index("log_bp_high_gamma")
    line = list(WINDOW_NODE_FEATURE_NAMES).index("line_length_per_sec")
    for time_idx in range(5):
        base[time_idx, :, high] = time_idx * np.asarray([1.0, 2.0, 3.0])
        base[time_idx, :, line] = time_idx * np.asarray([3.0, 2.0, 1.0])
    derived = _derive_s5_8(base, centers)
    assert derived.shape == (5, 3, 8)
    assert derived.dtype == np.float32
    assert np.isfinite(derived).all()
    assert np.all(derived[:, 0, 0] < derived[:, 2, 0])


def test_no_hfo_columns_in_s5_8() -> None:
    assert not any("hfo" in name.lower() for name in S5_8)
    assert len(WINDOW_NODE_FEATURE_NAMES) + len(S5_8) == 28


def test_nez_label_and_score_semantics_with_mask() -> None:
    torch.manual_seed(3)
    model = PatientChannelClassifier(8, num_heads=2, dropout=0.0, positive_label="nez", use_a9v8_lcbo=True)
    mask = torch.tensor([[True, True, False]])
    output = model(torch.randn(1, 3, 8), mask)
    assert output["positive_label"] == "nez"
    assert output["score_semantics"] == "nez_probability"
    assert torch.allclose(output["score_nez"][mask] + output["score_ez"][mask], torch.ones(2), atol=1e-6)
    assert output["score_nez"][~mask].eq(0).all()
    assert output["score_ez"][~mask].eq(0).all()


def test_ez_and_nez_architecture_parameter_count_matches() -> None:
    ez = PatientChannelClassifier(8, num_heads=2, positive_label="ez", use_a9v8_lcbo=True)
    nez = PatientChannelClassifier(8, num_heads=2, positive_label="nez", use_a9v8_lcbo=True)
    assert sum(parameter.numel() for parameter in ez.parameters()) == sum(parameter.numel() for parameter in nez.parameters())
    assert set(ez.state_dict()) == set(nez.state_dict())


def test_nez_asymmetric_lcbo_loss_is_finite_and_backward() -> None:
    logits_broad = torch.tensor([[1.0, 0.5, -0.5, -1.0]], requires_grad=True)
    logits_core = torch.tensor([[2.0, 1.0, -1.0, -2.0]], requires_grad=True)
    logits_eval = logits_broad + 0.1 * logits_core
    mask = torch.ones((1, 4), dtype=torch.bool)
    score_nez = torch.sigmoid(logits_eval)
    outputs = {
        "logits_broad": logits_broad,
        "logits_core": logits_core,
        "logits_eval": logits_eval,
        "score_broad": torch.sigmoid(logits_broad),
        "score_core": torch.sigmoid(logits_core),
        "score_eval": score_nez,
        "score_nez": score_nez,
        "score_ez": 1.0 - score_nez,
    }
    batch = {
        "labels_ez": torch.tensor([[0.0, 0.0, 1.0, 1.0]]),
        "channel_mask": mask,
        "subject_id": ["p1"],
        "canonical_channels": [["a", "b", "c", "d"]],
    }
    lookup = {
        ("p1", channel): {
            "pseudo_core_q": 1.0 if channel == "a" else 0.0,
            "teacher_score": teacher,
            "target_semantics": "nez",
        }
        for channel, teacher in zip(("a", "b", "c", "d"), (0.9, 0.7, 0.2, 0.1))
    }
    args = SimpleNamespace(
        experiment_mode="NEZ_S5_8_ASYMMETRIC", clinical_ez_bce_scale=0.25,
        core_rank_margin=0.05, lcbo_hard_neg_top_frac=0.3, lcbo_min_core_mass=1.0,
        soft_mrr_tau=0.1, subset_eps=0.05, lambda_core_rank=0.2,
        lambda_soft_mrr=0.05, lambda_subset=0.02, lambda_core_distill=0.05,
        lambda_weak_ez_mil=0.1, weak_ez_margin=0.05,
        weak_ez_core_fraction=0.2, weak_ez_hard_nez_fraction=0.2,
    )
    loss, parts = compute_a9v8_lcbo_loss(outputs, batch, lookup, args)
    assert torch.isfinite(loss)
    assert np.isfinite(parts["mean_weak_ez_mil_loss"])
    loss.backward()
    assert logits_broad.grad is not None and torch.isfinite(logits_broad.grad).all()


def test_validation_threshold_selection_uses_only_supplied_records() -> None:
    record = {
        "subject_id": "p1", "center": "hup", "center_id": 0,
        "canonical_channels": ["a", "b", "c", "d"],
        "channel_mask": np.ones(4, dtype=bool),
        "labels": np.asarray([1, 1, 0, 0], dtype=np.float32),
        "labels_nez": np.asarray([1, 1, 0, 0], dtype=np.float32),
        "labels_ez": np.asarray([0, 0, 1, 1], dtype=np.float32),
        "scores": np.asarray([0.9, 0.8, 0.3, 0.2], dtype=np.float32),
        "score_nez": np.asarray([0.9, 0.8, 0.3, 0.2], dtype=np.float32),
        "score_ez": np.asarray([0.1, 0.2, 0.7, 0.8], dtype=np.float32),
    }
    rule, summary, _ = select_best_decision_rule([record])
    assert rule["strategy"] == "validation_only_threshold"
    assert summary["threshold_source"] == "validation_only"
    assert 0.0 < rule["threshold"] < 1.0
