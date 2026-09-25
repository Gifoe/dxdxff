from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from data_factory import build_outer_splits
from ez_features import WINDOW_NODE_FEATURE_NAMES
from exp_ez_hybrid import _summarize_prediction_records
from neuroez_c.n5_pu_rankcal import (
    apply_hierarchical_calibration,
    assert_oof_subject_partition,
    build_reliability_lookup,
    confidence_weighted_pu_loss,
    fit_hierarchical_calibration,
    patient_reliability_rank_loss,
    reliability_from_nez_confidence,
    select_constrained_nez_threshold,
)
from neuroez_c.protocol import (
    STEP4B_N5_NEZ_PU_RANKCAL_PROTOCOL,
    STEP4B_STATIC_TOP20_FEATURES,
    assert_fixed_all90_protocol,
)


ROOT = Path(__file__).resolve().parents[1]


def _record(subject: str, center: str, scores: list[float], labels_ez: list[float]) -> dict:
    n = len(scores)
    return {
        "subject_id": subject,
        "center": center,
        "center_id": {"hup": 0, "lzu": 1, "multicenter": 2, "pediatric": 3}[center],
        "canonical_channels": [f"c{idx}" for idx in range(n)],
        "labels_ez": np.asarray(labels_ez, dtype=np.float32),
        "labels_nez": 1.0 - np.asarray(labels_ez, dtype=np.float32),
        "labels": 1.0 - np.asarray(labels_ez, dtype=np.float32),
        "score_nez": np.asarray(scores, dtype=np.float32),
        "score_ez": 1.0 - np.asarray(scores, dtype=np.float32),
        "scores": np.asarray(scores, dtype=np.float32),
        "channel_mask": np.ones(n, dtype=bool),
    }


def _n5_args(**updates) -> SimpleNamespace:
    values = {
        "fixed_all90_protocol_name": STEP4B_N5_NEZ_PU_RANKCAL_PROTOCOL,
        "fixed_all90_cache_audit_path": "",
        "positive_label": "nez",
        "score_semantics": "nez_probability",
        "split_strategy": "5fold",
        "n_splits": 5,
        "random_seed": 42,
        "drop_high_ez_fraction_lzu": False,
        "require_n_patients": 90,
        "physics_state_features": ",".join(STEP4B_STATIC_TOP20_FEATURES),
        "physics_feature_parts": "abs",
        "use_physics_dynamics": True,
        "use_channel_attention": True,
        "use_patient_relative_z": True,
        "group_robust_mode": "none",
        "use_diffusion_residual": False,
        "use_ez_ranking_loss": False,
        "use_hard_topk_loss": False,
        "use_negative_anchor_head": False,
        "use_two_expert_router": False,
        "use_feature_separated_two_expert": False,
        "use_broad_ez_mil_loss": False,
        "use_a9v8_lcbo": False,
        "use_teacher_anchor_eval": False,
        "teacher_anchor_apply_to_train_loss": False,
        "train_subject_dropout_file": "",
        "train_subject_dropout_count": 0,
        "use_n5_nez_pu_rankcal": True,
        "n5_inner_splits": 3,
        "n5_oof_epochs": 8,
        "n5_eta": 2.0,
        "n5_r_min": 0.25,
        "n5_lambda_u": 0.75,
        "n5_gamma_pos": 1.0,
        "n5_gamma_neg": 0.0,
        "n5_rank_loss_weight": 0.05,
        "n5_rank_margin": 0.10,
        "n5_hard_negatives_per_patient": 8,
        "n5_reliable_ez_gate": 0.5,
        "n5_min_ez_f1": 0.40,
        "n5_min_balanced_accuracy": 0.65,
        "n5_min_worst_center_f1": 0.54,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_n5_reliability_formula_is_clipped_without_hard_relabel() -> None:
    values = reliability_from_nez_confidence(np.asarray([0.0, 0.5, 0.9]), eta=2.0, r_min=0.25)
    np.testing.assert_allclose(values, np.asarray([1.0, 0.25, 0.25]))
    lookup, rows = build_reliability_lookup(
        [_record("p1", "hup", [0.9, 0.1], [1.0, 0.0])], eta=2.0, r_min=0.25
    )
    assert lookup[("p1", "c0")] == pytest.approx(0.25)
    assert lookup[("p1", "c1")] == pytest.approx(1.0)
    assert all(row["label_ez"] in {0.0, 1.0} for row in rows)


def test_n5_pu_loss_is_asymmetric_and_differentiable() -> None:
    logits = torch.tensor([[2.0, -1.0, 1.5]], requires_grad=True)
    labels_ez = torch.tensor([[0.0, 1.0, 1.0]])
    mask = torch.ones_like(labels_ez, dtype=torch.bool)
    reliability = torch.tensor([[1.0, 1.0, 0.25]])
    loss, parts = confidence_weighted_pu_loss(
        logits, labels_ez, mask, reliability,
        lambda_u=0.75, gamma_pos=1.0, gamma_neg=0.0,
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.isfinite(logits.grad).all()
    assert parts["n5_n_nez_channels"] == 1.0
    assert parts["n5_n_ez_channels"] == 2.0
    assert parts["n5_mean_ez_reliability"] == pytest.approx(0.625)


def test_n5_rank_uses_only_reliable_patient_local_semihard_ez() -> None:
    logits = torch.tensor([[1.0, 0.8, 2.0, -1.0], [0.5, 0.4, -2.0, -3.0]])
    labels_ez = torch.tensor([[0.0, 1.0, 1.0, 1.0], [0.0, 1.0, 1.0, 1.0]])
    reliability = torch.tensor([[1.0, 0.8, 0.25, 0.9], [1.0, 0.4, 0.3, 0.2]])
    mask = torch.ones_like(labels_ez, dtype=torch.bool)
    loss, parts = patient_reliability_rank_loss(
        logits, labels_ez, mask, reliability,
        margin=0.1, hard_negatives_per_patient=1, reliable_ez_gate=0.5,
    )
    assert loss.item() > 0.0
    assert parts["n5_rank_patients"] == 1.0
    assert parts["n5_rank_pairs"] == 1.0


def test_n5_calibration_has_shrunk_center_bias_and_unseen_zero() -> None:
    records = [
        _record("h1", "hup", [0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]),
        _record("l1", "lzu", [0.8, 0.7, 0.3, 0.2], [0, 0, 1, 1]),
    ]
    calibration = fit_hierarchical_calibration(records, l2_weight=10.0, max_iter=30)
    assert calibration.temperature > 0.0
    assert max(abs(value) for value in calibration.center_biases.values()) < 1.0
    unknown = dict(_record("x1", "hup", [0.8, 0.2], [0, 1]))
    unknown["center"] = "unknown"
    calibrated = apply_hierarchical_calibration([unknown], calibration)[0]
    assert calibrated["n5_calibration_center_bias"] == 0.0
    assert "score_nez_uncalibrated" in calibrated


def test_n5_threshold_selection_is_constrained_then_nez_lexicographic() -> None:
    records = [_record("p1", "hup", [0.2, 0.6, 0.8], [1, 0, 0])]

    def summarize(_records, threshold):
        if threshold < 0.6:
            return {
                "patient_macro_ez_f1": 0.30,
                "patient_macro_balanced_accuracy": 0.80,
                "worst_center_f1": 0.70,
                "patient_macro_nez_f1": 0.99,
            }
        if threshold < 0.8:
            return {
                "patient_macro_ez_f1": 0.41,
                "patient_macro_balanced_accuracy": 0.66,
                "worst_center_f1": 0.55,
                "patient_macro_nez_f1": 0.82,
                "patient_macro_nez_recall": 0.85,
                "patient_macro_nez_precision": 0.80,
                "worst_center_nez_f1": 0.70,
                "center_gap_nez_f1": 0.10,
            }
        return {
            "patient_macro_ez_f1": 0.45,
            "patient_macro_balanced_accuracy": 0.70,
            "worst_center_f1": 0.60,
            "patient_macro_nez_f1": 0.75,
            "patient_macro_nez_recall": 0.90,
            "patient_macro_nez_precision": 0.70,
            "worst_center_nez_f1": 0.60,
            "center_gap_nez_f1": 0.05,
        }

    threshold, selected, _ = select_constrained_nez_threshold(
        records, summarize,
        min_ez_f1=0.40, min_balanced_accuracy=0.65, min_worst_center_f1=0.54,
    )
    assert threshold == pytest.approx(0.6)
    assert selected["constraints_satisfied"] is True


def test_n5_oof_partition_rejects_outer_test_leakage() -> None:
    records = [_record("p1", "hup", [0.8, 0.2], [0, 1]), _record("p3", "lzu", [0.7, 0.3], [0, 1])]
    with pytest.raises(RuntimeError, match="leakage"):
        assert_oof_subject_partition(records, ["p1", "p2"], ["p3"])


def test_n5_summary_reports_center_nez_metrics() -> None:
    records = [
        _record("h1", "hup", [0.9, 0.1], [0, 1]),
        _record("l1", "lzu", [0.4, 0.6], [0, 1]),
    ]
    summary, _ = _summarize_prediction_records(records, classification_threshold=0.5)
    assert "center_hup_nez_f1" in summary
    assert "center_lzu_nez_f1" in summary
    assert "worst_center_nez_f1" in summary
    assert "center_gap_nez_f1" in summary


def test_n5_fixed_all90_protocol_accepts_only_full_n5() -> None:
    patient_index = {f"p{idx:03d}": {} for idx in range(90)}
    splits = build_outer_splits(patient_index, split_strategy="5fold", n_splits=5, random_seed=42)
    feature_names = list(WINDOW_NODE_FEATURE_NAMES) + list(STEP4B_STATIC_TOP20_FEATURES)
    audit = assert_fixed_all90_protocol(_n5_args(), patient_index, splits, feature_names)
    assert audit["method"] == "N5_NEZ_PU_RankCal"
    assert audit["n_patients"] == 90
    assert audit["test_data_used_for_calibration_or_threshold"] is False

    with pytest.raises(ValueError, match="forbids patient dropout"):
        assert_fixed_all90_protocol(
            _n5_args(train_subject_dropout_file="drop.csv", train_subject_dropout_count=1),
            patient_index,
            splits,
            feature_names,
        )


def test_n5_runner_is_all90_and_does_not_include_n0_to_n4() -> None:
    runner = (ROOT / "scripts" / "run_step4b_n5_nez_pu_rankcal_all90.ps1").read_text(encoding="utf-8")
    assert '"--require-n-patients", "90"' in runner
    assert '"--use_n5_nez_pu_rankcal"' in runner
    assert '"--positive_label", "nez"' in runner
    assert "TrainSubjectDropout" not in runner
    for name in ("N0", "N1", "N2", "N3", "N4"):
        assert name not in runner

