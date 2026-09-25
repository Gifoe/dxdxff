from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import exp_ez_hybrid
from data_factory import build_outer_splits
from exp_ez_hybrid import _select_n5f_validation_threshold
from neuroez_c.model import NeuroEZCModel
from neuroez_c.n5_final_nez_pu import (
    bounded_patient_prior,
    build_patient_context,
    cross_seizure_consistency_loss,
    patient_conditional_nnpu_loss,
    patient_latent_soft_rank_loss,
    selection_inverse_weights,
    selection_propensity_loss,
)
from neuroez_c.protocol import (
    STEP4B_N5_FINAL_NEZ_PU_PROTOCOL,
    STEP4B_STATIC_TOP20_FEATURES,
    assert_fixed_all90_protocol,
)
from run_neuroez_c import build_parser, validate_n5f_args


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_step4b_n5_final_nez_pu_all90.ps1"


def _model_args(**updates: object) -> SimpleNamespace:
    values = {
        "model_dim": 8,
        "num_heads": 2,
        "dropout": 0.0,
        "positive_label": "nez",
        "use_channel_attention": True,
        "use_patient_relative_z": True,
        "use_physics_dynamics": False,
        "use_diffusion_residual": False,
        "use_negative_anchor_head": False,
        "use_view_gated_fusion": False,
        "use_two_expert_router": False,
        "use_feature_separated_two_expert": False,
        "use_a9v8_lcbo": False,
        "use_n5_final_nez_pu": True,
        "group_robust_mode": "none",
        "temporal_pooling": "mean",
        "channel_pooling_mode": "mean",
        "record_pooling": "mean",
        "n5f_pi_min": 0.02,
        "n5f_pi_max": 0.40,
        "n5f_pi_anchor": 0.15,
        "n5f_propensity_e_min": 0.10,
        "n5f_propensity_init": 0.70,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _protocol_args(**updates: object) -> SimpleNamespace:
    values = {
        "fixed_all90_protocol_name": STEP4B_N5_FINAL_NEZ_PU_PROTOCOL,
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
        "early_stop_metric": "patient_macro_auprc_nez",
        "loss_mode": "n5f_patient_conditional_nnpu",
        "use_n5_final_nez_pu": True,
        "n5f_pi_min": 0.02,
        "n5f_pi_max": 0.40,
        "n5f_pi_anchor": 0.15,
        "n5f_prior_anchor_weight": 0.05,
        "n5f_propensity_e_min": 0.10,
        "n5f_propensity_init": 0.70,
        "n5f_propensity_w_max": 5.0,
        "n5f_propensity_loss_weight": 0.10,
        "n5f_rank_loss_weight": 0.05,
        "n5f_rank_margin": 0.10,
        "n5f_consistency_loss_weight": 0.05,
        "n5f_consistency_min_seizures": 2,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _protocol_inputs() -> tuple[dict[str, dict], list[dict]]:
    patients = {f"p{index:03d}": {} for index in range(90)}
    splits = build_outer_splits(patients, split_strategy="5fold", n_splits=5, random_seed=42)
    return patients, splits


def _valid_features() -> list[str]:
    return ["base"] + list(STEP4B_STATIC_TOP20_FEATURES)


def _prediction(subject: str, scores: list[float], labels_ez: list[int]) -> dict:
    return {
        "subject_id": subject,
        "center": "hup",
        "score_nez": np.asarray(scores, dtype=np.float64),
        "labels_nez": 1.0 - np.asarray(labels_ez, dtype=np.float64),
        "channel_mask": np.ones(len(scores), dtype=bool),
        "canonical_channels": [f"ch{index}" for index in range(len(scores))],
    }


def test_patient_prior_is_bounded() -> None:
    prior = bounded_patient_prior(torch.tensor([-100.0, 0.0, 100.0]), pi_min=0.02, pi_max=0.40)
    assert torch.all(prior >= 0.02)
    assert torch.all(prior <= 0.40)


def test_patient_prior_initializes_to_anchor() -> None:
    model = NeuroEZCModel(_model_args())
    context = torch.randn(4, 32)
    prior = bounded_patient_prior(model.n5f_prior_head(context).squeeze(-1), pi_min=0.02, pi_max=0.40)
    assert torch.allclose(prior, torch.full_like(prior, 0.15), atol=1e-6)


def test_propensity_is_bounded() -> None:
    raw = torch.tensor([-100.0, 0.0, 100.0])
    propensity = 0.10 + 0.90 * torch.sigmoid(raw)
    assert torch.all(propensity >= 0.10)
    assert torch.all(propensity <= 1.0)


def test_propensity_initializes_to_configured_value() -> None:
    model = NeuroEZCModel(_model_args())
    raw = model.n5f_propensity_head(torch.randn(2, 3, 48)).squeeze(-1)
    propensity = 0.10 + 0.90 * torch.sigmoid(raw)
    assert torch.allclose(propensity, torch.full_like(propensity, 0.70), atol=1e-6)


def test_inverse_propensity_weights_are_clipped() -> None:
    propensity = torch.tensor([[0.01, 0.20, 0.90]])
    labels = torch.tensor([[0.0, 0.0, 1.0]])
    weights, diagnostics = selection_inverse_weights(
        propensity, labels, torch.ones_like(labels, dtype=torch.bool), w_max=5.0
    )
    assert float(weights.max()) <= 5.0
    assert diagnostics["n5f_ipw_max"] <= 5.0


def test_inverse_propensity_weights_have_patient_mean_one() -> None:
    propensity = torch.tensor([[0.2, 0.8, 0.5], [0.1, 0.4, 0.7]])
    labels = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    weights, _ = selection_inverse_weights(
        propensity, labels, torch.ones_like(labels, dtype=torch.bool), w_max=5.0
    )
    assert torch.allclose(weights[:, :2].mean(dim=1), torch.ones(2), atol=1e-6)


def test_propensity_loss_is_patient_balanced() -> None:
    labels = torch.tensor([[0.0, 1.0, -1.0, -1.0], [0.0, 1.0, 0.0, 1.0]])
    mask = labels >= 0
    propensity = torch.tensor([[0.8, 0.3, 0.0, 0.0], [0.8, 0.3, 0.8, 0.3]])
    loss, _ = selection_propensity_loss(propensity, labels, mask)
    expected = 0.5 * (-torch.log(torch.tensor(0.8)) - torch.log(torch.tensor(0.7)))
    assert torch.allclose(loss, expected, atol=1e-6)


def test_nnpu_is_patient_balanced() -> None:
    logits = torch.tensor([[2.0, -1.0, 0.0, 0.0], [-2.0, 1.0, -2.0, 1.0]])
    labels = torch.tensor([[0.0, 1.0, -1.0, -1.0], [0.0, 1.0, 0.0, 1.0]])
    mask = labels >= 0
    weights = (labels == 0).float()
    loss, _ = patient_conditional_nnpu_loss(logits, labels, mask, torch.tensor([0.15, 0.15]), weights)
    compact_logits = torch.tensor([[2.0, -1.0], [-2.0, 1.0]])
    compact_labels = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    compact_loss, _ = patient_conditional_nnpu_loss(
        compact_logits,
        compact_labels,
        torch.ones_like(compact_labels, dtype=torch.bool),
        torch.tensor([0.15, 0.15]),
        (compact_labels == 0).float(),
    )
    assert torch.allclose(loss, compact_loss, atol=1e-6)


def test_nnpu_nonnegative_correction() -> None:
    logits = torch.tensor([[10.0, -10.0]], requires_grad=True)
    labels = torch.tensor([[0.0, 1.0]])
    loss, diagnostics = patient_conditional_nnpu_loss(
        logits, labels, torch.ones_like(labels, dtype=torch.bool), torch.tensor([0.40]), torch.tensor([[1.0, 0.0]])
    )
    assert diagnostics["n5f_raw_negative_risk"] < 0.0
    assert diagnostics["n5f_nonnegative_correction_rate"] == 1.0
    assert float(loss.detach()) >= 0.0


def test_nnpu_skips_missing_class_patient() -> None:
    labels = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    loss, diagnostics = patient_conditional_nnpu_loss(
        torch.zeros_like(labels, requires_grad=True),
        labels,
        torch.ones_like(labels, dtype=torch.bool),
        torch.tensor([0.15, 0.15]),
        (labels == 0).float(),
    )
    assert float(loss.detach()) == 0.0
    assert diagnostics["n5f_skipped_no_positive_patient_count"] == 1.0
    assert diagnostics["n5f_skipped_no_unlabeled_patient_count"] == 1.0


def test_soft_rank_prefers_higher_nez_logits_for_clean_nez() -> None:
    labels = torch.tensor([[0.0, 1.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    low, _ = patient_latent_soft_rank_loss(torch.tensor([[0.0, -1.0]]), labels, mask, margin=0.1)
    high, _ = patient_latent_soft_rank_loss(torch.tensor([[3.0, -1.0]]), labels, mask, margin=0.1)
    assert high < low


def test_soft_rank_uses_fixed_pair_denominator() -> None:
    labels = torch.tensor([[0.0, 1.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    high_mass, _ = patient_latent_soft_rank_loss(torch.tensor([[-10.0, -4.0]]), labels, mask, margin=0.1)
    low_mass, _ = patient_latent_soft_rank_loss(torch.tensor([[-10.0, 4.0]]), labels, mask, margin=0.1)
    assert low_mass < high_mass


def test_consistency_zero_for_identical_seizure_rankings() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]])
    loss, _ = cross_seizure_consistency_loss(
        logits, torch.tensor([[True, True]]), torch.ones_like(logits, dtype=torch.bool), torch.ones((1, 3), dtype=torch.bool), min_seizures=2
    )
    assert float(loss) < 1e-7


def test_consistency_positive_for_disagreeing_seizures() -> None:
    logits = torch.tensor([[[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]])
    loss, diagnostics = cross_seizure_consistency_loss(
        logits, torch.tensor([[True, True]]), torch.ones_like(logits, dtype=torch.bool), torch.ones((1, 3), dtype=torch.bool), min_seizures=2
    )
    assert float(loss) > 0.0
    assert diagnostics["n5f_consistency_channel_count"] == 3.0


def test_consistency_ignores_padding_and_single_observation_channels() -> None:
    logits = torch.tensor([[[1.0, 2.0, 100.0], [1.0, 2.0, -100.0], [50.0, -50.0, 0.0]]])
    seizure_mask = torch.tensor([[True, True, False]])
    seizure_channels = torch.tensor([[[True, True, True], [True, True, False], [True, True, True]]])
    loss, diagnostics = cross_seizure_consistency_loss(
        logits, seizure_mask, seizure_channels, torch.ones((1, 3), dtype=torch.bool), min_seizures=2
    )
    altered_padding = logits.clone()
    altered_padding[:, 2, :] = torch.tensor([[-999.0, 999.0, 500.0]])
    altered_loss, _ = cross_seizure_consistency_loss(
        altered_padding, seizure_mask, seizure_channels, torch.ones((1, 3), dtype=torch.bool), min_seizures=2
    )
    assert torch.allclose(loss, altered_loss)
    assert torch.isfinite(loss)
    assert diagnostics["n5f_consistency_channel_count"] == 2.0


def test_propensity_head_does_not_backpropagate_to_backbone() -> None:
    model = NeuroEZCModel(_model_args())
    embedding = torch.randn(2, 3, 16, requires_grad=True)
    context = build_patient_context(embedding, torch.ones((2, 3), dtype=torch.bool))
    raw = model.n5f_propensity_head(torch.cat([embedding.detach(), context.detach().unsqueeze(1).expand(-1, 3, -1)], dim=-1)).squeeze(-1)
    propensity = 0.10 + 0.90 * torch.sigmoid(raw)
    labels = torch.tensor([[0.0, 1.0, 0.0], [1.0, 0.0, 1.0]])
    loss, _ = selection_propensity_loss(propensity, labels, torch.ones_like(labels, dtype=torch.bool))
    loss.backward()
    assert embedding.grad is None
    assert all(parameter.grad is not None for parameter in model.n5f_propensity_head.parameters())


def test_prior_head_does_not_backpropagate_context_to_backbone() -> None:
    model = NeuroEZCModel(_model_args())
    embedding = torch.randn(2, 3, 16, requires_grad=True)
    context = build_patient_context(embedding, torch.ones((2, 3), dtype=torch.bool))
    prior = bounded_patient_prior(model.n5f_prior_head(context.detach()).squeeze(-1), pi_min=0.02, pi_max=0.40)
    loss = (prior - 0.20).square().mean()
    loss.backward()
    assert embedding.grad is None
    assert all(parameter.grad is not None for parameter in model.n5f_prior_head.parameters())


def test_no_center_id_in_n5f_heads() -> None:
    model = NeuroEZCModel(_model_args())
    names = [name.lower() for name, _ in model.named_parameters() if name.startswith("n5f_")]
    assert names
    assert all("center" not in name for name in names)


def test_validation_threshold_does_not_use_test_records(monkeypatch: pytest.MonkeyPatch) -> None:
    validation = [_prediction("v1", [0.9, 0.1], [0, 1]), _prediction("v2", [0.8, 0.2], [0, 1])]
    test_a = [_prediction("t1", [0.99, 0.98], [1, 1])]
    test_b = [_prediction("t1", [0.01, 0.02], [0, 0])]
    seen_subject_sets: list[set[str]] = []

    def summarize(records: list[dict], classification_threshold: float) -> tuple[dict[str, float], list[dict]]:
        seen_subject_sets.append({str(record["subject_id"]) for record in records})
        quality = 1.0 - abs(float(classification_threshold) - 0.5)
        return {
            "patient_macro_f1": quality,
            "patient_macro_balanced_accuracy": quality,
            "patient_macro_nez_f1": quality,
            "patient_macro_ez_f1": quality,
        }, list(records)

    monkeypatch.setattr(exp_ez_hybrid, "_summarize_prediction_records", summarize)
    threshold, _, _ = _select_n5f_validation_threshold(validation)
    assert _select_n5f_validation_threshold(validation)[0] == threshold
    assert seen_subject_sets
    assert all(subjects == {"v1", "v2"} for subjects in seen_subject_sets)
    assert all("t1" not in subjects for subjects in seen_subject_sets)
    assert test_a[0]["subject_id"] == test_b[0]["subject_id"]


def test_final_runner_uses_all90_nez_protocol() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "fixed_all90_step4b_n5_final_nez_pu" in text
    assert '"--positive_label", "nez"' in text
    assert '"--early_stop_metric", "patient_macro_auprc_nez"' in text
    assert '"--require-n-patients", "90"' in text


def test_final_runner_contains_no_old_n5_oof_arguments() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    forbidden = [
        "n5_inner_" + "splits",
        "n5_oof_" + "epochs",
        "n5_" + "eta",
        "n5_reliable_" + "ez_gate",
        "n5_" + "calibration",
    ]
    assert not any(name in text for name in forbidden)


def test_protocol_forbids_legacy_n5_flags() -> None:
    patients, splits = _protocol_inputs()
    legacy_name = "use_n5_" + "nez_pu_rankcal"
    with pytest.raises(ValueError, match="legacy N5"):
        assert_fixed_all90_protocol(
            _protocol_args(**{legacy_name: True}), patients, splits, cache_feature_names=_valid_features()
        )


def test_protocol_accepts_exact_n5f_configuration() -> None:
    patients, splits = _protocol_inputs()
    audit = assert_fixed_all90_protocol(
        _protocol_args(), patients, splits, cache_feature_names=_valid_features()
    )
    assert audit["method"] == "N5F_NEZ_PC_SA_nnPU_RankCons"
    assert audit["inner_oof_teacher"] is False
    assert audit["threshold_source"] == "final_model_validation_only"


@pytest.mark.parametrize(
    "updates",
    [
        {"positive_label": "ez"},
        {"use_two_expert_router": True},
        {"use_a9v8_lcbo": True},
        {"use_negative_anchor_head": True},
        {"use_diffusion_residual": True},
        {"group_robust_mode": "group_dro"},
    ],
)
def test_model_rejects_incompatible_n5f_branches(updates: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="N5F model configuration"):
        NeuroEZCModel(_model_args(**updates))


def test_parser_builds_with_only_final_n5_arguments() -> None:
    parser = build_parser()
    args = parser.parse_args(["--use_n5_final_nez_pu"])
    validate_n5f_args(args)
    assert args.n5f_pi_anchor == pytest.approx(0.15)
    assert args.loss_mode == "n5f_patient_conditional_nnpu"
    assert not hasattr(args, "n5_inner_" + "splits")


@pytest.mark.parametrize(
    "arguments",
    [
        ["--n5f_pi_min", "0.20", "--n5f_pi_anchor", "0.15"],
        ["--n5f_propensity_e_min", "0.8", "--n5f_propensity_init", "0.7"],
        ["--n5f_propensity_w_max", "0.5"],
        ["--n5f_rank_loss_weight", "-0.1"],
        ["--n5f_rank_margin", "-0.1"],
        ["--n5f_consistency_min_seizures", "1"],
    ],
)
def test_parser_rejects_invalid_n5f_parameters(arguments: list[str]) -> None:
    args = build_parser().parse_args(["--use_n5_final_nez_pu", *arguments])
    with pytest.raises(ValueError, match="N5F"):
        validate_n5f_args(args)
