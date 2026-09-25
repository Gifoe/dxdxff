from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

import exp_ez_hybrid
from data_factory import build_outer_splits
from exp_ez_hybrid import _cane_checkpoint_is_better, _summarize_prediction_records
from neuroez_c.cane_set_heads import (
    CleanNEZPrototypeAnchor,
    MultiSeizureNEZEvidenceResidual,
    PatientNEZCardinalityHead,
    deterministic_topk_mask,
    initialize_clean_nez_prototypes,
)
from neuroez_c.cane_set_loss import (
    beta_binomial_count_nll,
    clean_nez_prototype_loss,
    compute_cane_set_loss,
    high_confidence_ez_rank_loss,
    patient_balanced_nez_bce,
    patient_soft_macro_f1_loss,
    select_high_confidence_ez,
)
from neuroez_c.config import apply_pruned_defaults
from neuroez_c.model import NeuroEZCModel
from neuroez_c.protocol import STEP4B_STATIC_TOP20_FEATURES, assert_fixed_all90_protocol
from patient_channel_ranker import PatientChannelClassifier
from run_neuroez_c import build_parser, validate_cane_args
from scripts.aggregate_cane_set_ensemble import aggregate_cane_set_ensemble


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_step4c_cane_set_nez_all90.ps1"


def _args(*extra: str, model_dim: int = 8):
    values = [
        "--use_cane_set_nez", "--positive_label", "nez",
        "--loss_mode", "cane_set_nez_countfree",
        "--use_physics_dynamics", "--physics_feature_parts", "abs",
        "--physics_state_features", ",".join(STEP4B_STATIC_TOP20_FEATURES),
        "--class_weight_mode", "none", "--ez_negative_weight", "1",
        "--model_dim", str(model_dim), "--num_heads", "2", "--dropout", "0",
        "--fixed_all90_protocol_name", "fixed_all90_step4c_cane_set_nez",
        "--require-n-patients", "90", "--early_stop_metric", "cane_predcount_patient_macro_f1",
        "--min_epochs_before_early_stop", "18", "--split_seed", "42", "--model_seed", "42",
    ]
    args = build_parser().parse_args(values + list(extra))
    apply_pruned_defaults(args)
    args.random_seed = args.split_seed
    validate_cane_args(args)
    return args


def _batch(labels: bool = False) -> dict[str, torch.Tensor | list[str] | list[list[str]]]:
    torch.manual_seed(3)
    batch: dict[str, object] = {
        "b0_features": torch.randn(2, 3, 4, 5, 36),
        "physics_features": torch.randn(2, 3, 4, 5, 8),
        "channel_mask": torch.tensor([[1, 1, 1, 1, 0], [1, 1, 1, 1, 1]], dtype=torch.bool),
        "seizure_mask": torch.tensor([[1, 1, 0], [1, 1, 1]], dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 3, 5, dtype=torch.bool),
        "window_mask": torch.ones(2, 3, 4, dtype=torch.bool),
        "subject_id": ["p1", "p2"],
        "canonical_channels": [[f"a{i}" for i in range(5)], [f"b{i}" for i in range(5)]],
    }
    batch["seizure_channel_mask"][0, 2] = False
    if labels:
        batch["labels_ez"] = torch.tensor([[1, 0, 1, 0, -1], [0, 1, 0, 1, 0]], dtype=torch.float32)
    return batch  # type: ignore[return-value]


def _model() -> NeuroEZCModel:
    torch.manual_seed(5)
    return NeuroEZCModel(_args()).eval()


def _summary_record() -> dict:
    return {
        "subject_id": "p1", "center": "hup", "center_id": 0,
        "canonical_channels": ["a", "b", "c"],
        "labels_nez": np.array([1, 0, 1], np.float32),
        "labels_ez": np.array([0, 1, 0], np.float32),
        "scores": np.array([0.8, 0.2, 0.7], np.float32),
        "score_nez": np.array([0.8, 0.2, 0.7], np.float32),
        "score_ez": np.array([0.2, 0.8, 0.3], np.float32),
        "channel_mask": np.array([1, 1, 1], bool),
        "predicted_nez_mask": np.array([1, 0, 1], bool),
        "predicted_ez_mask": np.array([0, 1, 0], bool),
        "predicted_nez_fraction": 2 / 3,
        "decision_rule": "predicted_nez_cardinality_topk",
        "positive_label": "nez",
    }


def test_cane_requires_positive_label_nez() -> None:
    args = _args()
    args.positive_label = "ez"
    args.score_semantics = "ez_probability"
    with pytest.raises(ValueError, match="positive_label nez"):
        validate_cane_args(args)


def test_cane_logit_semantics_are_nez() -> None:
    out = _model()(_batch())
    valid = _batch()["channel_mask"]
    assert torch.allclose(out["score_nez"][valid], torch.sigmoid(out["final_nez_logits"])[valid])


def test_score_ez_is_one_minus_score_nez() -> None:
    out = _model()(_batch())
    assert torch.allclose(out["score_ez"], 1.0 - out["score_nez"])


def test_model_forward_is_label_blind() -> None:
    model = _model()
    bare = _batch()
    first = model(bare)["final_nez_logits"]
    changed = dict(bare)
    changed["labels_ez"] = torch.rand(2, 5)
    changed["true_nez_count"] = torch.tensor([0, 5])
    assert torch.equal(first, model(changed)["final_nez_logits"])


def test_contextual_channel_embedding_is_returned() -> None:
    out = PatientChannelClassifier(8, dropout=0)(torch.randn(2, 4, 8), torch.ones(2, 4, dtype=torch.bool))
    assert out["contextual_channel_embedding"].shape == (2, 4, 8)


def test_invalid_channels_have_zero_context_embedding() -> None:
    mask = torch.tensor([[1, 0]], dtype=torch.bool)
    out = PatientChannelClassifier(4, num_heads=2, dropout=0)(torch.randn(1, 2, 4), mask)
    assert torch.equal(out["contextual_channel_embedding"][0, 1], torch.zeros(4))


def _identity_anchor() -> CleanNEZPrototypeAnchor:
    anchor = CleanNEZPrototypeAnchor(2, projection_dim=2, num_prototypes=2, temperature=0.01)
    anchor.projector = nn.Identity()
    with torch.no_grad():
        anchor.prototypes.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    return anchor


def test_anchor_uses_normalized_prototypes() -> None:
    anchor = _identity_anchor()
    with torch.no_grad():
        anchor.prototypes.mul_(7)
    assert torch.allclose(anchor.normalized_prototypes.norm(dim=-1), torch.ones(2))


def test_anchor_distance_is_lower_near_prototype() -> None:
    anchor = _identity_anchor()
    out = anchor(torch.tensor([[[1.0, 0.0], [-1.0, 0.0]]]), torch.ones(1, 2, dtype=torch.bool))
    assert out["anchor_distance"][0, 0] < out["anchor_distance"][0, 1]


def test_anchor_residual_is_monotonic_for_nez_evidence() -> None:
    anchor = _identity_anchor()
    out = anchor(torch.tensor([[[1.0, 0.0], [0.7, 0.7], [-1.0, 0.0]]]), torch.ones(1, 3, dtype=torch.bool))
    order = torch.argsort(out["anchor_nez_evidence"][0])
    assert torch.all(torch.diff(out["anchor_residual"][0, order]) >= -1e-7)


def test_anchor_residual_is_bounded() -> None:
    out = _identity_anchor()(torch.randn(3, 9, 2), torch.ones(3, 9, dtype=torch.bool))
    assert float(out["anchor_residual"].abs().max()) <= 0.300001


def test_anchor_residual_initializes_near_zero() -> None:
    out = _identity_anchor()(torch.randn(2, 5, 2), torch.ones(2, 5, dtype=torch.bool))
    assert float(out["anchor_residual"].abs().max()) < 0.001


def test_anchor_compactness_uses_clean_nez_only() -> None:
    distance = torch.tensor([[0.1, 9.0]])
    loss = clean_nez_prototype_loss(distance, torch.tensor([[1.0, 0.0]]), torch.ones(1, 2, dtype=torch.bool), torch.eye(2))[1]
    assert loss == pytest.approx(0.1)


def test_observed_ez_does_not_enter_anchor_compactness() -> None:
    base = clean_nez_prototype_loss(torch.tensor([[0.2, 1.0]]), torch.tensor([[1.0, 0.0]]), torch.ones(1, 2, dtype=torch.bool), torch.eye(2))[1]
    changed = clean_nez_prototype_loss(torch.tensor([[0.2, 99.0]]), torch.tensor([[1.0, 0.0]]), torch.ones(1, 2, dtype=torch.bool), torch.eye(2))[1]
    assert torch.equal(base, changed)


def test_prototype_diversity_penalizes_collapse() -> None:
    collapsed = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    orthogonal = torch.eye(2)
    labels = torch.tensor([[1.0]])
    mask = torch.ones(1, 1, dtype=torch.bool)
    assert clean_nez_prototype_loss(torch.zeros(1, 1), labels, mask, collapsed)[2] > clean_nez_prototype_loss(torch.zeros(1, 1), labels, mask, orthogonal)[2]


def _fit_loader() -> list[dict]:
    batch = _batch(labels=True)
    return [batch]


def test_prototype_initialization_uses_fit_loader_only() -> None:
    audit = initialize_clean_nez_prototypes(_model(), _fit_loader(), torch.device("cpu"), 42, max_samples=20)
    assert audit["prototype_initialization_source"] == "fit_clean_nez_only"
    assert audit["used_validation"] is False and audit["used_test"] is False


def test_prototype_initialization_is_deterministic() -> None:
    first, second = _model(), _model()
    initialize_clean_nez_prototypes(first, _fit_loader(), torch.device("cpu"), 42)
    initialize_clean_nez_prototypes(second, _fit_loader(), torch.device("cpu"), 42)
    assert torch.equal(first.clean_nez_anchor.prototypes, second.clean_nez_anchor.prototypes)


def _evidence() -> MultiSeizureNEZEvidenceResidual:
    return MultiSeizureNEZEvidenceResidual(6)


def test_seizure_evidence_output_shapes() -> None:
    out = _evidence()(torch.randn(2, 3, 4, 6), torch.ones(2, 3, dtype=torch.bool), torch.ones(2, 3, 4, dtype=torch.bool))
    assert out["seizure_nez_logit"].shape == (2, 3, 4) and out["seizure_residual"].shape == (2, 4)


def test_seizure_evidence_masks_invalid_seizures() -> None:
    mask = torch.zeros(1, 2, 3, dtype=torch.bool)
    out = _evidence()(torch.randn(1, 2, 3, 6), torch.zeros(1, 2, dtype=torch.bool), mask)
    assert sum(float(out[key].abs().sum()) for key in ("seizure_residual", "seizure_nez_probability_mean", "seizure_nez_agreement")) == 0


def test_seizure_residual_initializes_to_zero() -> None:
    out = _evidence()(torch.randn(1, 2, 3, 6), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, 3, dtype=torch.bool))
    assert torch.equal(out["seizure_residual"], torch.zeros_like(out["seizure_residual"]))


def test_seizure_residual_is_bounded() -> None:
    module = _evidence()
    with torch.no_grad():
        module.evidence_residual_mlp[-1].bias.fill_(100)
    out = module(torch.randn(1, 2, 3, 6), torch.ones(1, 2, dtype=torch.bool), torch.ones(1, 2, 3, dtype=torch.bool))
    assert float(out["seizure_residual"].abs().max()) <= 0.250001


def test_seizure_agreement_is_between_zero_and_one() -> None:
    out = _evidence()(torch.randn(2, 4, 3, 6), torch.ones(2, 4, dtype=torch.bool), torch.ones(2, 4, 3, dtype=torch.bool))
    assert float(out["seizure_nez_agreement"].min()) >= 0 and float(out["seizure_nez_agreement"].max()) <= 1


def test_patient_balanced_bce_balances_classes_within_patient() -> None:
    logits = torch.tensor([[0.0, 0.0, 0.0]])
    loss, _ = patient_balanced_nez_bce(logits, torch.tensor([[1.0, 0.0, 0.0]]), torch.ones(1, 3, dtype=torch.bool))
    assert loss == pytest.approx(np.log(2))


def test_patient_balanced_bce_is_not_channel_pooled() -> None:
    logits = torch.tensor([[4.0, -4.0, 0.0, 0.0], [4.0, -4.0, -4.0, -4.0]])
    labels = torch.tensor([[1.0, 0.0, -1.0, -1.0], [1.0, 0.0, 0.0, 0.0]])
    full = patient_balanced_nez_bce(logits, labels, labels >= 0)[0]
    compact = patient_balanced_nez_bce(logits[:, :2], labels[:, :2], torch.ones(2, 2, dtype=torch.bool))[0]
    assert torch.allclose(full, compact)


def test_soft_macro_f1_uses_both_nez_and_ez() -> None:
    labels = torch.tensor([[1.0, 0.0]])
    good = patient_soft_macro_f1_loss(torch.tensor([[5.0, -5.0]]), labels, torch.ones(1, 2, dtype=torch.bool))[0]
    bad_ez = patient_soft_macro_f1_loss(torch.tensor([[5.0, 5.0]]), labels, torch.ones(1, 2, dtype=torch.bool))[0]
    assert good < bad_ez


def _hc_inputs():
    labels = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones(1, 4, dtype=torch.bool)
    direct = torch.tensor([[0.9, 0.1, 0.5, 0.8]], requires_grad=True)
    anchor = torch.tensor([[1.0, -1.0, 0.0, 1.0]], requires_grad=True)
    seizure = torch.tensor([[0.9, 0.2, 0.6, 0.8]], requires_grad=True)
    return labels, mask, direct, anchor, seizure


def test_hc_ez_selection_only_uses_observed_ez() -> None:
    args = _hc_inputs()
    selected = select_high_confidence_ez(*args, fraction=0.5)
    assert not selected[0, 0] and torch.all(args[0][selected] == 0)


def test_hc_ez_selection_picks_lowest_nez_likeness() -> None:
    selected = select_high_confidence_ez(*_hc_inputs(), fraction=0.25)
    assert selected[0, 1]


def test_hc_ez_selection_is_detached() -> None:
    selected = select_high_confidence_ez(*_hc_inputs(), fraction=0.5)
    assert selected.requires_grad is False


def test_rank_sign_is_correct_for_nez_positive_logits() -> None:
    labels = torch.tensor([[1.0, 0.0]])
    mask = torch.ones(1, 2, dtype=torch.bool)
    hc = torch.tensor([[0, 1]], dtype=torch.bool)
    low = high_confidence_ez_rank_loss(torch.tensor([[0.0, 0.0]]), labels, mask, hc)[0]
    high = high_confidence_ez_rank_loss(torch.tensor([[3.0, 0.0]]), labels, mask, hc)[0]
    assert high < low


def test_rank_uses_fixed_pair_denominator() -> None:
    logits = torch.tensor([[1.0, 0.0, -1.0]])
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones(1, 3, dtype=torch.bool)
    hc = torch.tensor([[0, 1, 1]], dtype=torch.bool)
    loss = high_confidence_ez_rank_loss(logits, labels, mask, hc)[0]
    expected = torch.nn.functional.softplus(0.1 - logits[0, 0] + logits[0, 1:]).sum() / 2
    assert torch.allclose(loss, expected)


def test_rank_disabled_before_start_epoch() -> None:
    model = _model().train()
    batch = _batch(labels=True)
    out = model({key: value for key, value in batch.items() if key != "labels_ez"})
    _, parts = compute_cane_set_loss(out, batch, _args(), epoch=15)
    assert parts["cane_hc_rank_loss"] == 0


def test_beta_binomial_nll_is_finite() -> None:
    value = beta_binomial_count_nll(torch.tensor([2.0]), torch.tensor([3.0]), torch.tensor([[1.0, 0.0, 1.0]]), torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(value)


def test_count_head_forward_does_not_read_true_count() -> None:
    signature = inspect.signature(PatientNEZCardinalityHead.forward)
    assert "true_nez_count" not in signature.parameters and "labels_nez" not in signature.parameters


def test_predicted_nez_count_is_within_valid_range() -> None:
    out = PatientNEZCardinalityHead(4)(torch.randn(2, 5, 4), torch.randn(2, 5), torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 1]], dtype=torch.bool), torch.ones(2, 3, dtype=torch.bool))
    assert torch.all(out["predicted_nez_count"] >= 0) and torch.all(out["predicted_nez_count"] <= torch.tensor([2, 5]))


def test_predicted_nez_mask_uses_predicted_count() -> None:
    scores = torch.tensor([[0.2, 0.9, 0.5]])
    selected = deterministic_topk_mask(scores, torch.ones(1, 3, dtype=torch.bool), torch.tensor([2]))
    assert selected.tolist() == [[False, True, True]]


def test_true_count_change_does_not_change_inference() -> None:
    model, batch = _model(), _batch()
    first = model(batch)["predicted_nez_mask"]
    batch["true_nez_count"] = torch.tensor([0, 99])
    assert torch.equal(first, model(batch)["predicted_nez_mask"])


def test_topk_tie_break_is_deterministic() -> None:
    selected = deterministic_topk_mask(torch.ones(1, 4), torch.ones(1, 4, dtype=torch.bool), torch.tensor([2]))
    assert selected.tolist() == [[True, True, False, False]]


def test_predicted_ez_mask_is_valid_complement() -> None:
    out = _model()(_batch())
    mask = _batch()["channel_mask"]
    assert torch.equal(out["predicted_ez_mask"], mask & ~out["predicted_nez_mask"])


def test_cane_summary_uses_explicit_prediction_masks() -> None:
    record = _summary_record()
    record["score_nez"] = np.array([0.1, 0.9, 0.1])
    summary, _ = _summarize_prediction_records([record])
    assert summary["patient_macro_f1"] == 1.0


def test_cane_summary_does_not_use_threshold() -> None:
    record = _summary_record()
    summary, enriched = _summarize_prediction_records([record], classification_threshold=0.99)
    assert np.isnan(summary["classification_threshold"]) and enriched[0]["threshold_source"] == "not_used"


def _protocol_args():
    args = _args()
    args.enforce_fixed_all90_protocol = True
    return args


def _protocol_data(n: int = 90):
    patients = {f"p{i:03d}": {} for i in range(n)}
    return patients, build_outer_splits(patients, split_strategy="5fold", n_splits=5, random_seed=42)


def test_cane_protocol_forbids_n6() -> None:
    args = _protocol_args(); args.use_n6_dual_view_ema = True
    patients, splits = _protocol_data()
    with pytest.raises(ValueError, match="use_n6_dual_view_ema"):
        assert_fixed_all90_protocol(args, patients, splits, STEP4B_STATIC_TOP20_FEATURES)


def test_cane_protocol_forbids_center_input() -> None:
    args = _protocol_args(); args.model_input_features = "center_id"
    patients, splits = _protocol_data()
    with pytest.raises(ValueError, match="center_id"):
        assert_fixed_all90_protocol(args, patients, splits, STEP4B_STATIC_TOP20_FEATURES)


def test_cane_protocol_requires_nez_probability() -> None:
    args = _protocol_args(); args.score_semantics = "ez_probability"
    patients, splits = _protocol_data()
    with pytest.raises(ValueError, match="nez_probability"):
        assert_fixed_all90_protocol(args, patients, splits, STEP4B_STATIC_TOP20_FEATURES)


def test_cane_protocol_requires_90_patients() -> None:
    patients, splits = _protocol_data(89)
    with pytest.raises(ValueError, match="90 patients"):
        assert_fixed_all90_protocol(_protocol_args(), patients, splits, STEP4B_STATIC_TOP20_FEATURES)


def test_split_seed_is_separate_from_model_seed() -> None:
    args = _args("--model_seed", "44")
    assert args.split_seed == 42 and args.model_seed == 44 and args.random_seed == 42


def test_test_loader_not_evaluated_before_checkpoint_selection() -> None:
    source = inspect.getsource(exp_ez_hybrid.Exp_EZHybridLocalization.run)
    assert source.index("model.load_state_dict(best_state)") < source.index('split_name="test"')


def _write_ensemble_seed(root: Path, seed: int, *, key_suffix: str = "", logit_shift: float = 0.0) -> None:
    seed_dir = root / f"seed_{seed}"
    seed_dir.mkdir(parents=True)
    channels = pd.DataFrame({
        "fold_idx": [1, 1, 1, 1],
        "subject_id": ["p1"] * 4,
        "channel_name": ["a", "b", "c", "d" + key_suffix],
        "channel_id": [0, 1, 2, 3],
        "center": ["hup"] * 4,
        "center_id": [0] * 4,
        "true_nez": [1, 0, 1, 0],
        "true_ez": [0, 1, 0, 1],
        "valid_channel": [True] * 4,
        "final_nez_logit": np.array([2.0, -2.0, 1.0, -1.0]) + logit_shift,
    })
    channels.to_csv(seed_dir / "test_channel_predictions_neuroez_v2_fold_1.csv", index=False)
    pd.DataFrame({"fold_idx": [1], "subject_id": ["p1"], "predicted_nez_fraction": [0.5 + logit_shift * 0.01]}).to_csv(seed_dir / "test_patient_predictions_neuroez_v2_fold_1.csv", index=False)


def test_ensemble_requires_identical_channel_keys(tmp_path: Path) -> None:
    _write_ensemble_seed(tmp_path, 42); _write_ensemble_seed(tmp_path, 43); _write_ensemble_seed(tmp_path, 44, key_suffix="x")
    with pytest.raises(ValueError, match="key mismatch"):
        aggregate_cane_set_ensemble(tmp_path)


def test_ensemble_averages_logits_and_fraction(tmp_path: Path) -> None:
    _write_ensemble_seed(tmp_path, 42, logit_shift=0); _write_ensemble_seed(tmp_path, 43, logit_shift=1); _write_ensemble_seed(tmp_path, 44, logit_shift=2)
    outputs = aggregate_cane_set_ensemble(tmp_path)
    channels = pd.read_csv(outputs["ensemble_dir"] / "heldout_channel_predictions_cane_set.csv")
    patients = pd.read_csv(outputs["ensemble_dir"] / "heldout_patient_predictions_cane_set.csv")
    assert channels.loc[0, "ensemble_nez_logit"] == pytest.approx(3.0)
    assert patients.loc[0, "predicted_nez_fraction"] == pytest.approx(0.51)


def test_ensemble_does_not_select_best_seed_using_test_labels() -> None:
    source = (ROOT / "scripts" / "aggregate_cane_set_ensemble.py").read_text(encoding="utf-8")
    assert "best_seed" not in source and "test_labels_used_for_ensemble\": False" in source


def test_final_runner_contains_three_fixed_seeds() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    assert "$seeds = @(42, 43, 44)" in text


def test_final_runner_contains_no_raw_cache() -> None:
    assert "RawCachePath" not in RUNNER.read_text(encoding="utf-8")


def test_final_runner_contains_no_true_k_prediction() -> None:
    text = RUNNER.read_text(encoding="utf-8").lower()
    assert "true_k" not in text and "true-count" not in text


def test_final_runner_contains_no_n6_args() -> None:
    assert "n6_" not in RUNNER.read_text(encoding="utf-8").lower()


def test_synthetic_forward_backward_has_finite_gradients() -> None:
    model = NeuroEZCModel(_args()).train()
    batch = _batch(labels=True)
    forward_batch = {key: value for key, value in batch.items() if key != "labels_ez"}
    out = model(forward_batch)
    loss, _ = compute_cane_set_loss(out, batch, _args(), epoch=16)
    loss.backward()
    assert torch.isfinite(loss)
    for name in ("b0_encoder", "channel_classifier", "clean_nez_anchor", "multi_seizure_evidence", "cardinality_head"):
        assert any(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in getattr(model, name).parameters())

