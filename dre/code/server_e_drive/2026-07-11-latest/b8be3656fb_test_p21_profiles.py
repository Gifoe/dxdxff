from __future__ import annotations

import pytest
import torch

from neuroez_c.config import apply_pruned_defaults
from neuroez_c.model import NeuroEZCModel
from neuroez_c.p21_loss import compute_p21_loss
from neuroez_c.p21_profiles import P21_PROFILES, get_p21_profile
from neuroez_c.p21_trainer import _checkpoint_key
from run_neuroez_c import build_parser, validate_p21_args
from tests.cane_path_test_helpers import cane_args, synthetic_batch


def p21_args(*extra):
    args = build_parser().parse_args([
        "--use-p21-v3-asrr-nez", "--positive-label", "nez", "--loss-mode", "p21_v3_asrr_nez",
        "--cohort-mode", "sensitivity80", "--require-n-patients", "80", "--p21-init-mode", "from_scratch",
        "--use-causal-propagation-residual", "--use_physics_dynamics", "--physics_feature_parts", "abs",
        "--physics_state_features", "early_high_gamma_slope,early_line_length_slope,onset_latency_high_gamma,onset_latency_line_length,onset_rank_high_gamma,onset_rank_line_length,high_gamma_top20pct_mean,line_length_top20pct_mean",
        "--v3-anchor-ledger-path", "dummy.csv", "--dry-run-config-only", *extra,
    ])
    args.score_semantics = "nez_probability"
    return args


def test_profiles_are_one_shared_cumulative_configuration():
    assert len(P21_PROFILES) == 6
    assert get_p21_profile("R0_CURRENT_P2").legacy_p2
    assert not get_p21_profile("R1_BOUNDED_SIMPLEX").v3_anchor
    assert get_p21_profile("R2_WEIGHTED_RANK_PRESERVE").weighted_rank
    assert get_p21_profile("R3_V3_ANCHORED").v3_anchor
    assert get_p21_profile("R4_Q10_MULTI_SEIZURE").q10_seizure
    assert get_p21_profile("R5_FULL").center_alignment


def test_validation_fail_closed_and_checkpoint_is_threshold_free():
    validate_p21_args(p21_args("--p21-profile", "R5_FULL", "--v3-anchor-mode", "precomputed_oof_screening"))
    with pytest.raises(ValueError, match="gate priors"):
        validate_p21_args(p21_args("--p21-gate-prior-noop", "0.5"))
    with pytest.raises(ValueError, match="requires a V3"):
        validate_p21_args(p21_args("--v3-anchor-ledger-path", ""))
    first = {"checkpoint_patient_macro_f1": .7, "checkpoint_patient_nez_f1": .8, "checkpoint_patient_ez_f1": .6, "balanced_patient_auprc_hmean": .7, "patient_macro_auprc_ez": .5, "patient_macro_ez_mrr": .6}
    second = {**first, "checkpoint_patient_macro_f1": .71}
    assert _checkpoint_key(second, 2) > _checkpoint_key(first, 1)


def test_r5_synthetic_forward_loss_backward_and_stage_contract():
    args = p21_args("--p21-profile", "R5_FULL", "--model_dim", "8", "--num_heads", "2", "--dropout", "0")
    apply_pruned_defaults(args)
    model = NeuroEZCModel(args).train()
    batch = synthetic_batch(labels=True)
    shape = batch["channel_mask"].shape
    batch.update({
        "cp_valid_window_fraction": torch.ones(shape),
        "cp_valid_seizure_count": torch.full(shape, 3.0),
        "cp_mean_var_stability": torch.full(shape, 0.5),
        "v3_score_nez": torch.sigmoid(torch.randn(shape)),
        "v3_logit_nez": torch.randn(shape),
        "v3_anchor_valid": batch["channel_mask"].clone(),
    })
    output = model(batch)
    assert torch.allclose(output["w_noop"] + output["w_anchor"] + output["w_seizure"] + output["w_causal"], batch["channel_mask"].float())
    assert output["delta"].abs().max() <= 0.2 + 1e-7
    loss, diagnostics = compute_p21_loss(output, batch, args, epoch=10)
    assert torch.isfinite(loss)
    assert diagnostics["p21_profile"] == "R5_FULL"
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad and parameter.grad is not None]
    assert gradients and all(torch.isfinite(gradient).all() for gradient in gradients)


def test_r0_is_numerically_the_historical_p2_path():
    import copy

    historical_args = cane_args("--use-causal-propagation-residual")
    r0_args = copy.deepcopy(historical_args)
    r0_args.use_cane_path_cp_nez = False
    r0_args.use_p21_v3_asrr_nez = True
    r0_args.p21_profile = "R0_CURRENT_P2"
    r0_args.loss_mode = "p21_v3_asrr_nez"
    torch.manual_seed(123)
    historical = NeuroEZCModel(historical_args).eval()
    torch.manual_seed(123)
    r0 = NeuroEZCModel(r0_args).eval()
    batch = synthetic_batch(labels=True)
    with torch.no_grad():
        torch.manual_seed(999)
        expected = historical(batch)
        torch.manual_seed(999)
        actual = r0(batch)
    assert historical.state_dict().keys() == r0.state_dict().keys()
    assert all(torch.equal(historical.state_dict()[key], r0.state_dict()[key]) for key in historical.state_dict())
    for field in ("logits", "score_nez", "score_ez", "anchor_residual", "seizure_residual", "causal_residual"):
        assert torch.equal(expected[field], actual[field])
    assert actual["p21_legacy_p2_path"] is True
