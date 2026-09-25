import inspect
import math

import numpy as np
import pytest
import torch

from exp_ez_hybrid import _summarize_prediction_records
from neuroez_c.cane_path_cp_trainer import _optimizer
from neuroez_c.p2_scope_v2 import P2ScopeV2BoundaryReranker, P2ScopeV2CardinalityHead
from neuroez_c.p2_scope_v2_decoder import (
    FORMAL_PREDICTION_SOURCE,
    PREDICTED_K_DECISION_RULE,
    beta_binomial_log_prob,
    decode_predicted_k_topk,
    predict_beta_binomial_mode,
)
from neuroez_c.p2_scope_v2_trainer import _apply_p2_lr_schedule, set_scope_fold_seed


def _record(*, masks=True, source=FORMAL_PREDICTION_SOURCE, true_count=False):
    result = {
        "subject_id": "hup:test", "center": "hup", "center_id": 0,
        "canonical_channels": ["a", "b", "c", "d"], "channel_mask": np.array([True, True, True, True]),
        "labels_nez": np.array([1., 0., 1., 0.]), "labels_ez": np.array([0., 1., 0., 1.]),
        # A threshold of .5 would produce the inverse of formal masks.
        "score_nez": np.array([.9, .1, .9, .1]), "score_ez": np.array([.1, .9, .1, .9]),
        "decision_rule": PREDICTED_K_DECISION_RULE, "scope_predicted_k": 2,
        "formal_prediction_source": source, "true_count_used_for_prediction": true_count,
    }
    if masks:
        result.update({"predicted_ez_mask": np.array([True, True, False, False]), "predicted_nez_mask": np.array([False, False, True, True])})
    return result


def test_predicted_k_rule_is_in_explicit_mask_rules_and_uses_masks_not_threshold():
    summary, enriched = _summarize_prediction_records([_record()])
    assert summary["decision_rule"] == PREDICTED_K_DECISION_RULE
    assert summary["formal_prediction_source"] == FORMAL_PREDICTION_SOURCE
    assert math.isnan(summary["classification_threshold"])
    assert summary["threshold_source"] == "not_used"
    assert enriched[0]["predicted_ez_mask"] == [1, 1, 0, 0]


def test_predicted_k_summary_rejects_missing_masks():
    with pytest.raises(ValueError):
        _summarize_prediction_records([_record(masks=False)])


def test_predicted_k_summary_rejects_count_mismatch():
    row = _record(); row["scope_predicted_k"] = 1
    with pytest.raises(RuntimeError):
        _summarize_prediction_records([row])


def test_predicted_k_summary_rejects_bad_source_and_true_count():
    with pytest.raises(RuntimeError):
        _summarize_prediction_records([_record(source="bad")])
    with pytest.raises(RuntimeError):
        _summarize_prediction_records([_record(true_count=True)])


@pytest.mark.parametrize("n,alpha,beta", [
    (1, .2, .7), (2, .5, 1.5), (3, 1., 1.), (4, 2., 5.), (5, 5., 2.),
    (6, .8, 3.), (7, 3., .8), (8, 4., 4.), (9, 1.2, 2.3), (10, 2.3, 1.2),
    (11, .3, .3), (12, 8., 1.1), (13, 1.1, 8.), (14, 2.2, 5.5), (15, 5.5, 2.2),
    (16, 9., 9.), (17, .7, 4.), (18, 4., .7), (19, 1.5, 1.5), (20, 10., 3.),
])
def test_beta_binomial_probabilities_sum_to_one(n, alpha, beta):
    k = torch.arange(n + 1, dtype=torch.float64)
    logp = beta_binomial_log_prob(k, n, torch.tensor(alpha, dtype=torch.float64), torch.tensor(beta, dtype=torch.float64))
    assert torch.allclose(torch.exp(logp).sum(), torch.tensor(1., dtype=torch.float64), atol=1e-10)


@pytest.mark.parametrize("n,k,alpha,beta", [(2, 1, 2., 5.), (5, 2, 1.2, 3.4), (9, 8, 4., 1.1), (12, 0, .7, .8), (12, 12, .7, .8), (20, 7, 5., 2.), (3, 2, 1., 1.), (7, 3, 2.5, 4.5)])
def test_beta_binomial_formula_matches_manual_reference(n, k, alpha, beta):
    expected = math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1) + math.lgamma(k + alpha) + math.lgamma(n - k + beta) - math.lgamma(n + alpha + beta) + math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    actual = beta_binomial_log_prob(k, n, torch.tensor(alpha, dtype=torch.float64), torch.tensor(beta, dtype=torch.float64)).item()
    assert actual == pytest.approx(expected, abs=1e-6)


def test_public_decoder_has_no_labels_or_true_k_argument():
    parameters = inspect.signature(decode_predicted_k_topk).parameters
    assert "labels" not in parameters and "true_k" not in parameters and "labels_ez" not in parameters


@pytest.mark.parametrize("permutation", [
    [0, 1, 2, 3, 4], [4, 3, 2, 1, 0], [1, 3, 0, 4, 2], [2, 0, 4, 1, 3], [3, 4, 1, 0, 2],
    [1, 0, 3, 2, 4], [2, 4, 0, 3, 1], [3, 1, 4, 2, 0], [4, 2, 1, 3, 0], [0, 2, 4, 3, 1],
])
def test_decoder_is_permutation_equivariant(permutation):
    base = torch.tensor([[.1, .8, .6, .2, .4]])
    mask = torch.ones_like(base, dtype=torch.bool)
    original = decode_predicted_k_topk(scope_ez_score=base, channel_mask=mask, alpha=torch.tensor([2.]), beta=torch.tensor([3.]))
    perm = torch.tensor(permutation)
    moved = decode_predicted_k_topk(scope_ez_score=base[:, perm], channel_mask=mask[:, perm], alpha=torch.tensor([2.]), beta=torch.tensor([3.]))
    inverse = torch.argsort(perm)
    assert torch.equal(original["predicted_ez_mask"], moved["predicted_ez_mask"][:, inverse])


def test_decoder_selects_exactly_predicted_k_and_only_valid_channels():
    score = torch.tensor([[.1, .8, .7, .2, .9]])
    mask = torch.tensor([[True, False, True, True, True]])
    out = decode_predicted_k_topk(scope_ez_score=score, channel_mask=mask, alpha=torch.tensor([2.]), beta=torch.tensor([5.]))
    assert int(out["predicted_ez_mask"].sum()) == int(out["scope_predicted_k"][0])
    assert not bool((out["predicted_ez_mask"] & ~mask).any())
    assert out["decision_rule"] == PREDICTED_K_DECISION_RULE


def test_mode_tie_break_is_deterministic():
    # Symmetric alpha/beta has equal central modes for an even n; choose the lower K.
    mode = predict_beta_binomial_mode(4, torch.tensor(1.), torch.tensor(1.), min_k=0, max_k=4)
    assert mode == 2


def test_scope_boundary_is_zero_initialized_and_bounded():
    head = P2ScopeV2BoundaryReranker(8); x = torch.randn(1, 5, 8); mask = torch.ones(1, 5, dtype=torch.bool)
    out = head(x, torch.randn(1, 5), torch.rand(1, 5), torch.rand(1, 5), torch.rand(1, 5), torch.rand(1, 5), torch.ones(1, 5), mask)
    assert torch.allclose(out["scope_boundary_delta"], torch.zeros_like(out["scope_boundary_delta"]))
    assert out["scope_boundary_delta"].abs().max() <= .10


def test_cardinality_uses_embedding_vectors_and_zero_output_initialization():
    head = P2ScopeV2CardinalityHead(8)
    assert head.mean_projection[1].in_features == 8 and head.std_projection[1].in_features == 8
    assert torch.count_nonzero(head.out.weight) == 0 and torch.count_nonzero(head.out.bias) == 0
    x = torch.randn(1, 4, 8); mask = torch.ones(1, 4, dtype=torch.bool)
    out = head(x, torch.randn(1, 4), torch.rand(1, 4), torch.rand(1, 4), torch.rand(1, 4), torch.rand(1, 4), torch.ones(1, 2, dtype=torch.bool), mask, .25)
    assert out["scope_count_mu_delta"].item() == pytest.approx(0., abs=1e-7)
    assert out["scope_count_predicted_mu"].item() == pytest.approx(.25, abs=1e-7)
    assert out["scope_count_predicted_kappa"].item() == pytest.approx(16., abs=1e-7)


def test_fold_seed_is_deterministic_and_distinct():
    first = set_scope_fold_seed(42, 1); a = torch.rand(4)
    second = set_scope_fold_seed(42, 1); b = torch.rand(4)
    assert first == second and torch.equal(a, b)
    assert set_scope_fold_seed(42, 2) != first


def test_scope_heads_are_in_head_optimizer_group_and_lr_schedule_is_p2_style():
    class Args: cane_backbone_lr_stage1 = 1e-4; cane_backbone_lr_after_warmup = 2e-5; cane_head_lr = 3e-4; weight_decay = .001
    class Tiny(torch.nn.Module):
        def __init__(self): super().__init__(); self.backbone = torch.nn.Linear(2, 2); self.p2_scope_boundary = torch.nn.Linear(2, 2); self.p2_scope_cardinality = torch.nn.Linear(2, 2)
    opt = _optimizer(Tiny(), Args())
    assert {group["group_name"] for group in opt.param_groups} == {"backbone", "cane_heads"}
    b1, h1 = _apply_p2_lr_schedule(opt, Args(), 5); b2, h2 = _apply_p2_lr_schedule(opt, Args(), 6)
    assert b1 == 1e-4 and b2 == 2e-5 and h1 == h2 == 3e-4


def test_summary_source_uses_fixed_label_set():
    source = inspect.getsource(_summarize_prediction_records)
    assert "labels=[0, 1]" in source


def test_trainer_has_single_public_decoder_and_disk_checkpoint_resume_contract():
    from neuroez_c import p2_scope_v2_trainer
    source = inspect.getsource(p2_scope_v2_trainer)
    assert "decode_predicted_k_topk" in source
    assert "def _decode(" not in source
    assert "best_model.pt" in source and "completion.json" in source and "validation_ledger_hash" in source


def test_checkpoint_is_rolling_and_cannot_be_before_min_epoch():
    from neuroez_c import p2_scope_v2_trainer
    source = inspect.getsource(p2_scope_v2_trainer)
    assert "len(history) >= 3" in source
    assert "epoch >= int(args.scope_checkpoint_min_epoch)" in source


def test_ledger_builder_uses_runtime_outer_splits_and_strata():
    import scripts.build_p2_scope_v2_validation_ledger as ledger
    source = inspect.getsource(ledger)
    assert "build_outer_splits" in source and "build_sensitivity80_cohort" in source
    assert "seizure_bin" in source and "validation count" in source
