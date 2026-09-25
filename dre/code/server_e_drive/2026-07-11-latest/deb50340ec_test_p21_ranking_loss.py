from __future__ import annotations

import torch

from neuroez_c.p21_ranking_loss import compute_observed_ez_reliability, ranking_preservation_loss, reliability_weighted_pairwise_rank_loss


def test_reliability_is_detached_bounded_and_omits_missing_causal():
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool)
    direct = torch.tensor([[0.9, 0.1, 0.8]], requires_grad=True)
    result = compute_observed_ez_reliability(labels, mask, direct, direct, direct, direct, causal_available=torch.tensor([[0, 0, 1]]))
    assert result["ez_reliability"].requires_grad is False
    assert result["ez_reliability"][0, 1] > result["ez_reliability"][0, 2]
    assert result["reliability_component_count"][0, 1] == 3
    assert result["ez_reliability"].min() >= 0.1 and result["ez_reliability"].max() <= 1


def test_rank_sign_cap_determinism_and_warmup():
    labels = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool); reliability = torch.ones_like(labels)
    good = torch.tensor([[2.0, 1.5, -1.0, -2.0]], requires_grad=True)
    bad = -good.detach().clone().requires_grad_(True)
    warm, diag = reliability_weighted_pairwise_rank_loss(good, labels, mask, reliability, epoch=1, active_from_epoch=2)
    assert warm == 0 and diag["weighted_rank_pair_count"] == 0
    loss_good, diag1 = reliability_weighted_pairwise_rank_loss(good, labels, mask, reliability, epoch=2, active_from_epoch=2, max_pairs_per_patient=3, subject_ids=["a"])
    loss_bad, _ = reliability_weighted_pairwise_rank_loss(bad, labels, mask, reliability, epoch=2, active_from_epoch=2, max_pairs_per_patient=3, subject_ids=["a"])
    loss_again, diag2 = reliability_weighted_pairwise_rank_loss(good, labels, mask, reliability, epoch=2, active_from_epoch=2, max_pairs_per_patient=3, subject_ids=["a"])
    assert loss_good < loss_bad
    assert diag1["weighted_rank_pair_count"] == 3
    assert torch.equal(loss_good, loss_again) and torch.equal(diag1["weighted_rank_pair_count"], diag2["weighted_rank_pair_count"])


def test_preserve_safe_margin_beta_and_zero_violation():
    labels = torch.tensor([[1.0, 0.0, 0.0]])
    mask = torch.ones_like(labels, dtype=torch.bool); rel = torch.ones_like(labels)
    reference = torch.tensor([[1.0, 0.8, 0.0]])
    final = torch.tensor([[1.0, 0.8, 0.0]], requires_grad=True)
    loss, diag = ranking_preservation_loss(final, reference, labels, mask, rel, safe_margin=0.3, beta=0.8, epoch=10, active_from_epoch=10)
    assert diag["preserve_pair_count"] == 1
    assert loss == 0
    warm, _ = ranking_preservation_loss(final, reference, labels, mask, rel, epoch=1, active_from_epoch=10)
    assert warm == 0

