from types import SimpleNamespace

import torch

from exp_ez_hybrid import _lcbo_core_rank_loss


def test_core_rank_is_invariant_to_duplicate_hard_negatives() -> None:
    labels_positive = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    core_target = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    mask = torch.ones_like(labels_positive, dtype=torch.bool)
    target_mask = mask.clone()
    base = torch.tensor([[1.5, 0.2, -0.1, 0.7]], requires_grad=True)
    duplicated = torch.tensor([[1.5, 0.2, -0.1, 0.7, 0.2, -0.1, 0.7]], requires_grad=True)
    base_loss = _lcbo_core_rank_loss(base, labels_positive, core_target, mask, target_mask, margin=0.05, hard_neg_top_frac=1.0, min_core_mass=1.0)
    dup_labels = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dup_q = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    dup_mask = torch.ones_like(dup_labels, dtype=torch.bool)
    dup_loss = _lcbo_core_rank_loss(duplicated, dup_labels, dup_q, dup_mask, dup_mask, margin=0.05, hard_neg_top_frac=1.0, min_core_mass=1.0)
    assert torch.isfinite(base_loss) and torch.isfinite(dup_loss)
    assert torch.allclose(base_loss, dup_loss, atol=1e-6)
    dup_loss.backward()
    assert duplicated.grad is not None and torch.isfinite(duplicated.grad).all()
