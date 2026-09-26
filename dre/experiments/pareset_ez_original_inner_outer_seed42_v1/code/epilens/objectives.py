"""Independent PRQ-Net and BCR-Net training objectives."""

from __future__ import annotations

import torch
import torch.nn.functional as functional


def prq_loss(logit_nez: torch.Tensor, label_nez: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Masked channel BCE with NEZ=1 and EZ=0."""
    loss = functional.binary_cross_entropy_with_logits(
        logit_nez, label_nez.to(logit_nez.dtype), reduction="none"
    )
    return loss[valid].mean()


def boundary_loss(
    logit_ez: torch.Tensor,
    label_nez: torch.Tensor,
    valid: torch.Tensor,
    margin: float = 0.05,
) -> torch.Tensor:
    """Separate weak annotated-EZ scores from the strongest NEZ impostors."""
    ez = valid & (label_nez == 0)
    nez = valid & (label_nez == 1)
    n_ez = int(ez.sum())
    n_nez = int(nez.sum())
    if n_ez == 0 or n_nez == 0:
        return logit_ez.new_zeros(())
    positive_count = max(1, int(torch.ceil(logit_ez.new_tensor(0.30 * n_ez)).item()))
    negative_count = min(n_nez, max(1, min(16, n_ez)))
    weak_ez = torch.topk(logit_ez[ez], positive_count, largest=False).values
    hard_nez = torch.topk(logit_ez[nez], negative_count, largest=True).values
    return functional.softplus(margin + hard_nez.mean() - weak_ez.mean())


def coverage_loss(
    logit_ez: torch.Tensor,
    label_nez: torch.Tensor,
    valid: torch.Tensor,
    rank_temperature: float = 0.10,
    membership_temperature: float = 0.25,
) -> torch.Tensor:
    """Encourage every annotated EZ channel to enter the leading true-K ranks."""
    scores = logit_ez[valid]
    labels = label_nez[valid]
    ez_indices = torch.nonzero(labels == 0, as_tuple=False).flatten()
    true_k = int(len(ez_indices))
    if true_k == 0:
        return logit_ez.new_zeros(())
    pairwise = scores.unsqueeze(0) - scores.unsqueeze(1)
    # Row c estimates how many channels have a larger EZ logit than channel c.
    soft_rank = 1.0 + torch.sigmoid(pairwise / rank_temperature).sum(dim=1) - 0.5
    membership = torch.sigmoid(
        (true_k + 0.5 - soft_rank) / membership_temperature
    )
    return 1.0 - membership[ez_indices].mean()


def bcr_loss(
    logit_ez: torch.Tensor,
    label_nez: torch.Tensor,
    valid: torch.Tensor,
    boundary_weight: float = 0.05,
    coverage_weight: float = 0.08,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Patient-balanced BCR objective for one patient."""
    target_ez = 1 - label_nez
    bce = functional.binary_cross_entropy_with_logits(
        logit_ez[valid], target_ez[valid].to(logit_ez.dtype)
    )
    boundary = boundary_loss(logit_ez, label_nez, valid)
    coverage = coverage_loss(logit_ez, label_nez, valid)
    total = bce + boundary_weight * boundary + coverage_weight * coverage
    return total, {"bce": bce, "boundary": boundary, "coverage": coverage}
