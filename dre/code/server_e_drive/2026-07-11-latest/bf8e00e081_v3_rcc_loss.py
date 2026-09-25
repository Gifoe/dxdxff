"""Patient-equal V3-RCC loss components.

All functions use EZ-positive logits because RCC preserves the historical V3
supervised adapter.  Formal decoder semantics remain NEZ probability.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def patient_mean_unweighted_bce(logits: torch.Tensor, labels: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    """Mean BCE per patient, then mean across patients (never channel pooled)."""
    patient_losses = []
    for index in range(logits.shape[0]):
        valid = channel_mask[index].bool() & (labels[index] >= 0.0)
        if torch.any(valid):
            patient_losses.append(F.binary_cross_entropy_with_logits(logits[index, valid], labels[index, valid]))
    return torch.stack(patient_losses).mean() if patient_losses else logits.sum() * 0.0


def linear_ramp_weight(epoch: int, start_epoch: int = 3, end_epoch: int = 10, max_weight: float = 0.03) -> float:
    """One-based epoch ramp: 0 through ``start_epoch``, full at ``end_epoch``."""
    if end_epoch <= start_epoch:
        raise ValueError("end_epoch must be greater than start_epoch")
    if epoch <= start_epoch:
        return 0.0
    if epoch >= end_epoch:
        return float(max_weight)
    return float(max_weight) * float(epoch - start_epoch) / float(end_epoch - start_epoch)


def soft_topk_coverage_loss(final_nez_logit: torch.Tensor, labels_ez: torch.Tensor, channel_mask: torch.Tensor, *, tau_rank: float = 0.10, tau_topk: float = 0.25) -> torch.Tensor:
    if tau_rank <= 0 or tau_topk <= 0:
        raise ValueError("soft Top-K temperatures must be positive")
    losses = []
    for index in range(final_nez_logit.shape[0]):
        valid = channel_mask[index].bool()
        y_ez = labels_ez[index, valid].to(final_nez_logit.dtype)
        k = int((y_ez > 0.5).sum().item())
        if k <= 0:
            continue
        score = -final_nez_logit[index, valid]
        # score[j] > score[i] means j ranks ahead of i.
        pairwise = (score.unsqueeze(0) - score.unsqueeze(1)) / tau_rank
        pairwise = pairwise.masked_fill(torch.eye(score.numel(), dtype=torch.bool, device=score.device), 0.0)
        soft_rank = 1.0 + torch.sigmoid(pairwise).sum(dim=1) - 0.5
        membership = torch.sigmoid((float(k) + 0.5 - soft_rank) / tau_topk)
        losses.append(1.0 - (y_ez * membership).sum() / float(k))
    return torch.stack(losses).mean() if losses else final_nez_logit.sum() * 0.0


def rcc_loss_components(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], patient_balanced_bce: torch.Tensor, args: Any, epoch: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    from .v3_rcc_profiles import get_v3_rcc_profile

    profile = get_v3_rcc_profile(getattr(args, "v3_rcc_profile", "R0_BASE"))
    logits = outputs["logits"]
    mean_bce = patient_mean_unweighted_bce(logits, batch["labels"], batch["channel_mask"])
    balanced_weight = float(getattr(args, "v3_rcc_balanced_bce_weight", 0.75)) if profile.use_hybrid_bce else 1.0
    unweighted_weight = float(getattr(args, "v3_rcc_unweighted_bce_weight", 0.25)) if profile.use_hybrid_bce else 0.0
    classification = balanced_weight * patient_balanced_bce + unweighted_weight * mean_bce
    coverage = soft_topk_coverage_loss(
        outputs["final_nez_logit"], batch["labels_ez"], batch["channel_mask"],
        tau_rank=float(getattr(args, "v3_rcc_soft_rank_tau", 0.10)),
        tau_topk=float(getattr(args, "v3_rcc_soft_topk_tau", 0.25)),
    ) if profile.use_coverage else logits.sum() * 0.0
    effective = linear_ramp_weight(
        int(epoch), int(getattr(args, "v3_rcc_coverage_start_epoch", 3)),
        int(getattr(args, "v3_rcc_coverage_end_epoch", 10)),
        float(getattr(args, "v3_rcc_coverage_weight", 0.03)),
    ) if profile.use_coverage else 0.0
    return classification + effective * coverage, {
        "patient_balanced_bce": patient_balanced_bce,
        "patient_mean_unweighted_bce": mean_bce,
        "classification_loss": classification,
        "soft_topk_coverage_loss": coverage,
        "coverage_weight_effective": logits.new_tensor(effective),
    }


__all__ = ["patient_mean_unweighted_bce", "linear_ramp_weight", "soft_topk_coverage_loss", "rcc_loss_components"]
