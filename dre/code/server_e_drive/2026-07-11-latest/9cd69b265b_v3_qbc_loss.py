"""Patient-equal boundary and soft Top-K coverage losses for V3-QBC."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F

from .v3_qbc_profiles import get_v3_qbc_profile


def boundary_hard_ranking_loss(
    final_nez_logit: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    hard_positive_fraction: float = 0.30,
    hard_negative_multiplier: float = 1.0,
    hard_negative_cap: int = 16,
    margin: float = 0.05,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    ez_score = -final_nez_logit
    patient_losses = []
    positive_scores = []
    negative_scores = []
    for patient_idx in range(ez_score.shape[0]):
        valid = channel_mask[patient_idx].bool()
        ez = valid & (labels_ez[patient_idx] > 0.5)
        nez = valid & ~ez
        if not torch.any(ez) or not torch.any(nez):
            continue
        n_ez = int(ez.sum().item())
        n_nez = int(nez.sum().item())
        n_pos = max(1, int(math.ceil(n_ez * float(hard_positive_fraction))))
        n_neg = min(
            n_nez,
            max(1, min(int(hard_negative_cap), int(math.ceil(n_ez * float(hard_negative_multiplier))))),
        )
        hard_pos = torch.topk(ez_score[patient_idx, ez], k=n_pos, largest=False).values.mean()
        hard_neg = torch.topk(ez_score[patient_idx, nez], k=n_neg, largest=True).values.mean()
        patient_losses.append(F.softplus(float(margin) + hard_neg - hard_pos))
        positive_scores.append(hard_pos)
        negative_scores.append(hard_neg)
    zero = final_nez_logit.sum() * 0.0
    loss = torch.stack(patient_losses).mean() if patient_losses else zero
    mean_pos = torch.stack(positive_scores).mean() if positive_scores else zero
    mean_neg = torch.stack(negative_scores).mean() if negative_scores else zero
    return loss, {
        "boundary_loss": loss,
        "mean_hard_positive_score": mean_pos,
        "mean_hard_negative_score": mean_neg,
        "boundary_margin_observed": mean_pos - mean_neg,
        "n_boundary_valid_patients": final_nez_logit.new_tensor(float(len(patient_losses))),
    }


def soft_topk_coverage_loss(
    final_nez_logit: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    tau_rank: float = 0.10,
    tau_topk: float = 0.25,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if tau_rank <= 0 or tau_topk <= 0:
        raise ValueError("tau_rank and tau_topk must be positive")
    patient_losses = []
    coverages = []
    for patient_idx in range(final_nez_logit.shape[0]):
        valid = channel_mask[patient_idx].bool()
        y_ez = labels_ez[patient_idx, valid].to(final_nez_logit.dtype)
        k = int((y_ez > 0.5).sum().item())
        if k <= 0 or y_ez.numel() == 0:
            continue
        score = -final_nez_logit[patient_idx, valid]
        pairwise = (score.unsqueeze(0) - score.unsqueeze(1)) / float(tau_rank)
        diagonal_mask = ~torch.eye(score.numel(), dtype=torch.bool, device=score.device)
        outrank = torch.sigmoid(pairwise) * diagonal_mask.to(dtype=score.dtype)
        soft_rank = 1.0 + outrank.sum(dim=1)
        membership = torch.sigmoid((float(k) + 0.5 - soft_rank) / float(tau_topk))
        coverage = (y_ez * membership).sum() / float(k)
        coverages.append(coverage)
        patient_losses.append(1.0 - coverage)
    zero = final_nez_logit.sum() * 0.0
    loss = torch.stack(patient_losses).mean() if patient_losses else zero
    coverage = torch.stack(coverages).mean() if coverages else zero
    return loss, {
        "soft_topk_coverage_loss": loss,
        "soft_topk_mean_coverage": coverage,
        "n_coverage_valid_patients": final_nez_logit.new_tensor(float(len(patient_losses))),
    }


def compute_v3_qbc_loss(
    outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], args: Any
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    final_nez_logit = outputs["final_nez_logit"]
    profile = get_v3_qbc_profile(getattr(args, "v3_qbc_profile", "BCR_BC_ONLY"))
    zero = final_nez_logit.sum() * 0.0

    # BCE is applied by the caller.  These profile-gated terms are auxiliary
    # only, so an ablation never silently drops channel-level supervision.
    if profile.use_boundary:
        boundary, boundary_diag = boundary_hard_ranking_loss(
            final_nez_logit,
            batch["labels_ez"],
            batch["channel_mask"],
            hard_positive_fraction=float(getattr(args, "v3_qbc_hard_positive_fraction", 0.30)),
            hard_negative_multiplier=float(getattr(args, "v3_qbc_hard_negative_multiplier", 1.0)),
            hard_negative_cap=int(getattr(args, "v3_qbc_hard_negative_cap", 16)),
            margin=float(getattr(args, "v3_qbc_boundary_margin", 0.05)),
        )
    else:
        boundary = zero
        boundary_diag = {
            "boundary_loss": zero,
            "mean_hard_positive_score": zero,
            "mean_hard_negative_score": zero,
            "boundary_margin_observed": zero,
            "n_boundary_valid_patients": zero,
        }
    if profile.use_coverage:
        coverage, coverage_diag = soft_topk_coverage_loss(
            final_nez_logit,
            batch["labels_ez"],
            batch["channel_mask"],
            tau_rank=float(getattr(args, "v3_qbc_soft_rank_tau", 0.10)),
            tau_topk=float(getattr(args, "v3_qbc_soft_topk_tau", 0.25)),
        )
    else:
        coverage = zero
        coverage_diag = {
            "soft_topk_coverage_loss": zero,
            "soft_topk_mean_coverage": zero,
            "n_coverage_valid_patients": zero,
        }
    weighted = (
        float(getattr(args, "v3_qbc_boundary_weight", 0.05)) * boundary
        + float(getattr(args, "v3_qbc_coverage_weight", 0.08)) * coverage
    )
    return weighted, {
        **boundary_diag,
        **coverage_diag,
        "v3_qbc_aux_loss": weighted,
        "boundary_loss_enabled": final_nez_logit.new_tensor(float(profile.use_boundary)),
        "coverage_loss_enabled": final_nez_logit.new_tensor(float(profile.use_coverage)),
    }


__all__ = ["boundary_hard_ranking_loss", "soft_topk_coverage_loss", "compute_v3_qbc_loss"]
