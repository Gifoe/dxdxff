"""Locked objective-only variants for the matched R0 inference architecture."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def per_patient_weighted_bce(
    logits: torch.Tensor, labels_nez: torch.Tensor, labels_ez: torch.Tensor, channel_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """R0 EZ=2/NEZ=1 weighted BCE, normalized separately within each patient."""
    if logits.shape != labels_nez.shape or logits.shape != labels_ez.shape or logits.shape != channel_mask.shape:
        raise ValueError("Patient BCE tensor shapes disagree")
    valid = channel_mask.bool() & (labels_nez >= 0) & (labels_ez >= 0)
    safe_labels = torch.where(valid, labels_nez, torch.zeros_like(labels_nez))
    channel_loss = F.binary_cross_entropy_with_logits(logits, safe_labels, reduction="none")
    weights = torch.where(labels_ez > 0.5, 2.0, 1.0).to(logits.dtype) * valid.to(logits.dtype)
    denominator = weights.sum(dim=1)
    active = denominator > 0
    per_patient = (channel_loss * weights).sum(dim=1) / denominator.clamp_min(1e-6)
    return per_patient, active


def patient_equal_weighted_bce_loss(
    logits: torch.Tensor, labels_nez: torch.Tensor, labels_ez: torch.Tensor, channel_mask: torch.Tensor
) -> torch.Tensor:
    per_patient, active = per_patient_weighted_bce(logits, labels_nez, labels_ez, channel_mask)
    if not torch.any(active):
        return logits.sum() * 0.0
    return per_patient[active].mean()


def patient_soft_macro_f1_loss(
    logits: torch.Tensor,
    labels_nez: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable, patient-equal symmetric F1 on the same final NEZ logits.

    A missing true class gets F1=0, matching the locked zero-division convention.
    The class-support audit is expected to find both classes in active patients.
    """
    if logits.shape != labels_nez.shape or logits.shape != labels_ez.shape or logits.shape != channel_mask.shape:
        raise ValueError("Patient soft-F1 tensor shapes disagree")
    valid = channel_mask.bool() & (labels_nez >= 0) & (labels_ez >= 0)
    active = valid.any(dim=1)
    if not torch.any(active):
        zero = logits.sum() * 0.0
        return zero, zero, zero
    y_nez = torch.where(valid, labels_nez, torch.zeros_like(labels_nez))
    y_ez = torch.where(valid, labels_ez, torch.zeros_like(labels_ez))
    p_nez = torch.sigmoid(logits) * valid.to(logits.dtype)
    p_ez = (1.0 - torch.sigmoid(logits)) * valid.to(logits.dtype)
    soft_nez = (2.0 * (p_nez * y_nez).sum(dim=1) + eps) / (p_nez.sum(dim=1) + y_nez.sum(dim=1) + eps)
    soft_ez = (2.0 * (p_ez * y_ez).sum(dim=1) + eps) / (p_ez.sum(dim=1) + y_ez.sum(dim=1) + eps)
    soft_nez = torch.where(y_nez.sum(dim=1) > 0, soft_nez, torch.zeros_like(soft_nez))
    soft_ez = torch.where(y_ez.sum(dim=1) > 0, soft_ez, torch.zeros_like(soft_ez))
    return 1.0 - (0.5 * (soft_nez[active] + soft_ez[active])).mean(), soft_nez[active].mean(), soft_ez[active].mean()


def beta_at_epoch(epoch: int) -> float:
    if not 1 <= epoch <= 30:
        raise ValueError("A2 epoch must be 1..30")
    if epoch <= 6:
        return 0.0
    if epoch <= 10:
        return 0.025 * (epoch - 6)
    return 0.10
