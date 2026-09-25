"""N6 dual-view clean-NEZ anchored EMA robust objectives."""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def ema_observed_ez_reliability(
    teacher_logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    epoch: int,
    warmup_epochs: int,
    noise_discount: float,
    reliability_min: float,
) -> torch.Tensor:
    """Return detached reliability for observed-EZ (U) channels only."""
    valid_u = channel_mask.to(device=teacher_logits.device, dtype=torch.bool) & (labels_ez.to(teacher_logits.device) > 0.5)
    reliability = torch.ones_like(teacher_logits)
    if int(epoch) > int(warmup_epochs):
        discounted = 1.0 - float(noise_discount) * torch.sigmoid(teacher_logits)
        reliability = torch.where(valid_u, discounted.clamp(float(reliability_min), 1.0), reliability)
    return reliability.detach()


def compute_patient_clean_nez_robust_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    reliability: torch.Tensor,
    reliability_min: float = 0.50,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Patient-balanced clean-positive/full-weight and U/fixed-denominator BCE."""
    labels = labels_ez.to(device=logits.device, dtype=logits.dtype)
    valid = channel_mask.to(device=logits.device, dtype=torch.bool) & (labels >= 0.0)
    rel = reliability.to(device=logits.device, dtype=logits.dtype).detach()
    patient_losses: list[torch.Tensor] = []
    clean_losses: list[torch.Tensor] = []
    observed_losses: list[torch.Tensor] = []
    single_class = 0
    clean_count = 0
    observed_count = 0
    for patient_idx in range(logits.shape[0]):
        clean = valid[patient_idx] & (labels[patient_idx] <= 0.5)
        observed = valid[patient_idx] & (labels[patient_idx] > 0.5)
        pieces: list[torch.Tensor] = []
        if bool(clean.any()):
            clean_loss = F.binary_cross_entropy_with_logits(logits[patient_idx][clean], torch.ones_like(logits[patient_idx][clean]))
            pieces.append(clean_loss)
            clean_losses.append(clean_loss)
            clean_count += int(clean.sum())
        if bool(observed.any()):
            terms = F.binary_cross_entropy_with_logits(logits[patient_idx][observed], torch.zeros_like(logits[patient_idx][observed]), reduction="none")
            observed_loss = (rel[patient_idx][observed] * terms).sum() / float(int(observed.sum()))
            pieces.append(observed_loss)
            observed_losses.append(observed_loss)
            observed_count += int(observed.sum())
        if len(pieces) == 1:
            single_class += 1
        if pieces:
            patient_losses.append(0.5 * (pieces[0] + pieces[1]) if len(pieces) == 2 else pieces[0])
    loss = torch.stack(patient_losses).mean() if patient_losses else _zero(logits)
    observed_values = rel[valid & (labels > 0.5)]
    return loss, {
        "n6_clean_nez_loss": float(torch.stack(clean_losses).mean().detach().cpu()) if clean_losses else 0.0,
        "n6_observed_ez_loss": float(torch.stack(observed_losses).mean().detach().cpu()) if observed_losses else 0.0,
        "n6_classification_loss": float(loss.detach().cpu()),
        "n6_mean_reliability": float(observed_values.mean().detach().cpu()) if observed_values.numel() else 1.0,
        "n6_min_reliability": float(observed_values.min().detach().cpu()) if observed_values.numel() else 1.0,
        "n6_max_reliability": float(observed_values.max().detach().cpu()) if observed_values.numel() else 1.0,
        "n6_reliability_floor_rate": float((observed_values <= observed_values.new_tensor(float(reliability_min) + 1e-7)).float().mean().detach().cpu()) if observed_values.numel() else 0.0,
        "n6_clean_nez_channel_count": float(clean_count),
        "n6_observed_ez_channel_count": float(observed_count),
        "n6_single_class_patient_count": float(single_class),
        "n6_valid_patient_count": float(len(patient_losses)),
    }


def compute_patient_ema_rank_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    reliability: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Patient-wise rank risk with the fixed ``|P| * |U|`` denominator."""
    labels = labels_ez.to(device=logits.device, dtype=logits.dtype)
    valid = channel_mask.to(device=logits.device, dtype=torch.bool) & (labels >= 0.0)
    rel = reliability.to(device=logits.device, dtype=logits.dtype).detach()
    losses: list[torch.Tensor] = []
    pair_count = 0
    for patient_idx in range(logits.shape[0]):
        clean = valid[patient_idx] & (labels[patient_idx] <= 0.5)
        observed = valid[patient_idx] & (labels[patient_idx] > 0.5)
        if not bool(clean.any()) or not bool(observed.any()):
            continue
        z_clean = logits[patient_idx][clean].reshape(-1, 1)
        z_observed = logits[patient_idx][observed].reshape(1, -1)
        weights = rel[patient_idx][observed].reshape(1, -1)
        denominator = int(z_clean.shape[0] * z_observed.shape[1])
        losses.append((weights * F.softplus(float(margin) - z_clean + z_observed)).sum() / float(max(denominator, 1)))
        pair_count += denominator
    loss = torch.stack(losses).mean() if losses else _zero(logits)
    return loss, {"n6_rank_loss": float(loss.detach().cpu()), "n6_rank_pair_count": float(pair_count), "n6_rank_patient_count": float(len(losses))}


def compute_n6_total_loss(
    student_outputs: Mapping[str, torch.Tensor],
    teacher_outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    args: Any,
    *,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the sole N6 objective without legacy BCE/PU terms."""
    logits = student_outputs["logits"]
    labels_ez = batch["labels_ez"].to(device=logits.device, dtype=logits.dtype)
    channel_mask = batch["channel_mask"].to(device=logits.device, dtype=torch.bool)
    reliability = ema_observed_ez_reliability(
        teacher_outputs["logits"], labels_ez, channel_mask,
        epoch=epoch,
        warmup_epochs=int(getattr(args, "n6_warmup_epochs", 5)),
        noise_discount=float(getattr(args, "n6_noise_discount", 0.50)),
        reliability_min=float(getattr(args, "n6_reliability_min", 0.50)),
    )
    reliability_min = float(getattr(args, "n6_reliability_min", 0.50))
    fused_loss, diagnostics = compute_patient_clean_nez_robust_loss(logits, labels_ez, channel_mask, reliability, reliability_min)
    feature_loss, _ = compute_patient_clean_nez_robust_loss(student_outputs["logits_feature"], labels_ez, channel_mask, reliability, reliability_min)
    raw_available = student_outputs["raw_channel_available"].to(device=logits.device, dtype=torch.bool)
    raw_loss, _ = compute_patient_clean_nez_robust_loss(student_outputs["logits_raw"], labels_ez, channel_mask & raw_available, reliability, reliability_min)
    rank_loss, rank_diag = compute_patient_ema_rank_loss(logits, labels_ez, channel_mask, reliability, margin=float(getattr(args, "n6_rank_margin", 0.10)))
    gate = student_outputs["fusion_gate_feature"]
    gate_mask = channel_mask & raw_available
    gate_loss = ((gate[gate_mask] - float(getattr(args, "n6_gate_anchor", 0.70))).square().mean() if bool(gate_mask.any()) else _zero(logits))
    rank_weight = 0.0 if int(epoch) <= int(getattr(args, "n6_warmup_epochs", 5)) else float(getattr(args, "n6_rank_loss_weight", 0.05))
    total = (
        fused_loss
        + float(getattr(args, "n6_feature_aux_weight", 0.15)) * feature_loss
        + float(getattr(args, "n6_raw_aux_weight", 0.15)) * raw_loss
        + rank_weight * rank_loss
        + float(getattr(args, "n6_gate_loss_weight", 0.005)) * gate_loss
    )
    diagnostics.update(rank_diag)
    diagnostics.update({
        "n6_total_loss": float(total.detach().cpu()),
        "n6_feature_aux_loss": float(feature_loss.detach().cpu()),
        "n6_raw_aux_loss": float(raw_loss.detach().cpu()),
        "n6_gate_loss": float(gate_loss.detach().cpu()),
        "n6_rank_weight_effective": float(rank_weight),
        "n6_fusion_gate_feature_mean": float(gate[channel_mask].mean().detach().cpu()) if bool(channel_mask.any()) else 1.0,
        "n6_fusion_gate_feature_std": float(gate[channel_mask].std(unbiased=False).detach().cpu()) if bool(channel_mask.any()) else 0.0,
        "n6_fusion_gate_feature_min": float(gate[channel_mask].min().detach().cpu()) if bool(channel_mask.any()) else 1.0,
        "n6_fusion_gate_feature_max": float(gate[channel_mask].max().detach().cpu()) if bool(channel_mask.any()) else 1.0,
        "n6_raw_channel_available_rate": float(raw_available[channel_mask].float().mean().detach().cpu()) if bool(channel_mask.any()) else 0.0,
    })
    return total, diagnostics


__all__ = ["compute_n6_total_loss", "compute_patient_clean_nez_robust_loss", "compute_patient_ema_rank_loss", "ema_observed_ez_reliability"]
