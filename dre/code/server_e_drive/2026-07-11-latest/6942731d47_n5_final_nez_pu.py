"""Final N5F patient-conditional selection-aware nnPU objective.

The cache keeps EZ-positive labels.  This module treats ``labels_ez == 0`` as
clinically clean NEZ positives (P) and ``labels_ez == 1`` as the contaminated
observed-EZ / unlabeled set (U).  Model logits always have NEZ semantics.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn.functional as F


def _zero(logits: torch.Tensor) -> torch.Tensor:
    return logits.sum() * 0.0


def build_patient_context(
    patient_channel_embedding: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    """Concatenate masked channel mean and standard deviation per patient."""
    if patient_channel_embedding.ndim != 3 or channel_mask.ndim != 2:
        raise ValueError("patient embeddings must be [B,C,D] and channel_mask must be [B,C]")
    mask = channel_mask.to(device=patient_channel_embedding.device, dtype=patient_channel_embedding.dtype)
    mask_expanded = mask.unsqueeze(-1)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (patient_channel_embedding * mask_expanded).sum(dim=1) / count
    centered = (patient_channel_embedding - mean.unsqueeze(1)) * mask_expanded
    variance = centered.square().sum(dim=1) / count
    std = torch.sqrt(variance.clamp_min(1e-8))
    context = torch.cat([mean, std], dim=-1)
    patient_valid = channel_mask.to(device=patient_channel_embedding.device, dtype=torch.bool).any(dim=1)
    return context * patient_valid.to(dtype=context.dtype).unsqueeze(-1)


def bounded_patient_prior(
    prior_raw: torch.Tensor,
    *,
    pi_min: float,
    pi_max: float,
) -> torch.Tensor:
    """Map unconstrained patient logits to the configured hidden-NEZ interval."""
    if not 0.0 <= float(pi_min) < float(pi_max) <= 1.0:
        raise ValueError("N5F requires 0 <= pi_min < pi_max <= 1")
    return float(pi_min) + (float(pi_max) - float(pi_min)) * torch.sigmoid(prior_raw)


def selection_propensity_loss(
    propensity: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Patient-balanced propensity BCE with explicit single-class handling."""
    valid = channel_mask.to(device=propensity.device, dtype=torch.bool) & (
        labels_ez.to(device=propensity.device) >= 0.0
    )
    labels = labels_ez.to(device=propensity.device, dtype=propensity.dtype)
    patient_losses: list[torch.Tensor] = []
    single_class_count = 0
    for patient_idx in range(propensity.shape[0]):
        clean_nez = valid[patient_idx] & (labels[patient_idx] <= 0.5)
        unlabeled = valid[patient_idx] & (labels[patient_idx] > 0.5)
        terms: list[torch.Tensor] = []
        if bool(clean_nez.any()):
            terms.append(F.binary_cross_entropy(
                propensity[patient_idx][clean_nez].clamp(1e-6, 1.0 - 1e-6),
                torch.ones_like(propensity[patient_idx][clean_nez]),
            ))
        if bool(unlabeled.any()):
            terms.append(F.binary_cross_entropy(
                propensity[patient_idx][unlabeled].clamp(1e-6, 1.0 - 1e-6),
                torch.zeros_like(propensity[patient_idx][unlabeled]),
            ))
        if not terms:
            continue
        if len(terms) == 1:
            single_class_count += 1
            patient_losses.append(terms[0])
        else:
            patient_losses.append(0.5 * (terms[0] + terms[1]))
    loss = torch.stack(patient_losses).mean() if patient_losses else _zero(propensity)
    valid_values = propensity[valid]
    return loss, {
        "n5f_propensity_loss": float(loss.detach().cpu()),
        "n5f_propensity_mean": float(valid_values.mean().detach().cpu()) if valid_values.numel() else 0.0,
        "n5f_propensity_min": float(valid_values.min().detach().cpu()) if valid_values.numel() else 0.0,
        "n5f_propensity_max": float(valid_values.max().detach().cpu()) if valid_values.numel() else 0.0,
        "n5f_propensity_single_class_patient_count": float(single_class_count),
    }


def selection_inverse_weights(
    propensity: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    w_max: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Create clipped, patient-normalized inverse propensity weights for P."""
    if float(w_max) < 1.0:
        raise ValueError("N5F propensity_w_max must be at least 1")
    valid = channel_mask.to(device=propensity.device, dtype=torch.bool) & (
        labels_ez.to(device=propensity.device) >= 0.0
    )
    clean_nez = valid & (labels_ez.to(device=propensity.device) <= 0.5)
    raw = torch.reciprocal(propensity.detach().clamp_min(1e-6)).clamp(1.0, float(w_max))
    weights = torch.zeros_like(propensity)
    for patient_idx in range(propensity.shape[0]):
        patient_positive = clean_nez[patient_idx]
        if not bool(patient_positive.any()):
            continue
        patient_weights = raw[patient_idx][patient_positive]
        normalized = patient_weights / patient_weights.mean().clamp_min(1e-6)
        weights[patient_idx][patient_positive] = normalized
    positive_weights = weights[clean_nez]
    return weights.detach(), {
        "n5f_ipw_mean": float(positive_weights.mean().detach().cpu()) if positive_weights.numel() else 0.0,
        "n5f_ipw_max": float(positive_weights.max().detach().cpu()) if positive_weights.numel() else 0.0,
    }


def patient_conditional_nnpu_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    pi: torch.Tensor,
    positive_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute patient-balanced selection-aware non-negative PU risk."""
    labels = labels_ez.to(device=logits.device, dtype=logits.dtype)
    valid = channel_mask.to(device=logits.device, dtype=torch.bool) & (labels >= 0.0)
    weights = positive_weights.to(device=logits.device, dtype=logits.dtype).detach()
    pi_values = pi.to(device=logits.device, dtype=logits.dtype).reshape(-1)
    patient_losses: list[torch.Tensor] = []
    positive_risks: list[torch.Tensor] = []
    positive_negative_risks: list[torch.Tensor] = []
    unlabeled_negative_risks: list[torch.Tensor] = []
    raw_negative_risks: list[torch.Tensor] = []
    corrected_count = 0
    skipped_no_positive = 0
    skipped_no_unlabeled = 0
    for patient_idx in range(logits.shape[0]):
        clean_nez = valid[patient_idx] & (labels[patient_idx] <= 0.5)
        unlabeled = valid[patient_idx] & (labels[patient_idx] > 0.5)
        if not bool(clean_nez.any()):
            skipped_no_positive += 1
            continue
        if not bool(unlabeled.any()):
            skipped_no_unlabeled += 1
            continue
        z_positive = logits[patient_idx][clean_nez]
        z_unlabeled = logits[patient_idx][unlabeled]
        patient_weights = weights[patient_idx][clean_nez]
        denominator = patient_weights.sum().clamp_min(1e-6)
        positive_loss = F.binary_cross_entropy_with_logits(
            z_positive, torch.ones_like(z_positive), reduction="none"
        )
        positive_as_negative = F.binary_cross_entropy_with_logits(
            z_positive, torch.zeros_like(z_positive), reduction="none"
        )
        unlabeled_as_negative = F.binary_cross_entropy_with_logits(
            z_unlabeled, torch.zeros_like(z_unlabeled), reduction="mean"
        )
        risk_positive = (patient_weights * positive_loss).sum() / denominator
        risk_positive_as_negative = (patient_weights * positive_as_negative).sum() / denominator
        raw_negative = unlabeled_as_negative - pi_values[patient_idx] * risk_positive_as_negative
        corrected_count += int(float(raw_negative.detach().cpu()) < 0.0)
        patient_loss = pi_values[patient_idx] * risk_positive + F.relu(raw_negative)
        patient_losses.append(patient_loss)
        positive_risks.append(risk_positive)
        positive_negative_risks.append(risk_positive_as_negative)
        unlabeled_negative_risks.append(unlabeled_as_negative)
        raw_negative_risks.append(raw_negative)
    loss = torch.stack(patient_losses).mean() if patient_losses else _zero(logits)

    def mean_value(values: list[torch.Tensor]) -> float:
        return float(torch.stack(values).mean().detach().cpu()) if values else 0.0

    n_valid = len(patient_losses)
    return loss, {
        "n5f_nnpu_loss": float(loss.detach().cpu()),
        "n5f_positive_risk": mean_value(positive_risks),
        "n5f_positive_as_negative_risk": mean_value(positive_negative_risks),
        "n5f_unlabeled_negative_risk": mean_value(unlabeled_negative_risks),
        "n5f_raw_negative_risk": mean_value(raw_negative_risks),
        "n5f_nonnegative_correction_rate": float(corrected_count / max(n_valid, 1)),
        "n5f_valid_pu_patient_count": float(n_valid),
        "n5f_skipped_no_positive_patient_count": float(skipped_no_positive),
        "n5f_skipped_no_unlabeled_patient_count": float(skipped_no_unlabeled),
    }


def patient_latent_soft_rank_loss(
    logits: torch.Tensor,
    labels_ez: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    margin: float,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Patient-wise soft ranking with a fixed |P|*|U| denominator."""
    labels = labels_ez.to(device=logits.device, dtype=logits.dtype)
    valid = channel_mask.to(device=logits.device, dtype=torch.bool) & (labels >= 0.0)
    patient_losses: list[torch.Tensor] = []
    latent_masses: list[torch.Tensor] = []
    pair_count = 0
    for patient_idx in range(logits.shape[0]):
        clean_nez = valid[patient_idx] & (labels[patient_idx] <= 0.5)
        unlabeled = valid[patient_idx] & (labels[patient_idx] > 0.5)
        if not bool(clean_nez.any()) or not bool(unlabeled.any()):
            continue
        z_positive = logits[patient_idx][clean_nez].reshape(-1, 1)
        z_unlabeled = logits[patient_idx][unlabeled].reshape(1, -1)
        latent_true_ez = (1.0 - torch.sigmoid(z_unlabeled)).detach()
        pair_terms = latent_true_ez * F.softplus(float(margin) - z_positive + z_unlabeled)
        denominator = max(1, int(z_positive.shape[0] * z_unlabeled.shape[1]))
        patient_losses.append(pair_terms.sum() / float(denominator))
        latent_masses.append(latent_true_ez.mean())
        pair_count += denominator
    loss = torch.stack(patient_losses).mean() if patient_losses else _zero(logits)
    return loss, {
        "n5f_rank_loss": float(loss.detach().cpu()),
        "n5f_rank_pair_count": float(pair_count),
        "n5f_rank_patient_count": float(len(patient_losses)),
        "n5f_latent_ez_mass_mean": (
            float(torch.stack(latent_masses).mean().detach().cpu()) if latent_masses else 0.0
        ),
    }


def cross_seizure_consistency_loss(
    seizure_logits: torch.Tensor,
    seizure_mask: torch.Tensor,
    seizure_channel_mask: torch.Tensor,
    channel_mask: torch.Tensor,
    *,
    min_seizures: int,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Penalize cross-seizure variance of patient-relative channel rankings."""
    if int(min_seizures) < 2:
        raise ValueError("N5F consistency_min_seizures must be at least 2")
    if seizure_logits.ndim != 3:
        raise ValueError("seizure_logits must have shape [B,S,C]")
    valid = (
        seizure_mask.to(device=seizure_logits.device, dtype=torch.bool).unsqueeze(-1)
        & seizure_channel_mask.to(device=seizure_logits.device, dtype=torch.bool)
        & channel_mask.to(device=seizure_logits.device, dtype=torch.bool).unsqueeze(1)
    )
    patient_losses: list[torch.Tensor] = []
    eligible_channel_count = 0
    skipped_patient_count = 0
    for patient_idx in range(seizure_logits.shape[0]):
        patient_valid = valid[patient_idx]
        eligible_seizure = patient_valid.sum(dim=1) >= 2
        effective = patient_valid & eligible_seizure.unsqueeze(-1)
        count_per_seizure = effective.sum(dim=1, keepdim=True).to(dtype=seizure_logits.dtype).clamp_min(1.0)
        masked_logits = seizure_logits[patient_idx].masked_fill(~effective, 0.0)
        mean = masked_logits.sum(dim=1, keepdim=True) / count_per_seizure
        centered = (seizure_logits[patient_idx] - mean).masked_fill(~effective, 0.0)
        variance = centered.square().sum(dim=1, keepdim=True) / count_per_seizure
        zscore = centered / torch.sqrt(variance.clamp_min(1e-5))
        observations = effective.sum(dim=0)
        eligible_channel = observations >= int(min_seizures)
        if not bool(eligible_channel.any()):
            skipped_patient_count += 1
            continue
        channel_count = observations.to(dtype=seizure_logits.dtype).clamp_min(1.0)
        channel_mean = (zscore * effective.to(dtype=zscore.dtype)).sum(dim=0) / channel_count
        channel_variance = (
            (zscore - channel_mean.unsqueeze(0)).square()
            * effective.to(dtype=zscore.dtype)
        ).sum(dim=0) / channel_count
        patient_losses.append(channel_variance[eligible_channel].mean())
        eligible_channel_count += int(eligible_channel.sum().detach().cpu())
    loss = torch.stack(patient_losses).mean() if patient_losses else _zero(seizure_logits)
    return loss, {
        "n5f_consistency_loss": float(loss.detach().cpu()),
        "n5f_consistency_patient_count": float(len(patient_losses)),
        "n5f_consistency_channel_count": float(eligible_channel_count),
        "n5f_consistency_skipped_patient_count": float(skipped_patient_count),
    }


def compute_n5_final_loss(
    outputs: Mapping[str, torch.Tensor],
    batch: Mapping[str, Any],
    args: Any,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute the complete N5F objective and diagnostics."""
    logits = outputs["logits"]
    labels_ez = batch["labels_ez"].to(device=logits.device, dtype=logits.dtype)
    channel_mask = batch["channel_mask"].to(device=logits.device, dtype=torch.bool)
    pi = outputs["n5f_pi"]
    propensity = outputs["n5f_propensity"]
    patient_valid = channel_mask.any(dim=1)

    prior_anchor = float(getattr(args, "n5f_pi_anchor", 0.15))
    prior_values = pi.reshape(-1)[patient_valid]
    prior_loss = (
        (prior_values - prior_anchor).square().mean()
        if prior_values.numel()
        else _zero(logits)
    )
    propensity_loss, propensity_parts = selection_propensity_loss(
        propensity, labels_ez, channel_mask
    )
    positive_weights, ipw_parts = selection_inverse_weights(
        propensity,
        labels_ez,
        channel_mask,
        w_max=float(getattr(args, "n5f_propensity_w_max", 5.0)),
    )
    nnpu_loss, nnpu_parts = patient_conditional_nnpu_loss(
        logits, labels_ez, channel_mask, pi, positive_weights
    )
    rank_loss, rank_parts = patient_latent_soft_rank_loss(
        logits,
        labels_ez,
        channel_mask,
        margin=float(getattr(args, "n5f_rank_margin", 0.10)),
    )
    consistency_loss, consistency_parts = cross_seizure_consistency_loss(
        outputs["n5f_seizure_logits"],
        batch["seizure_mask"],
        batch["seizure_channel_mask"],
        channel_mask,
        min_seizures=int(getattr(args, "n5f_consistency_min_seizures", 2)),
    )
    total = (
        nnpu_loss
        + float(getattr(args, "n5f_propensity_loss_weight", 0.10)) * propensity_loss
        + float(getattr(args, "n5f_prior_anchor_weight", 0.05)) * prior_loss
        + float(getattr(args, "n5f_rank_loss_weight", 0.05)) * rank_loss
        + float(getattr(args, "n5f_consistency_loss_weight", 0.05)) * consistency_loss
    )
    diagnostics = {
        **propensity_parts,
        **ipw_parts,
        **nnpu_parts,
        **rank_parts,
        **consistency_parts,
        "n5f_pi_mean": float(prior_values.mean().detach().cpu()) if prior_values.numel() else 0.0,
        "n5f_pi_std": float(prior_values.std(unbiased=False).detach().cpu()) if prior_values.numel() else 0.0,
        "n5f_pi_min": float(prior_values.min().detach().cpu()) if prior_values.numel() else 0.0,
        "n5f_pi_max": float(prior_values.max().detach().cpu()) if prior_values.numel() else 0.0,
        "n5f_pi_anchor_loss": float(prior_loss.detach().cpu()),
        "n5f_prior_valid_patient_count": float(patient_valid.sum().detach().cpu()),
        "n5f_total_loss": float(total.detach().cpu()),
    }
    return total, diagnostics


__all__ = [
    "bounded_patient_prior",
    "build_patient_context",
    "compute_n5_final_loss",
    "cross_seizure_consistency_loss",
    "patient_conditional_nnpu_loss",
    "patient_latent_soft_rank_loss",
    "selection_inverse_weights",
    "selection_propensity_loss",
]
