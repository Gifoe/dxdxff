"""Losses for clean-NEZ anchored, count-free set localization."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F


def _zero(reference: torch.Tensor) -> torch.Tensor:
    return reference.sum() * 0.0


def patient_balanced_nez_bce(
    final_nez_logits: torch.Tensor,
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Balance clean-NEZ and observed-EZ BCE inside each patient."""
    patient_losses: list[torch.Tensor] = []
    clean_losses: list[torch.Tensor] = []
    observed_losses: list[torch.Tensor] = []
    for patient_idx in range(final_nez_logits.shape[0]):
        valid = channel_mask[patient_idx].bool() & (labels_nez[patient_idx] >= 0.0)
        clean = valid & (labels_nez[patient_idx] > 0.5)
        observed = valid & (labels_nez[patient_idx] <= 0.5)
        class_losses: list[torch.Tensor] = []
        if torch.any(clean):
            value = F.binary_cross_entropy_with_logits(
                final_nez_logits[patient_idx][clean],
                torch.ones_like(final_nez_logits[patient_idx][clean]),
            )
            clean_losses.append(value)
            class_losses.append(value)
        if torch.any(observed):
            value = F.binary_cross_entropy_with_logits(
                final_nez_logits[patient_idx][observed],
                torch.zeros_like(final_nez_logits[patient_idx][observed]),
            )
            observed_losses.append(value)
            class_losses.append(value)
        if len(class_losses) == 2:
            patient_losses.append(0.5 * class_losses[0] + 0.5 * class_losses[1])
        elif class_losses:
            patient_losses.append(class_losses[0])
    zero = _zero(final_nez_logits)
    return (
        torch.stack(patient_losses).mean() if patient_losses else zero,
        {
            "clean_nez_bce": torch.stack(clean_losses).mean() if clean_losses else zero,
            "observed_ez_bce": torch.stack(observed_losses).mean() if observed_losses else zero,
        },
    )


def patient_soft_macro_f1_loss(
    final_nez_logits: torch.Tensor,
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Differentiable patient-mean macro F1 with symmetric NEZ/EZ terms."""
    nez_f1s: list[torch.Tensor] = []
    ez_f1s: list[torch.Tensor] = []
    for patient_idx in range(final_nez_logits.shape[0]):
        valid = channel_mask[patient_idx].bool() & (labels_nez[patient_idx] >= 0.0)
        if not torch.any(valid):
            continue
        p_nez = torch.sigmoid(final_nez_logits[patient_idx][valid])
        y_nez = labels_nez[patient_idx][valid]
        p_ez = 1.0 - p_nez
        y_ez = 1.0 - y_nez
        nez_f1s.append((2.0 * (p_nez * y_nez).sum() + eps) / (p_nez.sum() + y_nez.sum() + eps))
        ez_f1s.append((2.0 * (p_ez * y_ez).sum() + eps) / (p_ez.sum() + y_ez.sum() + eps))
    zero = _zero(final_nez_logits)
    if not nez_f1s:
        return zero, zero, zero
    mean_nez = torch.stack(nez_f1s).mean()
    mean_ez = torch.stack(ez_f1s).mean()
    return 1.0 - 0.5 * (mean_nez + mean_ez), mean_nez, mean_ez


def clean_nez_prototype_loss(
    anchor_distance: torch.Tensor,
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    normalized_prototypes: torch.Tensor,
    prototype_similarity_margin: float = 0.50,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact clean NEZ only and penalize collapsed prototype directions."""
    clean = channel_mask.bool() & (labels_nez > 0.5)
    compactness = anchor_distance[clean].mean() if torch.any(clean) else _zero(anchor_distance)
    if normalized_prototypes.shape[0] < 2:
        diversity = _zero(normalized_prototypes)
    else:
        similarity = normalized_prototypes @ normalized_prototypes.transpose(0, 1)
        off_diagonal = ~torch.eye(similarity.shape[0], dtype=torch.bool, device=similarity.device)
        diversity = F.relu(similarity[off_diagonal] - float(prototype_similarity_margin)).square().mean()
    return compactness + 0.10 * diversity, compactness, diversity


def _average_tie_percentile_rank(values: torch.Tensor) -> torch.Tensor:
    """Return detached ascending percentile ranks with average ranks for ties."""
    detached = values.detach()
    if detached.numel() <= 1:
        return torch.zeros_like(detached)
    order = torch.argsort(detached, stable=True)
    sorted_values = detached[order]
    sorted_ranks = torch.empty_like(sorted_values)
    start = 0
    denominator = float(detached.numel() - 1)
    while start < detached.numel():
        end = start + 1
        while end < detached.numel() and bool(sorted_values[end] == sorted_values[start]):
            end += 1
        average = 0.5 * (start + end - 1) / denominator
        sorted_ranks[start:end] = average
        start = end
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return ranks.detach()


def select_high_confidence_ez(
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    direct_score_nez: torch.Tensor,
    anchor_nez_evidence: torch.Tensor,
    seizure_nez_probability_mean: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    """Select low-NEZ-likeness observed-EZ channels using detached ranks."""
    selected = torch.zeros_like(channel_mask, dtype=torch.bool)
    for patient_idx in range(labels_nez.shape[0]):
        candidates = torch.nonzero(
            channel_mask[patient_idx].bool() & (labels_nez[patient_idx] >= 0.0) & (labels_nez[patient_idx] <= 0.5),
            as_tuple=False,
        ).flatten()
        if candidates.numel() == 0:
            continue
        ranks = (
            _average_tie_percentile_rank(direct_score_nez[patient_idx, candidates])
            + _average_tie_percentile_rank(anchor_nez_evidence[patient_idx, candidates])
            + _average_tie_percentile_rank(seizure_nez_probability_mean[patient_idx, candidates])
        ) / 3.0
        count = max(1, int(math.ceil(float(fraction) * int(candidates.numel()))))
        order = torch.argsort(ranks, stable=True)
        selected[patient_idx, candidates[order[:count]]] = True
    return selected.detach()


def high_confidence_ez_rank_loss(
    final_nez_logits: torch.Tensor,
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    high_confidence_ez_mask: torch.Tensor,
    margin: float = 0.10,
) -> tuple[torch.Tensor, int]:
    """Rank clean-NEZ logits over detached high-confidence EZ logits."""
    patient_losses: list[torch.Tensor] = []
    for patient_idx in range(final_nez_logits.shape[0]):
        clean = channel_mask[patient_idx].bool() & (labels_nez[patient_idx] > 0.5)
        hc_ez = channel_mask[patient_idx].bool() & high_confidence_ez_mask[patient_idx]
        n_clean = int(clean.sum().item())
        n_hc = int(hc_ez.sum().item())
        if n_clean == 0 or n_hc == 0:
            continue
        pairs = F.softplus(
            float(margin)
            - final_nez_logits[patient_idx][clean][:, None]
            + final_nez_logits[patient_idx][hc_ez][None, :]
        )
        patient_losses.append(pairs.sum() / float(n_clean * n_hc))
    return (
        torch.stack(patient_losses).mean() if patient_losses else _zero(final_nez_logits),
        len(patient_losses),
    )


def beta_binomial_count_nll(
    alpha: torch.Tensor,
    beta: torch.Tensor,
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    """Stable beta-binomial NLL; true counts are consumed only here."""
    valid = channel_mask.bool() & (labels_nez >= 0.0)
    counts = valid.sum(dim=1).to(device=alpha.device, dtype=alpha.dtype)
    successes = ((labels_nez > 0.5) & valid).sum(dim=1).to(device=alpha.device, dtype=alpha.dtype)
    keep = counts > 0
    if not torch.any(keep):
        return _zero(alpha)
    c = counts[keep]
    k = successes[keep]
    a = alpha[keep].clamp_min(1e-6)
    b = beta[keep].clamp_min(1e-6)
    log_choose = torch.lgamma(c + 1.0) - torch.lgamma(k + 1.0) - torch.lgamma(c - k + 1.0)
    log_beta_posterior = torch.lgamma(k + a) + torch.lgamma(c - k + b) - torch.lgamma(c + a + b)
    log_beta_prior = torch.lgamma(a) + torch.lgamma(b) - torch.lgamma(a + b)
    return -(log_choose + log_beta_posterior - log_beta_prior).mean()


def residual_l2_regularizer(
    anchor_residual: torch.Tensor,
    seizure_residual: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    valid = channel_mask.bool()
    if not torch.any(valid):
        return _zero(anchor_residual)
    return (anchor_residual[valid].square() + seizure_residual[valid].square()).mean()


def cane_stage(epoch: int, stage1_end: int = 5, stage2_end: int = 15) -> str:
    if int(epoch) <= int(stage1_end):
        return "stage1_direct_warmup"
    if int(epoch) <= int(stage2_end):
        return "stage2_anchor_evidence_cardinality"
    return "stage3_selective_discrimination"


def compute_cane_set_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    args: Any,
    epoch: int,
) -> tuple[torch.Tensor, dict[str, float | str]]:
    """Compute the staged CANE objective with explicit NEZ-positive semantics."""
    labels_ez = batch["labels_ez"].to(device=outputs["logits"].device, dtype=outputs["logits"].dtype)
    labels_nez = torch.where(labels_ez >= 0.0, 1.0 - labels_ez, torch.full_like(labels_ez, -1.0))
    channel_mask = batch["channel_mask"].to(device=outputs["logits"].device, dtype=torch.bool)
    pbce, bce_parts = patient_balanced_nez_bce(outputs["final_nez_logits"], labels_nez, channel_mask)
    soft_f1, soft_nez_f1, soft_ez_f1 = patient_soft_macro_f1_loss(
        outputs["final_nez_logits"], labels_nez, channel_mask
    )
    anchor, compactness, diversity = clean_nez_prototype_loss(
        outputs["anchor_distance"],
        labels_nez,
        channel_mask,
        outputs["normalized_prototypes"],
        float(getattr(args, "cane_prototype_similarity_margin", 0.50)),
    )
    hc_mask = select_high_confidence_ez(
        labels_nez,
        channel_mask,
        outputs["direct_score_nez"],
        outputs["anchor_nez_evidence"],
        outputs["seizure_nez_probability_mean"],
        float(getattr(args, "cane_hc_ez_fraction", 0.25)),
    )
    rank, valid_rank_patients = high_confidence_ez_rank_loss(
        outputs["final_nez_logits"],
        labels_nez,
        channel_mask,
        hc_mask,
        float(getattr(args, "cane_rank_margin", 0.10)),
    )
    count_nll = beta_binomial_count_nll(
        outputs["cardinality_alpha"], outputs["cardinality_beta"], labels_nez, channel_mask
    )
    residual_l2 = residual_l2_regularizer(
        outputs["anchor_residual"], outputs["seizure_residual"], channel_mask
    )

    stage1_end = int(getattr(args, "cane_stage1_end_epoch", 5))
    stage2_end = int(getattr(args, "cane_stage2_end_epoch", 15))
    stage = cane_stage(epoch, stage1_end, stage2_end)
    after_stage1 = int(epoch) > stage1_end
    rank_active = int(epoch) > int(getattr(args, "cane_rank_start_epoch", 15))
    soft_weight = float(getattr(args, "cane_soft_f1_weight", 0.15))
    anchor_weight = float(getattr(args, "cane_nez_anchor_weight", 0.05)) if after_stage1 else 0.0
    rank_weight = float(getattr(args, "cane_hc_rank_weight", 0.02)) if rank_active else 0.0
    count_weight = float(getattr(args, "cane_count_weight", 0.10)) if after_stage1 else 0.0
    residual_weight = float(getattr(args, "cane_residual_l2_weight", 0.005)) if after_stage1 else 0.0
    total = (
        pbce
        + soft_weight * soft_f1
        + anchor_weight * anchor
        + rank_weight * rank
        + count_weight * count_nll
        + residual_weight * residual_l2
    )
    valid_observed = channel_mask & (labels_nez >= 0.0) & (labels_nez <= 0.5)
    hc_count = int(hc_mask.sum().item()) if rank_active else 0
    observed_count = int(valid_observed.sum().item())
    valid = channel_mask & (labels_nez >= 0.0)
    true_nez_counts = ((labels_nez > 0.5) & valid).sum(dim=1).to(dtype=outputs["expected_nez_count"].dtype)
    utilization = outputs["prototype_utilization"]
    parts: dict[str, float | str] = {
        "cane_total_loss": float(total.detach().cpu()),
        "cane_patient_balanced_bce": float(pbce.detach().cpu()),
        "cane_clean_nez_bce": float(bce_parts["clean_nez_bce"].detach().cpu()),
        "cane_observed_ez_bce": float(bce_parts["observed_ez_bce"].detach().cpu()),
        "cane_soft_macro_f1_loss": float(soft_f1.detach().cpu()),
        "cane_soft_nez_f1": float(soft_nez_f1.detach().cpu()),
        "cane_soft_ez_f1": float(soft_ez_f1.detach().cpu()),
        "cane_anchor_loss": float(anchor.detach().cpu()),
        "cane_anchor_compactness": float(compactness.detach().cpu()),
        "cane_prototype_diversity_loss": float(diversity.detach().cpu()),
        "cane_hc_rank_loss": float(rank.detach().cpu()) if rank_active else 0.0,
        "cane_count_nll": float(count_nll.detach().cpu()),
        "cane_residual_l2": float(residual_l2.detach().cpu()),
        "cane_hc_ez_count": float(hc_count),
        "cane_hc_ez_fraction_actual": float(hc_count / max(observed_count, 1)),
        "cane_valid_rank_patient_count": float(valid_rank_patients if rank_active else 0),
        "cane_mean_anchor_residual": float(outputs["anchor_residual"][channel_mask].mean().detach().cpu()) if torch.any(channel_mask) else 0.0,
        "cane_mean_abs_anchor_residual": float(outputs["anchor_residual"][channel_mask].abs().mean().detach().cpu()) if torch.any(channel_mask) else 0.0,
        "cane_mean_seizure_residual": float(outputs["seizure_residual"][channel_mask].mean().detach().cpu()) if torch.any(channel_mask) else 0.0,
        "cane_mean_abs_seizure_residual": float(outputs["seizure_residual"][channel_mask].abs().mean().detach().cpu()) if torch.any(channel_mask) else 0.0,
        "cane_mean_seizure_agreement": float(outputs["seizure_nez_agreement"][channel_mask].mean().detach().cpu()) if torch.any(channel_mask) else 0.0,
        "cane_mean_predicted_nez_fraction": float(outputs["predicted_nez_fraction"].mean().detach().cpu()),
        "cane_mean_expected_nez_count": float(outputs["expected_nez_count"].mean().detach().cpu()),
        "cane_mean_true_nez_count": float(true_nez_counts.mean().detach().cpu()),
        "cane_mean_cardinality_concentration": float(outputs["cardinality_concentration"].mean().detach().cpu()),
        "cane_anchor_gate": float(outputs["anchor_gate"].detach().cpu()),
        "cane_stage": stage,
        "bce": float(pbce.detach().cpu()),
    }
    for index in range(int(utilization.numel())):
        parts[f"cane_prototype_utilization_{index}"] = float(utilization[index].detach().cpu())
    return total, parts


__all__ = [
    "beta_binomial_count_nll",
    "cane_stage",
    "clean_nez_prototype_loss",
    "compute_cane_set_loss",
    "high_confidence_ez_rank_loss",
    "patient_balanced_nez_bce",
    "patient_soft_macro_f1_loss",
    "residual_l2_regularizer",
    "select_high_confidence_ez",
]
