"""Label-noise-aware observed-EZ ranking and anchor preservation."""

from __future__ import annotations

import hashlib
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F


def _percentile_rank(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(values)
    for patient_idx in range(values.shape[0]):
        indices = torch.nonzero(valid[patient_idx], as_tuple=False).flatten()
        if indices.numel() == 0:
            continue
        selected = values[patient_idx, indices]
        order = torch.argsort(selected, stable=True)
        ranks = torch.empty_like(selected)
        ranks[order] = torch.arange(selected.numel(), device=values.device, dtype=values.dtype)
        ranks = ranks / max(selected.numel() - 1, 1)
        result[patient_idx, indices] = ranks
    return result


def compute_observed_ez_reliability(
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    direct_score_nez: torch.Tensor,
    anchor_nez_evidence: torch.Tensor,
    seizure_nez_probability_mean: torch.Tensor,
    cp_early_source_rank: torch.Tensor,
    *,
    seizure_available: torch.Tensor | None = None,
    causal_available: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    valid = channel_mask.bool() & (labels_nez >= 0)
    observed_ez = valid & (labels_nez < 0.5)
    with torch.no_grad():
        direct_component = 1.0 - _percentile_rank(direct_score_nez.detach(), valid)
        anchor_component = 1.0 - _percentile_rank(anchor_nez_evidence.detach(), valid)
        seizure_component = 1.0 - _percentile_rank(seizure_nez_probability_mean.detach(), valid)
        causal_component = _percentile_rank(cp_early_source_rank.detach(), valid)
        available = [valid, valid]
        available.append(valid if seizure_available is None else valid & seizure_available.bool())
        available.append(valid if causal_available is None else valid & causal_available.bool())
        components = [direct_component, anchor_component, seizure_component, causal_component]
        numerator = torch.zeros_like(direct_score_nez)
        count = torch.zeros_like(direct_score_nez)
        for component, component_valid in zip(components, available):
            numerator += component * component_valid.to(component.dtype)
            count += component_valid.to(component.dtype)
        raw = numerator / count.clamp_min(1.0)
        reliability = (0.10 + 0.90 * raw).clamp(0.10, 1.0)
        reliability = torch.where(observed_ez, reliability, torch.ones_like(reliability))
    return {
        "ez_reliability": reliability.detach(),
        "reliability_component_count": count.detach(),
        "reliability_direct_component": direct_component.detach(),
        "reliability_anchor_component": anchor_component.detach(),
        "reliability_seizure_component": seizure_component.detach(),
        "reliability_causal_component": causal_component.detach(),
    }


def _sample_pairs(
    clean_indices: torch.Tensor,
    ez_indices: torch.Tensor,
    reliability: torch.Tensor,
    max_pairs: int,
    seed_text: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    left = clean_indices.repeat_interleave(ez_indices.numel())
    right = ez_indices.repeat(clean_indices.numel())
    if left.numel() <= int(max_pairs):
        return left, right
    # Stratify by reliability quantiles, then sample deterministically inside each stratum.
    weights = reliability[right].detach().cpu().numpy()
    quantile_edges = np.quantile(weights, [0.0, 0.25, 0.50, 0.75, 1.0])
    digest = hashlib.sha256(seed_text.encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
    selected: list[int] = []
    quota = int(max_pairs) // 4
    for stratum in range(4):
        lower, upper = quantile_edges[stratum], quantile_edges[stratum + 1]
        candidates = np.flatnonzero((weights >= lower) & ((weights <= upper) if stratum == 3 else (weights < upper)))
        if candidates.size:
            selected.extend(rng.choice(candidates, size=min(quota, candidates.size), replace=False).tolist())
    remaining = np.setdiff1d(np.arange(len(weights)), np.asarray(selected, dtype=int), assume_unique=False)
    if len(selected) < int(max_pairs) and remaining.size:
        selected.extend(rng.choice(remaining, size=min(int(max_pairs) - len(selected), remaining.size), replace=False).tolist())
    chosen = torch.as_tensor(sorted(selected[: int(max_pairs)]), device=left.device, dtype=torch.long)
    return left[chosen], right[chosen]


def reliability_weighted_pairwise_rank_loss(
    final_nez_logit: torch.Tensor,
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    ez_reliability: torch.Tensor,
    *,
    margin: float = 0.10,
    max_pairs_per_patient: int = 4096,
    model_seed: int = 42,
    epoch: int = 1,
    subject_ids: Sequence[str] | None = None,
    active_from_epoch: int = 10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = final_nez_logit.sum() * 0.0
    if int(epoch) < int(active_from_epoch):
        return zero, {"weighted_rank_pair_count": zero, "weighted_rank_patient_count": zero}
    losses, pair_counts = [], []
    for patient_idx in range(final_nez_logit.shape[0]):
        valid = channel_mask[patient_idx].bool() & (labels_nez[patient_idx] >= 0)
        clean = torch.nonzero(valid & (labels_nez[patient_idx] > 0.5), as_tuple=False).flatten()
        observed = torch.nonzero(valid & (labels_nez[patient_idx] < 0.5), as_tuple=False).flatten()
        if clean.numel() == 0 or observed.numel() == 0:
            continue
        subject = str(subject_ids[patient_idx]) if subject_ids is not None else str(patient_idx)
        left, right = _sample_pairs(
            clean, observed, ez_reliability[patient_idx], max_pairs_per_patient,
            f"{model_seed}:{epoch}:{subject}:rank",
        )
        pair_loss = F.softplus(float(margin) - final_nez_logit[patient_idx, left] + final_nez_logit[patient_idx, right])
        weights = ez_reliability[patient_idx, right].detach()
        losses.append((pair_loss * weights).sum() / weights.sum().clamp_min(1e-8))
        pair_counts.append(left.numel())
    loss = torch.stack(losses).mean() if losses else zero
    return loss, {
        "weighted_rank_pair_count": torch.tensor(float(sum(pair_counts)), device=zero.device),
        "weighted_rank_patient_count": torch.tensor(float(len(losses)), device=zero.device),
    }


def ranking_preservation_loss(
    final_nez_logit: torch.Tensor,
    reference_logit: torch.Tensor,
    labels_nez: torch.Tensor,
    channel_mask: torch.Tensor,
    ez_reliability: torch.Tensor,
    *,
    safe_margin: float = 0.30,
    beta: float = 0.80,
    max_pairs_per_patient: int = 4096,
    model_seed: int = 42,
    epoch: int = 1,
    subject_ids: Sequence[str] | None = None,
    active_from_epoch: int = 10,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    zero = final_nez_logit.sum() * 0.0
    if int(epoch) < int(active_from_epoch):
        return zero, {
            "preserve_pair_count": zero, "preserve_violation_rate": zero,
            "mean_reference_margin": zero, "mean_final_margin_on_preserved_pairs": zero,
        }
    losses, violations, reference_values, final_values, pair_count = [], [], [], [], 0
    reference = reference_logit.detach()
    for patient_idx in range(final_nez_logit.shape[0]):
        valid = channel_mask[patient_idx].bool() & (labels_nez[patient_idx] >= 0)
        clean = torch.nonzero(valid & (labels_nez[patient_idx] > 0.5), as_tuple=False).flatten()
        observed = torch.nonzero(valid & (labels_nez[patient_idx] < 0.5), as_tuple=False).flatten()
        if clean.numel() == 0 or observed.numel() == 0:
            continue
        subject = str(subject_ids[patient_idx]) if subject_ids is not None else str(patient_idx)
        left, right = _sample_pairs(clean, observed, ez_reliability[patient_idx], max_pairs_per_patient, f"{model_seed}:{epoch}:{subject}:preserve")
        ref_margin = reference[patient_idx, left] - reference[patient_idx, right]
        selected = ref_margin >= float(safe_margin)
        if not torch.any(selected):
            continue
        left, right, ref_margin = left[selected], right[selected], ref_margin[selected]
        final_margin = final_nez_logit[patient_idx, left] - final_nez_logit[patient_idx, right]
        violation = F.relu(float(beta) * ref_margin - final_margin)
        weights = ez_reliability[patient_idx, right].detach()
        losses.append((violation * weights).sum() / weights.sum().clamp_min(1e-8))
        violations.append((violation > 0).to(final_nez_logit.dtype).mean())
        reference_values.append(ref_margin.mean())
        final_values.append(final_margin.mean())
        pair_count += int(left.numel())
    return (torch.stack(losses).mean() if losses else zero), {
        "preserve_pair_count": torch.tensor(float(pair_count), device=zero.device),
        "preserve_violation_rate": torch.stack(violations).mean() if violations else zero,
        "mean_reference_margin": torch.stack(reference_values).mean() if reference_values else zero,
        "mean_final_margin_on_preserved_pairs": torch.stack(final_values).mean() if final_values else zero,
    }


__all__ = [
    "compute_observed_ez_reliability", "ranking_preservation_loss",
    "reliability_weighted_pairwise_rank_loss",
]
