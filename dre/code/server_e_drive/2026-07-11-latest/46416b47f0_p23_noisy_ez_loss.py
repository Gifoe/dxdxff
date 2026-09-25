"""Asymmetric clean-NEZ / observed-noisy-EZ objective for P23."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def _percentile(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    result = torch.zeros_like(values)
    for batch_idx in range(values.shape[0]):
        valid = mask[batch_idx].bool()
        selected = values[batch_idx, valid]
        if selected.numel() <= 1:
            result[batch_idx, valid] = 0.5
        elif selected.numel():
            ranks = torch.argsort(torch.argsort(selected, stable=True), stable=True).to(values.dtype)
            result[batch_idx, valid] = ranks / float(selected.numel() - 1)
    return result


def observed_ez_reliability(
    teacher_score_nez: torch.Tensor, anchor_evidence: torch.Tensor,
    seizure_mean: torch.Tensor, seizure_q10: torch.Tensor, observed_ez: torch.Tensor,
    *, minimum: float = 0.10, maximum: float = 0.95,
) -> torch.Tensor:
    """Detached reliability: high when every NEZ-direction signal is low."""
    valid = observed_ez.bool()
    components = []
    for value in (teacher_score_nez, anchor_evidence, seizure_mean, seizure_q10):
        components.append(1.0 - _percentile(value.detach(), valid))
    raw = torch.stack(components).mean(dim=0)
    reliability = (minimum + (maximum - minimum) * raw).clamp(minimum, maximum)
    return reliability.masked_fill(~valid, 0.0).detach()


def noise_ramp(epoch: int, start: int, end: int) -> float:
    if epoch < start:
        return 0.0
    if end <= start:
        return 1.0
    return float(min(1.0, max(0.0, (epoch - start) / float(end - start))))


def reliability_weighted_rank_loss(
    logits: torch.Tensor, labels_nez: torch.Tensor, reliability: torch.Tensor,
    channel_mask: torch.Tensor, *, margin: float, max_pairs: int, seed: int,
) -> tuple[torch.Tensor, int]:
    rows, pairs = [], 0
    generator = torch.Generator(device=logits.device); generator.manual_seed(int(seed))
    for batch_idx in range(logits.shape[0]):
        valid = channel_mask[batch_idx].bool()
        clean = torch.where(valid & (labels_nez[batch_idx] > 0.5))[0]
        observed = torch.where(valid & (labels_nez[batch_idx] <= 0.5))[0]
        if not clean.numel() or not observed.numel():
            continue
        all_pairs = clean.numel() * observed.numel()
        if all_pairs > max_pairs:
            indices = torch.randperm(all_pairs, generator=generator, device=logits.device)[:max_pairs]
            clean_idx, observed_idx = clean[indices // observed.numel()], observed[indices % observed.numel()]
        else:
            clean_idx = clean.repeat_interleave(observed.numel())
            observed_idx = observed.repeat(clean.numel())
        pair_weight = reliability[batch_idx, observed_idx]
        rows.append((pair_weight * F.softplus(margin - logits[batch_idx, clean_idx] + logits[batch_idx, observed_idx])).sum() / pair_weight.sum().clamp_min(1e-6))
        pairs += int(clean_idx.numel())
    return (torch.stack(rows).mean() if rows else logits.sum() * 0.0), pairs


def core_existence_loss(logits: torch.Tensor, labels_nez: torch.Tensor, channel_mask: torch.Tensor, *, margin: float, tau: float) -> torch.Tensor:
    rows = []
    for batch_idx in range(logits.shape[0]):
        valid = channel_mask[batch_idx].bool()
        clean = logits[batch_idx, valid & (labels_nez[batch_idx] > 0.5)]
        observed = logits[batch_idx, valid & (labels_nez[batch_idx] <= 0.5)]
        if clean.numel() and observed.numel():
            softmin = -tau * torch.logsumexp(-observed / tau, dim=0) + tau * torch.log(torch.tensor(float(observed.numel()), device=logits.device))
            rows.append(F.softplus(torch.as_tensor(margin, dtype=logits.dtype, device=logits.device) - clean.mean() + softmin))
    return torch.stack(rows).mean() if rows else logits.sum() * 0.0


def compute_p23_loss(outputs: dict[str, torch.Tensor], batch: dict[str, Any], args: Any, *, epoch: int, teacher_score_nez: torch.Tensor | None) -> tuple[torch.Tensor, dict[str, float]]:
    logits = outputs["final_nez_logit"]
    mask = batch["channel_mask"].bool()
    labels_nez = 1.0 - batch["labels_ez"].to(logits.dtype)
    clean = mask & (labels_nez > 0.5)
    observed = mask & ~clean
    zero = logits.sum() * 0.0
    clean_rows, observed_rows = [], []
    noise_enabled = str(getattr(args, "p23_profile", "P5_FULL")).upper() in {"P4_NOISE_AWARE", "P5_FULL", "P6_FULL_WITH_CAUSAL"}
    ramp = noise_ramp(epoch, int(args.p23_noise_ramp_start), int(args.p23_noise_ramp_end)) if noise_enabled else 0.0
    if teacher_score_nez is None:
        teacher_score_nez = torch.sigmoid(logits.detach())
    teacher = teacher_score_nez.detach().clamp(0.05, 0.95)
    reliability = observed_ez_reliability(
        teacher, outputs["u_anchor"], outputs["seizure_nez_probability_mean"],
        outputs["seizure_nez_probability_q10"], observed,
        minimum=float(args.p23_observed_ez_reliability_min), maximum=float(args.p23_observed_ez_reliability_max),
    )
    soft_target = ((1.0 - reliability) * teacher).detach().masked_fill(~observed, 0.0)
    for batch_idx in range(logits.shape[0]):
        if torch.any(clean[batch_idx]):
            clean_rows.append(F.binary_cross_entropy_with_logits(logits[batch_idx, clean[batch_idx]], torch.ones_like(logits[batch_idx, clean[batch_idx]])))
        if torch.any(observed[batch_idx]):
            hard = F.binary_cross_entropy_with_logits(logits[batch_idx, observed[batch_idx]], torch.zeros_like(logits[batch_idx, observed[batch_idx]]))
            soft = F.binary_cross_entropy_with_logits(logits[batch_idx, observed[batch_idx]], soft_target[batch_idx, observed[batch_idx]])
            observed_rows.append((1.0 - ramp) * 0.5 * hard + ramp * soft)
    clean_loss = torch.stack(clean_rows).mean() if clean_rows else zero
    observed_loss = torch.stack(observed_rows).mean() if observed_rows else zero
    base = 0.5 * (clean_loss + observed_loss)
    noise_active = ramp > 0.0
    rank, pair_count = reliability_weighted_rank_loss(
        logits, labels_nez, reliability, mask, margin=float(args.p23_rank_margin),
        max_pairs=int(args.p23_max_pairs_per_patient), seed=int(getattr(args, "model_seed", 42)) + int(epoch),
    )
    if epoch < int(args.p23_rank_start_epoch):
        rank = zero; pair_count = 0
    core = core_existence_loss(logits, labels_nez, mask, margin=float(args.p23_core_margin), tau=float(args.p23_core_tau)) if noise_active else zero
    delta_l2 = outputs["delta"][mask].square().mean() if torch.any(mask) else zero
    temporal_l2 = outputs.get("temporal_delta_norm", torch.zeros_like(logits))[mask].square().mean() if torch.any(mask) else zero
    rank_weight = float(args.p23_rank_weight) if noise_enabled else 0.0
    total = base + rank_weight * rank + float(args.p23_core_weight) * core + float(args.p23_delta_l2_weight) * delta_l2 + float(args.p23_temporal_delta_l2_weight) * temporal_l2
    outputs["ema_teacher_score_nez"] = teacher
    outputs["observed_ez_reliability"] = reliability
    outputs["soft_target_nez"] = soft_target
    outputs["noise_aware_ramp"] = torch.full_like(logits, float(ramp))
    predicted_nez_fraction = (
        (torch.sigmoid(logits)[mask] >= 0.5).to(logits.dtype).mean()
        if torch.any(mask) else zero
    )
    soft_target_over_half = (
        (soft_target[observed] > 0.5).to(logits.dtype).mean()
        if torch.any(observed) else zero
    )
    return total, {
        "p23_total_loss": float(total.detach().cpu()), "p23_clean_nez_loss": float(clean_loss.detach().cpu()),
        "p23_observed_ez_loss": float(observed_loss.detach().cpu()), "p23_rank_loss": float(rank.detach().cpu()),
        "p23_core_existence_loss": float(core.detach().cpu()), "p23_noise_aware_ramp": float(ramp),
        "p23_mean_observed_ez_reliability": float(reliability[observed].mean().detach().cpu()) if torch.any(observed) else 0.0,
        "p23_ranking_pair_count": float(pair_count), "p23_delta_l2": float(delta_l2.detach().cpu()),
        "p23_temporal_delta_l2": float(temporal_l2.detach().cpu()),
        "p23_predicted_nez_fraction_at_0_5": float(predicted_nez_fraction.detach().cpu()),
        "p23_observed_ez_soft_target_gt_0_5_fraction": float(soft_target_over_half.detach().cpu()),
    }
