from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from .config import BNPDGSConfig


def bn_pdgs_loss(
    outputs: dict[str, torch.Tensor],
    batch: dict[str, Any],
    args: Any | None = None,
    pos_weight: float | torch.Tensor = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = BNPDGSConfig.from_args(args)
    logits = outputs["logits"]
    labels = batch["labels"].to(logits.device).float()
    channel_mask = batch["channel_mask"].to(logits.device).bool()

    bce = weighted_bce_loss(logits, labels, channel_mask, pos_weight)
    pairwise = pairwise_ranking_loss(logits, labels, channel_mask, margin=cfg.ranking_margin)
    listwise = listwise_ranking_loss(logits, labels, channel_mask)
    count = count_regularization(outputs["predicted_count"], labels, channel_mask)
    mass = score_mass_regularization(outputs["score_mass"], labels, channel_mask)
    smooth = temporal_smoothness_loss(outputs.get("temporal_attention"), batch.get("seizure_channel_mask"))

    total = (
        cfg.lambda_bce * bce
        + cfg.lambda_pairwise * pairwise
        + cfg.lambda_listwise * listwise
        + cfg.lambda_count * count
        + cfg.lambda_mass * mass
        + cfg.lambda_temporal_smooth * smooth
    )
    return total, {
        "loss": total.detach(),
        "bce": bce.detach(),
        "pairwise": pairwise.detach(),
        "listwise": listwise.detach(),
        "count": count.detach(),
        "mass": mass.detach(),
        "temporal_smooth": smooth.detach(),
    }


def weighted_bce_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    channel_mask: torch.Tensor,
    pos_weight: float | torch.Tensor = 1.0,
) -> torch.Tensor:
    if not torch.any(channel_mask):
        return logits.sum() * 0.0
    pos_weight_t = torch.as_tensor(float(pos_weight), dtype=logits.dtype, device=logits.device)
    return F.binary_cross_entropy_with_logits(
        logits[channel_mask],
        labels[channel_mask],
        pos_weight=pos_weight_t,
    )


def pairwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    channel_mask: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    losses = []
    for batch_idx in range(logits.shape[0]):
        valid = channel_mask[batch_idx]
        y = labels[batch_idx][valid]
        z = logits[batch_idx][valid]
        pos = z[y > 0.5]
        neg = z[y <= 0.5]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        losses.append(F.softplus(float(margin) - pos[:, None] + neg[None, :]).mean())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def listwise_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for batch_idx in range(logits.shape[0]):
        valid = channel_mask[batch_idx]
        y = labels[batch_idx][valid]
        if y.sum() <= 0.0:
            continue
        z = logits[batch_idx][valid]
        target = y / y.sum().clamp_min(1e-6)
        losses.append(-(target * F.log_softmax(z, dim=0)).sum())
    if not losses:
        return logits.sum() * 0.0
    return torch.stack(losses).mean()


def count_regularization(
    predicted_count: torch.Tensor,
    labels: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    valid_count = channel_mask.float().sum(dim=1).clamp_min(1.0)
    true_count = (labels * channel_mask.float()).sum(dim=1)
    return F.smooth_l1_loss(predicted_count / valid_count, true_count / valid_count)


def score_mass_regularization(
    score_mass: torch.Tensor,
    labels: torch.Tensor,
    channel_mask: torch.Tensor,
) -> torch.Tensor:
    valid_count = channel_mask.float().sum(dim=1).clamp_min(1.0)
    true_count = (labels * channel_mask.float()).sum(dim=1)
    return F.smooth_l1_loss(score_mass / valid_count, true_count / valid_count)


def temporal_smoothness_loss(
    temporal_attention: torch.Tensor | None,
    seizure_channel_mask: torch.Tensor | None,
) -> torch.Tensor:
    if temporal_attention is None:
        return torch.tensor(0.0)
    diff = temporal_attention[..., 1:] - temporal_attention[..., :-1]
    if seizure_channel_mask is None:
        return torch.square(diff).mean()
    valid = seizure_channel_mask.to(temporal_attention.device).bool().unsqueeze(-1)
    if not torch.any(valid):
        return temporal_attention.sum() * 0.0
    return torch.square(diff).masked_select(valid.expand_as(diff)).mean()


__all__ = [
    "bn_pdgs_loss",
    "count_regularization",
    "listwise_ranking_loss",
    "pairwise_ranking_loss",
    "score_mass_regularization",
    "temporal_smoothness_loss",
    "weighted_bce_loss",
]

