"""Prelocked patient-equal objectives; source NEZ-positive logits."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from objectives import patient_equal_weighted_bce_loss


def balanced_patient_losses(logits, labels_nez, labels_ez, mask):
    valid = mask.bool() & (labels_nez >= 0) & (labels_ez >= 0)
    ez = valid & (labels_ez > 0.5)
    nez = valid & (labels_ez < 0.5)
    if not bool((ez.any(dim=1) & nez.any(dim=1)).all()):
        raise RuntimeError("Single-class or inactive patient in balanced BCE")
    safe = torch.where(valid, labels_nez, torch.zeros_like(labels_nez))
    per_channel = F.binary_cross_entropy_with_logits(logits, safe, reduction="none")
    loss_ez = (per_channel * ez).sum(dim=1) / ez.sum(dim=1)
    loss_nez = (per_channel * nez).sum(dim=1) / nez.sum(dim=1)
    return 0.5 * (loss_ez + loss_nez)


def hard_negative_losses(logits, labels_ez, mask):
    valid = mask.bool() & (labels_ez >= 0)
    scores = -logits
    losses = []
    for row in range(logits.shape[0]):
        positive = scores[row][valid[row] & (labels_ez[row] > 0.5)]
        negatives = scores[row][valid[row] & (labels_ez[row] < 0.5)]
        if not positive.numel() or not negatives.numel():
            raise RuntimeError("Single-class patient in hard-negative ranking")
        top = torch.topk(negatives, min(16, negatives.numel())).values
        losses.append(F.softplus(0.10 - positive[:, None] + top[None, :]).mean())
    return torch.stack(losses)


def first_positive_losses(logits, labels_ez, mask):
    valid = mask.bool() & (labels_ez >= 0)
    scores = -logits
    losses = []
    for row in range(logits.shape[0]):
        positive = scores[row][valid[row] & (labels_ez[row] > 0.5)]
        negatives = scores[row][valid[row] & (labels_ez[row] < 0.5)]
        if not positive.numel() or not negatives.numel():
            raise RuntimeError("Single-class patient in first-positive objective")
        max_pos = 0.10 * torch.logsumexp(positive / 0.10, dim=0)
        max_neg = torch.topk(negatives, min(16, negatives.numel())).values.max()
        losses.append(F.softplus(0.10 - max_pos + max_neg))
    return torch.stack(losses)


def loss_for_variant(variant, capacity_base):
    balanced = variant in ("B10", "B11") or (variant.startswith("CAP") and capacity_base == "balanced") or (variant == "FP" and capacity_base == "balanced")
    ranked = variant in ("B01", "B11")
    first_positive = variant == "FP"

    def compute(self, outputs, batch, ez_negative_weight, *, split_name="train"):
        if float(ez_negative_weight) != 2.0:
            raise RuntimeError("Source EZ weight changed")
        logits = outputs["logits"]
        ynez, yez, mask = batch["labels"], batch["labels_ez"], batch["channel_mask"]
        bce = balanced_patient_losses(logits, ynez, yez, mask).mean() if balanced else patient_equal_weighted_bce_loss(logits, ynez, yez, mask)
        # Validation loss is diagnostic only; the VLOO selector uses the frozen metric grid.
        if split_name != "train" or (not ranked and not first_positive):
            return bce, {"bce": float(bce.detach())}
        extra = (hard_negative_losses(logits, yez, mask).mean() if ranked else first_positive_losses(logits, yez, mask).mean())
        total = bce + 0.05 * extra
        return total, {"bce": float(bce.detach()), "ranking": float(extra.detach())}
    return compute
