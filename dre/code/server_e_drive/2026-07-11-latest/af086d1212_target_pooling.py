from __future__ import annotations

import math
import torch


POOL_NAMES = ("global", "global_std", "target", "non_target", "target_contrast", "covered", "residual", "covered_residual_contrast", "top05", "top10", "top20", "top10_target", "top10_outside")


def _weighted(values: torch.Tensor, weight: torch.Tensor, eps: float) -> tuple[torch.Tensor, torch.Tensor]:
    denominator = weight.sum(dim=1, keepdim=True)
    pooled = (values * weight.unsqueeze(-1)).sum(dim=1) / denominator.clamp_min(eps)
    valid = denominator.squeeze(1) > eps
    return pooled.masked_fill(~valid.unsqueeze(-1), 0.0), valid


def compute_target_pooling(embedding: torch.Tensor, abnormality: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor, eps: float = 1e-8) -> dict[str, torch.Tensor]:
    mask = channel_mask.to(embedding.dtype)
    # Padded target entries are intentionally NaN so missing labels can never be
    # mistaken for non-target. Sanitize only padding after applying channel_mask;
    # NaN * 0 is still NaN and otherwise poisons division gradients.
    clean_target=torch.where(channel_mask.bool(),target.to(embedding.dtype),torch.zeros_like(target,dtype=embedding.dtype))
    t = clean_target * mask; nt = (1-clean_target)*mask
    global_mean, global_valid = _weighted(embedding, mask, eps)
    centered = (embedding-global_mean.unsqueeze(1)).square(); variance=_weighted(centered, mask, eps)[0]
    # sqrt(0) has an infinite derivative and can create 0*inf NaNs for constant
    # projected dimensions. The clamped branch is deterministic and finite.
    global_std = variance.clamp_min(eps).sqrt()
    h_target, target_valid = _weighted(embedding, t, eps); h_non, non_valid = _weighted(embedding, nt, eps)
    h_covered, covered_valid = _weighted(embedding, abnormality*t, eps); h_residual, residual_valid = _weighted(embedding, abnormality*nt, eps)
    output = {"global": global_mean, "global_std": global_std, "target": h_target, "non_target": h_non, "target_contrast": h_target-h_non,
              "covered": h_covered, "residual": h_residual, "covered_residual_contrast": h_covered-h_residual,
              "global_valid": global_valid, "target_valid": target_valid, "non_target_valid": non_valid, "covered_valid": covered_valid, "residual_valid": residual_valid}
    for fraction, name in ((.05,"top05"),(.10,"top10"),(.20,"top20")):
        weights = torch.zeros_like(abnormality)
        for index in range(embedding.shape[0]):
            valid_idx = torch.where(channel_mask[index].bool())[0]; k=min(valid_idx.numel(),max(1,math.ceil(valid_idx.numel()*fraction)))
            chosen=valid_idx[torch.topk(abnormality[index,valid_idx],k).indices]; weights[index,chosen]=1
        output[name], output[f"{name}_valid"] = _weighted(embedding,weights,eps)
        if name == "top10":
            output["top10_target"], output["top10_target_valid"] = _weighted(embedding,weights*t,eps)
            output["top10_outside"], output["top10_outside_valid"] = _weighted(embedding,weights*nt,eps)
    return output


__all__ = ["POOL_NAMES", "compute_target_pooling"]
