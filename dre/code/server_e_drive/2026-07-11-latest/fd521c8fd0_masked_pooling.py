from __future__ import annotations

import math

import torch


def masked_top_fraction(values: torch.Tensor, valid_mask: torch.Tensor, fraction: float) -> torch.Tensor:
    """Pool the highest-norm valid tokens using a per-sample token count."""

    if values.ndim != 3 or valid_mask.shape != values.shape[:2]:
        raise ValueError(f"Expected values [B,N,D] and mask [B,N], got {values.shape} and {valid_mask.shape}.")
    if not 0.0 < float(fraction) <= 1.0:
        raise ValueError("Top fraction must be in (0, 1].")
    pooled = []
    for sample_values, sample_mask in zip(values, valid_mask):
        selected = sample_values[sample_mask]
        if selected.shape[0] == 0:
            pooled.append(torch.zeros(values.shape[-1], device=values.device, dtype=values.dtype))
            continue
        k = max(1, int(math.ceil(selected.shape[0] * float(fraction))))
        indices = torch.topk(selected.norm(dim=-1), k=k, dim=0).indices
        pooled.append(selected[indices].mean(dim=0))
    return torch.stack(pooled, dim=0)


__all__ = ["masked_top_fraction"]
