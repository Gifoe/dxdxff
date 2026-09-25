from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn


def _gini(values: torch.Tensor) -> torch.Tensor:
    sorted_values = torch.sort(values, dim=-1).values
    count = values.shape[-1]
    weights = torch.arange(1, count + 1, device=values.device, dtype=values.dtype)
    numerator = (weights * sorted_values).sum(dim=-1)
    denominator = sorted_values.sum(dim=-1).clamp_min(1e-8)
    return (2.0 * numerator / denominator - (count + 1.0)) / max(count, 1)


class WindowFocalityDescriptor(nn.Module):
    output_dim = 8

    def forward(
        self,
        core_masses: torch.Tensor,
        background_mass: torch.Tensor,
        responsibilities: torch.Tensor,
        anchors: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        total = core_masses.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        normalized = core_masses / total
        entropy = -(normalized * torch.log(normalized.clamp_min(1e-8))).sum(dim=-1) / max(math.log(max(core_masses.shape[-1], 2)), 1e-8)
        effective = torch.exp(-(normalized * torch.log(normalized.clamp_min(1e-8))).sum(dim=-1)) / max(core_masses.shape[-1], 1)
        dominant = core_masses.max(dim=-1).values
        gini = _gini(core_masses)
        imbalance = core_masses.std(dim=-1, unbiased=False)
        core_responsibility = responsibilities[..., :-1] * valid_mask.unsqueeze(-1).to(responsibilities.dtype)
        channel_distribution = core_responsibility / core_responsibility.sum(dim=-2, keepdim=True).clamp_min(1e-8)
        raw_channel_entropy = -(channel_distribution * torch.log(channel_distribution.clamp_min(1e-8))).sum(dim=-2)
        valid_channel_count = valid_mask.sum(dim=-1)
        log_count = torch.log(valid_channel_count.clamp_min(2).to(responsibilities.dtype))
        normalized_channel_entropy = raw_channel_entropy / log_count.unsqueeze(-1)
        normalized_channel_entropy = torch.where(
            valid_channel_count.unsqueeze(-1) > 1,
            normalized_channel_entropy,
            torch.zeros_like(normalized_channel_entropy),
        )
        channel_entropy = normalized_channel_entropy.mean(dim=-1)
        anchor_norm = functional.normalize(anchors, dim=-1, eps=1e-8)
        similarity = torch.einsum("bkd,bld->bkl", anchor_norm, anchor_norm)
        eye = torch.eye(similarity.shape[-1], device=similarity.device, dtype=torch.bool).unsqueeze(0)
        distance = (1.0 - similarity).masked_fill(eye, 0.0)
        separation = distance.sum(dim=(-1, -2)) / max(similarity.shape[-1] * (similarity.shape[-1] - 1), 1)
        separation = separation[:, None, None].expand_as(entropy)
        descriptor = torch.stack([entropy, effective, dominant, background_mass, gini, imbalance, channel_entropy, separation], dim=-1)
        return descriptor * valid_mask.any(dim=-1).unsqueeze(-1).to(descriptor.dtype)


__all__ = ["WindowFocalityDescriptor"]
