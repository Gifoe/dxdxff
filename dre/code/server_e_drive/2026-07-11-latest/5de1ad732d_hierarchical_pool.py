from __future__ import annotations

import torch
from torch import nn

from .masked_pooling import masked_top_fraction


def _masked_statistics(values: torch.Tensor, mask: torch.Tensor, *, include_top: bool = True) -> torch.Tensor:
    if values.shape[:-1] != mask.shape:
        raise ValueError(f"Statistics mask {mask.shape} must match values token shape {values.shape[:-1]}.")
    dim = values.shape[-1]
    flat_values = values.reshape(-1, values.shape[-2], dim)
    flat_mask = mask.reshape(-1, mask.shape[-1])
    weight = flat_mask.unsqueeze(-1).to(values.dtype)
    count = weight.sum(dim=1).clamp_min(1.0)
    mean = (flat_values * weight).sum(dim=1) / count
    variance = ((flat_values - mean[:, None]).square() * weight).sum(dim=1) / count
    std = torch.sqrt(variance.clamp_min(1e-8))
    maximum = flat_values.masked_fill(~flat_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
    maximum = torch.nan_to_num(maximum, neginf=0.0, posinf=0.0)
    parts = [mean, std, maximum]
    if include_top:
        parts.append(masked_top_fraction(flat_values, flat_mask, 0.2))
    return torch.cat(parts, dim=-1).reshape(*values.shape[:-2], -1)


class _GatedAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.value = nn.Linear(dim, dim)
        self.gate = nn.Linear(dim, dim)
        self.score = nn.Linear(dim, 1, bias=False)

    def forward(self, values: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if values.shape[:-1] != mask.shape:
            raise ValueError("Attention mask must match the value token axes.")
        scores = self.score(torch.tanh(self.value(values)) * torch.sigmoid(self.gate(values))).squeeze(-1)
        scores = scores.masked_fill(~mask, float("-inf"))
        all_invalid = ~mask.any(dim=-1)
        if all_invalid.any():
            scores = scores.clone()
            scores[all_invalid] = 0.0
        weights = torch.softmax(scores, dim=-1) * mask.to(values.dtype)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return (values * weights.unsqueeze(-1)).sum(dim=-2), weights


class HierarchicalStatisticalPool(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        dim = int(model_dim)
        self.window_projection = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.LayerNorm(dim))
        self.seizure_projection = nn.Sequential(nn.Linear(dim * 4, dim), nn.GELU(), nn.LayerNorm(dim))
        self.patient_projection = nn.Sequential(nn.Linear(dim * 3, dim * 4), nn.GELU(), nn.LayerNorm(dim * 4))

    def forward(
        self,
        encoded: torch.Tensor,
        window_channel_mask: torch.Tensor,
        window_mask: torch.Tensor,
        seizure_mask: torch.Tensor,
        window_centers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        del window_centers
        window_embeddings = self.window_projection(_masked_statistics(encoded, window_channel_mask))
        window_embeddings = window_embeddings * window_mask.unsqueeze(-1).to(encoded.dtype)
        seizure_embeddings = self.seizure_projection(_masked_statistics(window_embeddings, window_mask))
        seizure_embeddings = seizure_embeddings * seizure_mask.unsqueeze(-1).to(encoded.dtype)
        patient_embedding = self.patient_projection(_masked_statistics(seizure_embeddings, seizure_mask, include_top=False))
        return {
            "patient_embedding": patient_embedding,
            "window_embeddings": window_embeddings,
            "seizure_embeddings": seizure_embeddings,
            "seizure_attention": seizure_mask.to(encoded.dtype) / seizure_mask.sum(dim=-1, keepdim=True).clamp_min(1).to(encoded.dtype),
        }


class HierarchicalAttentionMIL(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        dim = int(model_dim)
        self.channel_attention = _GatedAttention(dim)
        self.time_projection = nn.Sequential(nn.Linear(2, dim), nn.Tanh())
        self.window_attention = _GatedAttention(dim)
        self.seizure_attention = _GatedAttention(dim)
        self.patient_projection = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.LayerNorm(dim * 4))

    @staticmethod
    def _time_features(centers: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        weight = mask.to(centers.dtype)
        safe_centers = torch.where(mask, centers, torch.zeros_like(centers))
        count = weight.sum(dim=-1, keepdim=True).clamp_min(1.0)
        mean = safe_centers.sum(dim=-1, keepdim=True) / count
        centered = torch.where(mask, safe_centers - mean, torch.zeros_like(safe_centers))
        scale = centered.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        relative = centered / scale
        delta = torch.zeros_like(centers)
        adjacent_valid = mask[..., 1:] & mask[..., :-1]
        delta[..., 1:] = torch.where(
            adjacent_valid,
            safe_centers[..., 1:] - safe_centers[..., :-1],
            torch.zeros_like(safe_centers[..., 1:]),
        )
        delta_scale = delta.abs().amax(dim=-1, keepdim=True).clamp_min(1.0)
        return torch.stack([relative, delta / delta_scale], dim=-1)

    def forward(
        self,
        encoded: torch.Tensor,
        window_channel_mask: torch.Tensor,
        window_mask: torch.Tensor,
        seizure_mask: torch.Tensor,
        window_centers: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        window_embeddings, channel_attention = self.channel_attention(encoded, window_channel_mask)
        time = self.time_projection(self._time_features(window_centers, window_mask))
        window_embeddings = (window_embeddings + time) * window_mask.unsqueeze(-1).to(encoded.dtype)
        seizure_embeddings, window_attention = self.window_attention(window_embeddings, window_mask)
        seizure_embeddings = seizure_embeddings * seizure_mask.unsqueeze(-1).to(encoded.dtype)
        pooled, seizure_attention = self.seizure_attention(seizure_embeddings, seizure_mask)
        return {
            "patient_embedding": self.patient_projection(pooled),
            "window_embeddings": window_embeddings,
            "seizure_embeddings": seizure_embeddings,
            "channel_attention": channel_attention,
            "window_attention": window_attention,
            "seizure_attention": seizure_attention,
        }


__all__ = ["HierarchicalAttentionMIL", "HierarchicalStatisticalPool"]
