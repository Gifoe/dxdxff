from __future__ import annotations

import math

import torch
from torch import nn


class SeizureTrajectorySummary(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(int(model_dim)))
        nn.init.normal_(self.query, std=0.02)
        self.projection = nn.Sequential(nn.Linear(int(model_dim) * 4, int(model_dim) * 4), nn.GELU(), nn.LayerNorm(int(model_dim) * 4))

    def forward(self, embeddings: torch.Tensor, seizure_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        weight = seizure_mask.unsqueeze(-1).to(embeddings.dtype)
        count = weight.sum(dim=1).clamp_min(1.0)
        mean = (embeddings * weight).sum(dim=1) / count
        std = torch.sqrt(((embeddings - mean[:, None]).square() * weight).sum(dim=1) / count + 1e-8)
        maximum = embeddings.masked_fill(~seizure_mask.unsqueeze(-1), float("-inf")).max(dim=1).values
        maximum = torch.nan_to_num(maximum, neginf=0.0)
        scores = torch.einsum("bsd,d->bs", embeddings, self.query) / math.sqrt(max(embeddings.shape[-1], 1))
        scores = scores.masked_fill(~seizure_mask, float("-inf"))
        attention = torch.softmax(scores, dim=1) * seizure_mask.to(embeddings.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attended = torch.einsum("bs,bsd->bd", attention, embeddings)
        return self.projection(torch.cat([mean, std, maximum, attended], dim=-1)), attention


__all__ = ["SeizureTrajectorySummary"]
