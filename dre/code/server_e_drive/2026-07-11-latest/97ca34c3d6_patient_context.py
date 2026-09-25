from __future__ import annotations

import math

import torch
from torch import nn

from .masked_pooling import masked_top_fraction


def _masked_token_statistics(values: torch.Tensor, valid_mask: torch.Tensor, top_fraction: float = 0.2) -> torch.Tensor:
    batch, _, _, _, dim = values.shape
    flat = values.reshape(batch, -1, dim)
    valid = valid_mask.reshape(batch, -1)
    weight = valid.unsqueeze(-1).to(values.dtype)
    count = weight.sum(dim=1).clamp_min(1.0)
    mean = (flat * weight).sum(dim=1) / count
    variance = ((flat - mean[:, None, :]).square() * weight).sum(dim=1) / count
    std = torch.sqrt(variance.clamp_min(1e-8))
    maximum = flat.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(dim=1).values
    maximum = torch.nan_to_num(maximum, neginf=0.0, posinf=0.0)
    top = masked_top_fraction(flat, valid, top_fraction)
    return torch.cat([mean, std, maximum, top], dim=-1)


class PatientContextEncoder(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(int(model_dim)))
        nn.init.normal_(self.query, std=0.02)
        self.projection = nn.Sequential(
            nn.Linear(int(model_dim) * 5, int(model_dim) * 2),
            nn.GELU(),
            nn.Linear(int(model_dim) * 2, int(model_dim)),
            nn.LayerNorm(int(model_dim)),
        )

    def forward(self, values: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        batch, _, _, _, dim = values.shape
        flat = values.reshape(batch, -1, dim)
        valid = valid_mask.reshape(batch, -1)
        scores = torch.einsum("bnd,d->bn", flat, self.query) / math.sqrt(max(dim, 1))
        scores = scores.masked_fill(~valid, float("-inf"))
        all_invalid = ~valid.any(dim=1)
        if all_invalid.any():
            scores = scores.clone()
            scores[all_invalid] = 0.0
        attention = torch.softmax(scores, dim=1) * valid.to(values.dtype)
        attention = attention / attention.sum(dim=1, keepdim=True).clamp_min(1e-8)
        attended = torch.einsum("bn,bnd->bd", attention, flat)
        return self.projection(torch.cat([_masked_token_statistics(values, valid_mask), attended], dim=-1))


__all__ = ["PatientContextEncoder", "_masked_token_statistics"]
