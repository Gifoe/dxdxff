from __future__ import annotations

import torch
from torch import nn


class ChannelWindowEncoder(nn.Module):
    def __init__(self, input_dim: int, model_dim: int = 64, dropout: float = 0.1) -> None:
        super().__init__()
        self.input_projection = nn.Linear(int(input_dim), int(model_dim))
        self.block = nn.Sequential(
            nn.Linear(int(model_dim), int(model_dim) * 2),
            nn.LayerNorm(int(model_dim) * 2),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(model_dim) * 2, int(model_dim)),
        )
        self.output_norm = nn.LayerNorm(int(model_dim))

    def forward(self, features: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if features.ndim != 5 or valid_mask.shape != features.shape[:-1]:
            raise ValueError("ChannelWindowEncoder expects features [B,S,W,C,F] and matching valid_mask.")
        projected = self.input_projection(torch.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0))
        encoded = self.output_norm(projected + self.block(projected))
        return encoded * valid_mask.unsqueeze(-1).to(encoded.dtype)


__all__ = ["ChannelWindowEncoder"]
