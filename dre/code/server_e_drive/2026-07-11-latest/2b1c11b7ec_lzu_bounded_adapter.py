"""The only trainable module in Stage 2."""
from __future__ import annotations
import torch
from torch import nn

class LZUBoundedResidualAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.10, max_delta: float = 0.15) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.net[-1].weight); nn.init.zeros_(self.net[-1].bias)
    def forward(self, features: torch.Tensor, is_lzu: torch.Tensor) -> torch.Tensor:
        raw = self.net(features).squeeze(-1)
        # Gate is outside the MLP. Non-LZU output is bitwise zero.
        return torch.where(is_lzu.bool(), self.max_delta * torch.tanh(raw), torch.zeros_like(raw))

def apply_lzu_adapter(base_nez_logit: torch.Tensor, adapter_delta: torch.Tensor, is_lzu: torch.Tensor) -> torch.Tensor:
    return base_nez_logit + torch.where(is_lzu.bool(), adapter_delta, torch.zeros_like(adapter_delta))

__all__ = ["LZUBoundedResidualAdapter", "apply_lzu_adapter"]
