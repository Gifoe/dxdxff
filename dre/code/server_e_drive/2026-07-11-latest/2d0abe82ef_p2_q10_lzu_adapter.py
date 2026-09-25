"""Frozen P2-Q10 LZU-only bounded logit residual adapter."""
from __future__ import annotations

import torch
from torch import nn


class P2Q10LZUBoundedAdapter(nn.Module):
    """The sole trainable module; centre membership is deliberately external."""

    def __init__(self, input_dim: int, hidden_dim: int = 16, dropout: float = 0.10, max_delta: float = 0.30) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim != 16:
            raise ValueError("P2-Q10 adapter is fixed to a positive input dimension and hidden_dim=16")
        if not 0.0 < max_delta <= 1.0:
            raise ValueError("max_delta must be in (0, 1]")
        self.max_delta = float(max_delta)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 16),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @property
    def output_layer(self) -> nn.Linear:
        return self.net[-1]

    def trunk_parameters(self):
        yield from self.net[:-1].parameters()

    def output_parameters(self):
        yield from self.output_layer.parameters()

    def forward(self, adapter_input: torch.Tensor, is_lzu: torch.Tensor) -> torch.Tensor:
        if adapter_input.ndim != 2 or is_lzu.ndim != 1 or adapter_input.shape[0] != is_lzu.shape[0]:
            raise ValueError("adapter_input must be [channels, features] and align with is_lzu")
        raw_delta = self.net(adapter_input).squeeze(-1)
        bounded_delta = self.max_delta * torch.tanh(raw_delta)
        # This gate intentionally stays outside the MLP: center is never an input feature.
        return torch.where(is_lzu.to(dtype=torch.bool), bounded_delta, torch.zeros_like(bounded_delta))


def apply_p2_q10_lzu_adapter(base_nez_logit: torch.Tensor, adapter_delta: torch.Tensor, is_lzu: torch.Tensor) -> torch.Tensor:
    if base_nez_logit.shape != adapter_delta.shape or base_nez_logit.shape != is_lzu.shape:
        raise ValueError("base logits, delta, and hard gate must have identical shapes")
    return base_nez_logit + torch.where(is_lzu.to(dtype=torch.bool), adapter_delta, torch.zeros_like(adapter_delta))


__all__ = ["P2Q10LZUBoundedAdapter", "apply_p2_q10_lzu_adapter"]
