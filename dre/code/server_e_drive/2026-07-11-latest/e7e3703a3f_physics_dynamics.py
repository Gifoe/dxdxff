from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


class NeuralDynamicsResidualEncoder(nn.Module):
    """Linearized neural dynamics residual over channel-window state embeddings."""

    def __init__(self, args: Any | None = None, *, model_dim: int | None = None, num_heads: int | None = None) -> None:
        super().__init__()
        base_dim = int(getattr(args, "model_dim", 32) if args is not None else 32)
        heads = int(getattr(args, "num_heads", 2) if args is not None else 2)
        requested_dim = getattr(args, "physics_model_dim", None) if args is not None else None
        requested_heads = getattr(args, "physics_num_heads", None) if args is not None else None
        self.model_dim = int(model_dim or requested_dim or base_dim)
        self.num_heads = int(num_heads or requested_heads or heads)
        dropout = float(getattr(args, "dropout", 0.0) if args is not None else 0.0)
        self.detach_next_target = bool(getattr(args, "physics_detach_next_target", True) if args is not None else True)

        self.state_proj = nn.Sequential(
            nn.LazyLinear(self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.source_mlp = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Linear(self.model_dim, self.model_dim),
        )
        self.channel_attn = nn.MultiheadAttention(
            embed_dim=self.model_dim,
            num_heads=self.num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_norm = nn.LayerNorm(self.model_dim)
        self.damping_raw = nn.Parameter(torch.zeros((self.model_dim,), dtype=torch.float32))
        self.velocity_proj = nn.Linear(self.model_dim, self.model_dim)
        self.output_norm = nn.LayerNorm(self.model_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        physics_features: torch.Tensor,
        window_mask: torch.Tensor | None,
        seizure_channel_mask: torch.Tensor | None,
        window_centers: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        h = self.state_proj(physics_features)
        b, s, t, c, d = h.shape

        if seizure_channel_mask is None:
            seizure_channel_mask = torch.ones((b, s, c), dtype=torch.bool, device=h.device)
        if window_mask is None:
            window_mask = torch.ones((b, s, t), dtype=torch.bool, device=h.device)

        flat_h = h.reshape(b * s * t, c, d)
        key_padding_mask = (~seizure_channel_mask[:, :, None, :].expand(b, s, t, c)).reshape(b * s * t, c)
        all_invalid = key_padding_mask.all(dim=1)
        if torch.any(all_invalid):
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_invalid] = False
        propagated, _ = self.channel_attn(flat_h, flat_h, flat_h, key_padding_mask=key_padding_mask)
        propagated = self.attn_norm(flat_h + self.dropout(propagated)).reshape(b, s, t, c, d)

        source_drive = self.source_mlp(h)
        damping = F.softplus(self.damping_raw).view(1, 1, 1, 1, d)
        velocity = source_drive + propagated - damping * h
        valid_state = (window_mask[:, :, :, None] & seizure_channel_mask[:, :, None, :]).unsqueeze(-1).float()
        velocity = velocity * valid_state
        h_dyn = self.output_norm(self.velocity_proj(velocity)) * valid_state

        zero = physics_features.sum() * 0.0
        losses = {
            "physics_next_state_loss": self._next_state_loss(h, velocity, window_mask, seizure_channel_mask, window_centers, zero),
            "physics_source_sparse_loss": self._masked_mean_abs(source_drive, valid_state, zero),
            "physics_velocity_l2_loss": self._masked_mean_square(velocity, valid_state, zero),
        }
        return h_dyn, losses

    def _next_state_loss(
        self,
        h: torch.Tensor,
        velocity: torch.Tensor,
        window_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
        window_centers: torch.Tensor | None,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        if h.shape[2] < 2:
            return zero
        pair_mask = window_mask[:, :, :-1] & window_mask[:, :, 1:]
        pair_mask = pair_mask[:, :, :, None] & seizure_channel_mask[:, :, None, :]
        if not torch.any(pair_mask):
            return zero
        if window_centers is not None and tuple(window_centers.shape[:3]) == tuple(window_mask.shape):
            dt = torch.abs(window_centers[:, :, 1:] - window_centers[:, :, :-1]).clamp(1e-3, 60.0)
            dt = dt[:, :, :, None, None]
        else:
            dt = torch.ones_like(h[:, :, :-1, :, :1])
        pred_next = h[:, :, :-1] + dt * velocity[:, :, :-1]
        target_next = h[:, :, 1:]
        if self.detach_next_target:
            target_next = target_next.detach()
        sq = (pred_next - target_next).square()
        weights = pair_mask.unsqueeze(-1).float()
        denom = weights.sum().mul(float(h.shape[-1])).clamp_min(1.0)
        return (sq * weights).sum() / denom

    @staticmethod
    def _masked_mean_abs(values: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().mul(float(values.shape[-1])).clamp_min(1.0)
        if denom.item() <= 0.0:
            return zero
        return (values.abs() * mask).sum() / denom

    @staticmethod
    def _masked_mean_square(values: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().mul(float(values.shape[-1])).clamp_min(1.0)
        if denom.item() <= 0.0:
            return zero
        return (values.square() * mask).sum() / denom


__all__ = ["NeuralDynamicsResidualEncoder"]
