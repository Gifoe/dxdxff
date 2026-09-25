from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _softplus_inverse(value: float) -> float:
    value = float(max(value, 1e-6))
    return float(torch.log(torch.expm1(torch.tensor(value))).item())


class DiffusionSourceResidualEncoder(nn.Module):
    """Optional diffusion/source residual branch over window-level channel states."""

    def __init__(self, args: Any | None = None, *, model_dim: int | None = None) -> None:
        super().__init__()
        base_model_dim = int(getattr(args, "model_dim", 32) if args is not None else 32)
        self.model_dim = int(model_dim or base_model_dim)
        dropout = float(getattr(args, "dropout", 0.0) if args is not None else 0.0)
        self.graph_source = str(getattr(args, "diffusion_graph_source", "cache_or_functional") if args is not None else "cache_or_functional")
        self.state_proj = nn.Sequential(
            nn.LazyLinear(self.model_dim),
            nn.GELU(),
            nn.LayerNorm(self.model_dim),
        )
        self.residual_mlp = nn.Sequential(
            nn.Linear(self.model_dim, self.model_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.model_dim, self.model_dim),
            nn.LayerNorm(self.model_dim),
        )
        beta_init = float(getattr(args, "diffusion_beta_init", 0.10) if args is not None else 0.10)
        self.beta_raw = nn.Parameter(torch.tensor(_softplus_inverse(beta_init), dtype=torch.float32))

    def forward(
        self,
        physics_features: torch.Tensor,
        diffusion_adjacency: torch.Tensor | None,
        window_mask: torch.Tensor | None,
        seizure_channel_mask: torch.Tensor | None,
        window_centers: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        z = self.state_proj(physics_features)
        b, s, t, c, d = z.shape
        if seizure_channel_mask is None:
            seizure_channel_mask = torch.ones((b, s, c), dtype=torch.bool, device=z.device)
        if window_mask is None:
            window_mask = torch.ones((b, s, t), dtype=torch.bool, device=z.device)
        valid_state = (window_mask[:, :, :, None] & seizure_channel_mask[:, :, None, :]).unsqueeze(-1).float()

        dzdt = self._finite_difference(z, window_mask, seizure_channel_mask, window_centers)
        adjacency = self._resolve_adjacency(z, diffusion_adjacency, seizure_channel_mask)
        laplacian_z = self._laplacian_apply(adjacency, z, seizure_channel_mask)
        beta = F.softplus(self.beta_raw)
        source_residual = (dzdt + beta.view(1, 1, 1, 1, 1) * laplacian_z) * valid_state
        h_diff = self.residual_mlp(source_residual) * valid_state
        zero = physics_features.sum() * 0.0
        losses = {
            "diffusion_source_sparse_loss": self._masked_mean_abs(source_residual, valid_state, zero),
            "diffusion_residual_l2_loss": self._masked_mean_square(source_residual, valid_state, zero),
            "diffusion_beta": beta,
            "diffusion_graph_density": self._graph_density(adjacency, window_mask, seizure_channel_mask, zero),
        }
        return h_diff, losses

    def _finite_difference(
        self,
        z: torch.Tensor,
        window_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
        window_centers: torch.Tensor | None,
    ) -> torch.Tensor:
        dzdt = torch.zeros_like(z)
        if z.shape[2] < 2:
            return dzdt
        if window_centers is not None and tuple(window_centers.shape[:3]) == tuple(window_mask.shape):
            dt = torch.abs(window_centers[:, :, 1:] - window_centers[:, :, :-1]).clamp(1e-3, 60.0)
        else:
            dt = torch.ones((z.shape[0], z.shape[1], z.shape[2] - 1), dtype=z.dtype, device=z.device)
        pair_mask = window_mask[:, :, :-1] & window_mask[:, :, 1:]
        pair_mask = pair_mask[:, :, :, None] & seizure_channel_mask[:, :, None, :]
        increments = (z[:, :, 1:] - z[:, :, :-1]) / dt[:, :, :, None, None]
        increments = increments * pair_mask.unsqueeze(-1).float()
        dzdt[:, :, :-1] = increments
        dzdt[:, :, -1] = increments[:, :, -1] if increments.shape[2] > 0 else 0.0
        return dzdt

    def _resolve_adjacency(
        self,
        z: torch.Tensor,
        diffusion_adjacency: torch.Tensor | None,
        seizure_channel_mask: torch.Tensor,
    ) -> torch.Tensor:
        functional = self._functional_adjacency(z, seizure_channel_mask)
        if diffusion_adjacency is None:
            diffusion_adjacency = torch.zeros_like(functional)
        cache = self._sanitize_cache_adjacency(diffusion_adjacency, seizure_channel_mask)
        if self.graph_source == "functional":
            return functional
        if self.graph_source == "cache":
            return cache
        use_cache = cache.abs().sum(dim=(-1, -2), keepdim=True) > 0.0
        return torch.where(use_cache, cache, functional)

    @staticmethod
    def _sanitize_cache_adjacency(adjacency: torch.Tensor, seizure_channel_mask: torch.Tensor) -> torch.Tensor:
        adjacency = torch.nan_to_num(adjacency.float(), nan=0.0, posinf=0.0, neginf=0.0)
        adjacency = torch.relu(adjacency)
        valid_pairs = seizure_channel_mask[:, :, None, :, None] & seizure_channel_mask[:, :, None, None, :]
        adjacency = adjacency * valid_pairs.float()
        eye = torch.eye(adjacency.shape[-1], dtype=torch.bool, device=adjacency.device).view(1, 1, 1, adjacency.shape[-1], adjacency.shape[-1])
        adjacency = adjacency.masked_fill(eye, 0.0)
        return adjacency

    @staticmethod
    def _functional_adjacency(z: torch.Tensor, seizure_channel_mask: torch.Tensor) -> torch.Tensor:
        normed = F.normalize(z, dim=-1)
        similarity = torch.einsum("bstcd,bstkd->bstck", normed, normed)
        adjacency = torch.relu(similarity)
        valid_pairs = seizure_channel_mask[:, :, None, :, None] & seizure_channel_mask[:, :, None, None, :]
        adjacency = adjacency * valid_pairs.float()
        eye = torch.eye(adjacency.shape[-1], dtype=torch.bool, device=adjacency.device).view(1, 1, 1, adjacency.shape[-1], adjacency.shape[-1])
        adjacency = adjacency.masked_fill(eye, 0.0)
        return adjacency

    @staticmethod
    def _laplacian_apply(adjacency: torch.Tensor, z: torch.Tensor, seizure_channel_mask: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1).clamp_min(1e-6)
        propagated = torch.einsum("bstck,bstkd->bstcd", adjacency, z)
        random_walk = propagated / degree.unsqueeze(-1)
        valid = seizure_channel_mask[:, :, None, :].unsqueeze(-1).float()
        return (z - random_walk) * valid

    @staticmethod
    def _graph_density(
        adjacency: torch.Tensor,
        window_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
        zero: torch.Tensor,
    ) -> torch.Tensor:
        valid_pairs = seizure_channel_mask[:, :, None, :, None] & seizure_channel_mask[:, :, None, None, :]
        valid_pairs = valid_pairs & window_mask[:, :, :, None, None]
        eye = torch.eye(adjacency.shape[-1], dtype=torch.bool, device=adjacency.device).view(1, 1, 1, adjacency.shape[-1], adjacency.shape[-1])
        valid_pairs = valid_pairs & (~eye)
        denom = valid_pairs.float().sum().clamp_min(1.0)
        return (adjacency > 0).float().masked_fill(~valid_pairs, 0.0).sum() / denom if denom.item() > 0.0 else zero

    @staticmethod
    def _masked_mean_abs(values: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().mul(float(values.shape[-1])).clamp_min(1.0)
        return (values.abs() * mask).sum() / denom if denom.item() > 0.0 else zero

    @staticmethod
    def _masked_mean_square(values: torch.Tensor, mask: torch.Tensor, zero: torch.Tensor) -> torch.Tensor:
        denom = mask.sum().mul(float(values.shape[-1])).clamp_min(1.0)
        return (values.square() * mask).sum() / denom if denom.item() > 0.0 else zero


__all__ = ["DiffusionSourceResidualEncoder"]
