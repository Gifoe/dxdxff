from __future__ import annotations

import torch
from torch import nn


class ChannelTemporalEncoder(nn.Module):
    """
    Mean-pooling temporal encoder over peri-onset windows.

    TCN and recurrent encoders were removed for B0-Pruned-EZBackbone. Input is
    [B, S, T, C, D], output is [B, S, C, D].
    """

    def __init__(
        self,
        model_dim: int = 96,
        pooling: str = "mean",
        channel_pooling_mode: str | None = None,
        topk_fraction: float = 0.20,
        top_p: float | None = None,
        tau: float = 0.10,
        lse_pool_tau: float = 1.0,
        early_pool_frac: float = 0.25,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        mode = channel_pooling_mode if channel_pooling_mode is not None else pooling
        self.pooling = str(mode).lower()
        self.topk_fraction = float(topk_fraction)
        self.top_p = float(self.topk_fraction if top_p is None else top_p)
        self.tau = max(float(tau), 1e-6)
        self.lse_pool_tau = max(float(lse_pool_tau), 1e-6)
        self.early_pool_frac = min(1.0, max(1e-6, float(early_pool_frac)))
        self.last_n_channels_all_windows_invalid = 0
        if self.pooling not in {
            "mean",
            "mean_max_topk",
            "mean_max_top20",
            "mean_logsumexp_early",
            "gated_dynamic",
            "top_p_mean",
            "logsumexp",
            "noisy_or",
        }:
            raise ValueError(
                f"Unsupported temporal_pooling={self.pooling!r}; expected mean, mean_max_topk, "
                "mean_max_top20, mean_logsumexp_early, gated_dynamic, top_p_mean, logsumexp, or noisy_or."
            )
        self.pool_proj = (
            nn.Linear(self.model_dim * 3, self.model_dim)
            if self.pooling in {"mean_max_topk", "mean_max_top20", "mean_logsumexp_early"}
            else None
        )
        self.gate_mlp = (
            nn.Sequential(
                nn.Linear(self.model_dim * 4, self.model_dim),
                nn.GELU(),
                nn.Linear(self.model_dim, 4),
            )
            if self.pooling == "gated_dynamic"
            else None
        )

    @staticmethod
    def _top_p_mean(values: torch.Tensor, valid: torch.Tensor, p: float) -> torch.Tensor:
        b, s, t, c, d = values.shape
        scores = values.norm(dim=-1).masked_fill(~valid, float("-inf"))
        k = max(1, min(t, int(round(t * max(min(float(p), 1.0), 1e-6)))))
        top_idx = torch.topk(scores, k=k, dim=2).indices
        gather_idx = top_idx.unsqueeze(-1).expand(b, s, k, c, d)
        top_values = torch.gather(values, dim=2, index=gather_idx)
        top_valid = torch.gather(valid, dim=2, index=top_idx)
        denom = top_valid.float().sum(dim=2).clamp_min(1.0)
        return (top_values * top_valid.unsqueeze(-1).float()).sum(dim=2) / denom.unsqueeze(-1)

    @staticmethod
    def _early_mean(values: torch.Tensor, valid: torch.Tensor, frac: float) -> torch.Tensor:
        counts = valid.float().sum(dim=2)
        early_k = torch.ceil(counts * max(min(float(frac), 1.0), 1e-6)).clamp_min(1.0)
        valid_rank = valid.float().cumsum(dim=2)
        early = valid & (valid_rank <= early_k[:, :, None, :])
        denom = early.float().sum(dim=2).clamp_min(1.0)
        return (values * early.unsqueeze(-1).float()).sum(dim=2) / denom.unsqueeze(-1)

    def _logsumexp_mean(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        denom = valid.float().sum(dim=2).clamp_min(1.0)
        masked = values.masked_fill(~valid.unsqueeze(-1), float("-inf"))
        pooled = self.lse_pool_tau * torch.logsumexp(masked / self.lse_pool_tau, dim=2)
        pooled = pooled - self.lse_pool_tau * torch.log(denom.unsqueeze(-1).to(pooled.dtype))
        return torch.nan_to_num(pooled, neginf=0.0, posinf=0.0)

    def forward(
        self,
        window_embeddings: torch.Tensor,
        seizure_channel_mask: torch.Tensor | None = None,
        window_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, s, t, c, d = window_embeddings.shape
        if seizure_channel_mask is None:
            seizure_channel_mask = torch.ones((b, s, c), dtype=torch.bool, device=window_embeddings.device)
        time_mask = seizure_channel_mask[:, :, None, :].expand(b, s, t, c)
        if window_mask is not None:
            time_mask = time_mask & window_mask[:, :, :, None].expand(b, s, t, c)
        finite_mask = torch.isfinite(window_embeddings).all(dim=-1)
        time_mask = time_mask & finite_mask
        window_embeddings = torch.nan_to_num(window_embeddings, nan=0.0, posinf=0.0, neginf=0.0)
        all_invalid = seizure_channel_mask & (~time_mask.any(dim=2))
        self.last_n_channels_all_windows_invalid = int(all_invalid.detach().sum().cpu())

        weights = time_mask.float()
        denom = weights.sum(dim=2, keepdim=True).clamp_min(1.0)
        mean_pooled = (window_embeddings * weights.unsqueeze(-1)).sum(dim=2) / denom.squeeze(2).unsqueeze(-1)
        if self.pooling == "mean":
            pooled = mean_pooled
        elif self.pooling == "top_p_mean":
            pooled = self._top_p_mean(window_embeddings, time_mask, self.top_p)
        elif self.pooling == "logsumexp":
            pooled = self._logsumexp_mean(window_embeddings, time_mask)
        elif self.pooling == "noisy_or":
            probs = torch.sigmoid(window_embeddings).masked_fill(~time_mask.unsqueeze(-1), 0.0)
            pooled_probs = 1.0 - torch.prod(1.0 - probs.clamp(1e-6, 1.0 - 1e-6), dim=2)
            pooled = torch.logit(pooled_probs.clamp(1e-6, 1.0 - 1e-6))
        elif self.pooling == "mean_logsumexp_early":
            lse = self._logsumexp_mean(window_embeddings, time_mask)
            early = self._early_mean(window_embeddings, time_mask, self.early_pool_frac)
            pooled = self.pool_proj(torch.cat([mean_pooled, lse, early], dim=-1))
        elif self.pooling == "gated_dynamic":
            top20 = self._top_p_mean(window_embeddings, time_mask, 0.20)
            lse = self._logsumexp_mean(window_embeddings, time_mask)
            early = self._early_mean(window_embeddings, time_mask, self.early_pool_frac)
            candidates = torch.stack([mean_pooled, top20, lse, early], dim=-2)
            gate_logits = self.gate_mlp(torch.cat([mean_pooled, top20, lse, early], dim=-1))
            gate = torch.softmax(gate_logits, dim=-1).unsqueeze(-1)
            pooled = (gate * candidates).sum(dim=-2)
        else:
            masked_embeddings = window_embeddings.masked_fill(~time_mask.unsqueeze(-1), float("-inf"))
            max_pooled = masked_embeddings.max(dim=2).values
            max_pooled = torch.nan_to_num(max_pooled, neginf=0.0, posinf=0.0)
            top_p = 0.20 if self.pooling == "mean_max_top20" else self.topk_fraction
            topk_mean = self._top_p_mean(window_embeddings, time_mask, top_p)
            pooled = self.pool_proj(torch.cat([mean_pooled, max_pooled, topk_mean], dim=-1))
        pooled = pooled * seizure_channel_mask.float().unsqueeze(-1)
        temporal_weights = (weights / denom).permute(0, 1, 3, 2).contiguous()
        return pooled, temporal_weights


__all__ = ["ChannelTemporalEncoder"]
