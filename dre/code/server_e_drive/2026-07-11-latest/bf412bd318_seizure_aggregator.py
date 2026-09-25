from __future__ import annotations

import torch
from torch import nn


class CrossSeizureMILAggregator(nn.Module):
    """
    Cross-seizure mean plus variability aggregation for each channel.

    Output dimension is 2 * model_dim: concatenated masked mean and masked std.
    """

    def __init__(
        self,
        model_dim: int = 96,
        pooling: str = "mean",
        top_p: float = 0.30,
        alpha: float = 0.70,
        **_: object,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.pooling = str(pooling).lower()
        self.top_p = float(top_p)
        self.alpha = float(alpha)
        if self.pooling not in {"mean", "median", "top_p_mean", "top_p_mean_median_hybrid", "noisy_or"}:
            raise ValueError(
                f"Unsupported record_pooling={self.pooling!r}; expected mean, median, "
                "top_p_mean, top_p_mean_median_hybrid, or noisy_or."
            )
        self.output_dim = self.model_dim * 2

    @staticmethod
    def _masked_median(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        masked = values.masked_fill(~valid.unsqueeze(-1), float("nan"))
        median = torch.nanmedian(masked, dim=1).values
        return torch.nan_to_num(median, nan=0.0)

    @staticmethod
    def _top_p_mean(values: torch.Tensor, valid: torch.Tensor, p: float) -> torch.Tensor:
        b, s, c, d = values.shape
        scores = values.norm(dim=-1).masked_fill(~valid, float("-inf"))
        k = max(1, min(s, int(round(s * max(min(float(p), 1.0), 1e-6)))))
        top_idx = torch.topk(scores, k=k, dim=1).indices
        gather_idx = top_idx.unsqueeze(-1).expand(b, k, c, d)
        top_values = torch.gather(values, dim=1, index=gather_idx)
        top_valid = torch.gather(valid, dim=1, index=top_idx)
        denom = top_valid.float().sum(dim=1).clamp_min(1.0)
        return (top_values * top_valid.unsqueeze(-1).float()).sum(dim=1) / denom.unsqueeze(-1)

    def forward(
        self,
        seizure_channel_embedding: torch.Tensor,
        seizure_mask: torch.Tensor,
        seizure_channel_mask: torch.Tensor,
        seizure_weights: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        valid = seizure_mask[:, :, None] & seizure_channel_mask
        if seizure_weights is None:
            weights = valid.float()
        else:
            weights = valid.float() * seizure_weights.to(
                device=seizure_channel_embedding.device,
                dtype=seizure_channel_embedding.dtype,
            )[:, :, None].clamp_min(0.0)
        denom = weights.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (seizure_channel_embedding * weights.unsqueeze(-1)).sum(dim=1) / denom.squeeze(1).unsqueeze(-1)
        if self.pooling == "mean":
            location = mean
        elif self.pooling == "median":
            location = self._masked_median(seizure_channel_embedding, valid)
        elif self.pooling == "top_p_mean":
            location = self._top_p_mean(seizure_channel_embedding, valid, self.top_p)
        elif self.pooling == "top_p_mean_median_hybrid":
            top_mean = self._top_p_mean(seizure_channel_embedding, valid, self.top_p)
            median = self._masked_median(seizure_channel_embedding, valid)
            location = self.alpha * top_mean + (1.0 - self.alpha) * median
        else:
            probs = torch.sigmoid(seizure_channel_embedding).masked_fill(~valid.unsqueeze(-1), 0.0)
            pooled_probs = 1.0 - torch.prod(1.0 - probs.clamp(1e-6, 1.0 - 1e-6), dim=1)
            location = torch.logit(pooled_probs.clamp(1e-6, 1.0 - 1e-6))
        centered = (seizure_channel_embedding - mean[:, None, :, :]) * weights.unsqueeze(-1)
        var = centered.square().sum(dim=1) / denom.squeeze(1).unsqueeze(-1)
        std = torch.sqrt(var.clamp_min(1e-8))
        embedding = torch.cat([location, std], dim=-1)
        embedding = embedding * (weights.sum(dim=1) > 0.0).float().unsqueeze(-1)
        seizure_weights = (weights / denom).permute(0, 2, 1).contiguous()
        return embedding, seizure_weights


__all__ = ["CrossSeizureMILAggregator"]
