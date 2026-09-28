"""DRST-PR spectral ablations with identical backbone and optional latent PR."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from seizure_aggregator import CrossSeizureMILAggregator
from spectral import log_power_stft, spectral_views


VARIANT_VIEWS = {"S1_ABS_SPECTRAL": 1, "S2_ABS_SELF": 2,
                 "S3_DUAL_REFERENCE": 3, "S4_DRST_PR": 3}


class GatedTemporalBlock(nn.Module):
    def __init__(self, dim: int = 64, dilation: int = 1, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(dim, dim, 3, padding=dilation, dilation=dilation, groups=dim)
        self.pointwise = nn.Linear(dim, 2 * dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x).transpose(1, 2)
        h = self.depthwise(h).transpose(1, 2)
        return x + self.dropout(nn.functional.glu(self.pointwise(h), dim=-1))


class SpectralWindowEncoder(nn.Module):
    def __init__(self, n_views: int) -> None:
        super().__init__()
        self.tokenizer = nn.Sequential(nn.Linear(n_views * 51, 128), nn.GELU(),
                                       nn.LayerNorm(128), nn.Linear(128, 64), nn.GELU())
        self.blocks = nn.Sequential(GatedTemporalBlock(), GatedTemporalBlock())
        self.attention = nn.Linear(64, 1)

    def forward(self, views: torch.Tensor) -> torch.Tensor:
        # [N,V,F,T] -> [N,T,V*F]; view-major, then frequency-major.
        n, v, f, t = views.shape
        h = self.tokenizer(views.permute(0, 3, 1, 2).reshape(n, t, v * f))
        h = self.blocks(h)
        weights = self.attention(h).squeeze(-1).softmax(-1)
        return (h * weights.unsqueeze(-1)).sum(1)


class PeriOnsetEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.time = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, 64))
        self.blocks = nn.Sequential(*[GatedTemporalBlock(dilation=d, dropout=0.1) for d in (1, 2, 4, 8)])
        self.content = nn.Linear(64, 1)
        self.time_bias = nn.Sequential(nn.Linear(1, 16), nn.GELU(), nn.Linear(16, 1))

    def forward(self, windows: torch.Tensor, centers: torch.Tensor, mask: torch.Tensor):
        # [N,W,64], [N,W], [N,W]
        scaled_time = centers.float().unsqueeze(-1) / 30.0
        h = self.blocks(windows + self.time(scaled_time))
        score = (self.content(h) + self.time_bias(scaled_time)).squeeze(-1)
        score = score.masked_fill(~mask, -1e9)
        alpha = score.softmax(-1) * mask.float()
        alpha = alpha / alpha.sum(-1, keepdim=True).clamp_min(1e-12)
        return (h * alpha.unsqueeze(-1)).sum(1), alpha


class ChannelHead(nn.Module):
    def __init__(self, use_z: bool) -> None:
        super().__init__()
        self.use_z = use_z
        self.project = nn.Linear(128 if use_z else 64, 64)
        self.attention = nn.MultiheadAttention(64, 2, batch_first=True)
        self.norm = nn.LayerNorm(64)
        self.classifier = nn.Sequential(nn.Linear(64, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, u: torch.Tensor, mask: torch.Tensor, zero_z: bool = False) -> torch.Tensor:
        if self.use_z:
            valid = mask.float().unsqueeze(-1)
            count = valid.sum(1, keepdim=True).clamp_min(1)
            mean = (u * valid).sum(1, keepdim=True) / count
            std = (((u - mean).square() * valid).sum(1, keepdim=True) / count + 1e-5).sqrt()
            z = (u - mean) / std
            h = self.project(torch.cat([u, torch.zeros_like(z) if zero_z else z], -1))
        else:
            h = self.project(u)
        padding = ~mask
        if padding.all(1).any():
            padding = padding.clone()
            padding[padding.all(1)] = False
        context, _ = self.attention(h, h, h, key_padding_mask=padding, need_weights=False)
        return self.classifier(self.norm(h + context)).squeeze(-1).masked_fill(~mask, 0.0)


class DRSTPatientModel(nn.Module):
    def __init__(self, variant: str, frequency_mean: torch.Tensor, frequency_std: torch.Tensor,
                 chunk_size: int = 256) -> None:
        super().__init__()
        if variant not in VARIANT_VIEWS:
            raise ValueError(variant)
        if frequency_mean.shape != (51,) or frequency_std.shape != (51,):
            raise ValueError("Expected 51 FIT spectral moments")
        self.variant = variant
        self.n_views = VARIANT_VIEWS[variant]
        self.chunk_size = chunk_size
        self.register_buffer("frequency_mean", frequency_mean.float().clone())
        self.register_buffer("frequency_std", frequency_std.float().clone())
        self.window = SpectralWindowEncoder(self.n_views)
        self.peri_onset = PeriOnsetEncoder()
        self.seizure = CrossSeizureMILAggregator(model_dim=64, pooling="mean")
        self.seizure_project = nn.Sequential(nn.Linear(self.seizure.output_dim, 64), nn.GELU())
        self.head = ChannelHead(variant == "S4_DRST_PR")

    def forward(self, batch: dict, *, zero_view: str | None = None, zero_z: bool = False,
                zero_band: tuple[int, int] | None = None) -> dict:
        raw = batch["raw_windows"]
        valid = batch["raw_window_mask"].bool()
        if raw.ndim != 5 or valid.shape != raw.shape[:-1]:
            raise ValueError("Expected raw [B,S,C,W,500] and matching mask")
        logp = torch.empty((*valid.shape, 51, 12), device=raw.device, dtype=torch.float32)
        flat = raw.reshape(-1, 500)
        flat_out = logp.reshape(-1, 51, 12)
        for ids in torch.arange(flat.shape[0], device=raw.device).split(self.chunk_size):
            flat_out[ids] = log_power_stft(flat.index_select(0, ids))
        if zero_band is not None:
            lo, hi = zero_band
            if not 0 <= lo < hi <= 51:
                raise ValueError("Invalid frequency occlusion")
            logp[..., lo:hi, :] = self.frequency_mean[lo:hi].view(1, 1, 1, 1, hi-lo, 1)
        a, d, r = spectral_views(logp, valid, batch["window_centers"], self.frequency_mean, self.frequency_std)
        all_views = {"A": a, "D": d, "R": r}
        keys = ("A", "D", "R")[:self.n_views]
        views = torch.stack([torch.zeros_like(all_views[k]) if k == zero_view else all_views[k] for k in keys], dim=-3)
        # [B,S,C,W,V,F,T]. Process valid windows only in bounded chunks.
        flat_views = views.reshape(-1, self.n_views, 51, 12)
        valid_ids = torch.nonzero(valid.reshape(-1), as_tuple=False).flatten()
        embedding = torch.zeros((flat_views.shape[0], 64), device=raw.device)
        for ids in valid_ids.split(self.chunk_size):
            value = flat_views.index_select(0, ids)
            if self.training and torch.is_grad_enabled():
                h = checkpoint(self.window, value, use_reentrant=False)
            else:
                h = self.window(value)
            embedding = embedding.index_copy(0, ids, h)
        b, s, c, w = valid.shape
        windows = embedding.reshape(b, s, c, w, 64)
        times = batch["window_centers"].float()[:, :, None, :].expand(b, s, c, w)
        seizure_embedding, alpha = self.peri_onset(windows.reshape(-1, w, 64), times.reshape(-1, w), valid.reshape(-1, w))
        seizure_embedding = seizure_embedding.reshape(b, s, c, 64)
        patient_embedding, _ = self.seizure(seizure_embedding, batch["seizure_mask"].bool(),
                                           batch["raw_seizure_channel_mask"].bool())
        u = self.seizure_project(patient_embedding)
        channel_mask = batch["channel_mask"].bool()
        logits = self.head(u, channel_mask, zero_z=zero_z)
        score_nez = torch.sigmoid(logits).masked_fill(~channel_mask, 0.0)
        return {"logits": logits, "score_nez": score_nez,
                "score_ez": (1 - score_nez).masked_fill(~channel_mask, 0.0),
                "patient_channel_embedding": u, "onset_attention": alpha.reshape(b, s, c, w)}


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
