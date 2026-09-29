"""Single raw-iEEG CRST architecture shared by both benchmark training runs.

The deterministic physical-frequency front end is implemented in
``spectral_cache.py``. No patient, center, channel-index or seizure-state ID is
ever passed to this module. Shapes are B,R,C,W,F,L; masks are B,R,C,W.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from torch.nn import functional as F


def masked_median_mad(x: Tensor, mask: Tensor, dim: int, eps: float = 1e-4):
    """Robust normalization with finite outputs for empty padded slices."""
    valid = mask.unsqueeze(-1).expand_as(x)
    masked = x.masked_fill(~valid, torch.nan)
    center = torch.nanmedian(masked, dim=dim, keepdim=True).values
    center = torch.nan_to_num(center)
    dev = (x - center).abs().masked_fill(~valid, torch.nan)
    spread = torch.nanmedian(dev, dim=dim, keepdim=True).values
    spread = torch.nan_to_num(spread) * 1.4826 + eps
    return torch.where(valid, (x - center) / spread, 0)


class CanonicalSpectralTokenizer(nn.Module):
    """Learned common patch encoder; frequency grid/mask come from raw fs."""

    def __init__(self, frequencies: int = 64, subdivisions: int = 8, d: int = 96):
        super().__init__()
        self.activation_checkpointing = True
        self.frequencies = frequencies
        self.subdivisions = subdivisions
        self.embedding_dim = d
        self.encoder = nn.Sequential(
            nn.Conv2d(2, 16, 3, padding=1), nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1), nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 2)), nn.Flatten(),
            nn.Linear(32 * 4 * 2, d),
        )

    def forward(self, patch: Tensor, frequency_mask: Tensor,
                chunk_size: int = 2048) -> Tensor:
        if patch.ndim != 6 or patch.shape[-2:] != (self.frequencies, self.subdivisions):
            raise ValueError("Expected B,R,C,W,64,8 spectral patches")
        b = patch.shape[0]
        if frequency_mask.shape != (b, self.frequencies):
            raise ValueError("Physical-frequency availability mask shape mismatch")
        p = patch * frequency_mask[:, None, None, None, :, None]
        availability = frequency_mask[:, None, None, None, :, None].expand_as(p)
        flat = torch.stack((p, availability), -3).reshape(-1, 2, self.frequencies,
                                                           self.subdivisions)
        outputs = []
        for chunk in flat.split(chunk_size):
            if self.training and self.activation_checkpointing:
                outputs.append(torch.utils.checkpoint.checkpoint(self.encoder, chunk,
                                                                 use_reentrant=False))
            else:
                outputs.append(self.encoder(chunk))
        return torch.cat(outputs,
                         dim=0).reshape(*patch.shape[:4], self.embedding_dim)


class MaskedMultiheadAttention(nn.Module):
    """Permutation-equivariant attention unless an explicit temporal bias is used."""

    def __init__(self, d: int = 128, heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.d, self.heads, self.head_dim = d, heads, d // heads
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, valid: Tensor, bias: Tensor | None = None) -> Tensor:
        # x: N,S,D; bias broadcastable to N,H,S,S.
        n, s, d = x.shape
        q, k, v = self.qkv(x).reshape(n, s, 3, self.heads,
                                       self.head_dim).permute(2, 0, 3, 1, 4)
        logits = q @ k.transpose(-1, -2) / math.sqrt(self.head_dim)
        if bias is not None:
            logits = logits + bias
        logits = logits.masked_fill(~valid[:, None, None, :], -1e4)
        weights = self.dropout(torch.softmax(logits, dim=-1))
        out = (weights @ v).transpose(1, 2).reshape(n, s, d)
        return self.proj(out) * valid.unsqueeze(-1)


class CRSTBlock(nn.Module):
    def __init__(self, d: int = 128, heads: int = 8, dropout: float = 0.1,
                 maximum_windows: int = 59):
        super().__init__()
        self.temporal = MaskedMultiheadAttention(d, heads, dropout)
        self.channel = MaskedMultiheadAttention(d, heads, dropout)
        self.record = MaskedMultiheadAttention(d, heads, dropout)
        self.norms = nn.ModuleList(nn.LayerNorm(d) for _ in range(4))
        self.time_relative_bias = nn.Parameter(torch.zeros(heads, 2 * maximum_windows - 1))
        self.edge_bias = nn.Sequential(nn.Linear(15, 32), nn.GELU(), nn.Linear(32, heads))
        self.lambda_net = nn.Parameter(torch.zeros(()))
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(),
                                nn.Dropout(dropout), nn.Linear(4 * d, d),
                                nn.Dropout(dropout))

    def forward(self, x: Tensor, mask: Tensor, edges: Tensor,
                channel_attention: bool = True,
                connectivity_on: bool = True) -> Tensor:
        b, r, c, w, d = x.shape
        if w > (self.time_relative_bias.shape[1] + 1) // 2:
            raise ValueError("Window count exceeds frozen temporal-bias support")
        pos = torch.arange(w, device=x.device)
        rel = pos[:, None] - pos[None, :] + self.time_relative_bias.shape[1] // 2
        tbias = self.time_relative_bias[:, rel][None]
        xt = self.norms[0](x).reshape(b * r * c, w, d)
        mt = mask.reshape(b * r * c, w)
        x = x + self.temporal(xt, mt, tbias).reshape(b, r, c, w, d)
        x = x * mask.unsqueeze(-1)

        if channel_attention:
            xc = self.norms[1](x).permute(0, 1, 3, 2, 4).reshape(b * r * w, c, d)
            mc = mask.permute(0, 1, 3, 2).reshape(b * r * w, c)
            ebias = None
            if connectivity_on:
                ebias = self.edge_bias(edges).permute(0, 1, 4, 2, 3)
                ebias = ebias[:, :, None].expand(b, r, w, -1, -1, -1)
                ebias = (self.lambda_net * ebias).reshape(b * r * w, -1, c, c)
            delta = self.channel(xc, mc, ebias)
            x = x + delta.reshape(b, r, w, c, d).permute(0, 1, 3, 2, 4)
            x = x * mask.unsqueeze(-1)

        xr = self.norms[2](x).permute(0, 2, 3, 1, 4).reshape(b * c * w, r, d)
        mr = mask.permute(0, 2, 3, 1).reshape(b * c * w, r)
        delta = self.record(xr, mr).reshape(b, c, w, r, d).permute(0, 3, 1, 2, 4)
        x = (x + delta) * mask.unsqueeze(-1)
        return (x + self.ff(self.norms[3](x))) * mask.unsqueeze(-1)


class CRSTiEEG(nn.Module):
    """Common model class for ictal and interictal records."""

    def __init__(self, frequencies: int = 64, subdivisions: int = 8,
                 patch_dim: int = 96, d: int = 128, heads: int = 8,
                 blocks: int = 4, dropout: float = 0.1,
                 activation_checkpointing: bool = True):
        super().__init__()
        self.activation_checkpointing = activation_checkpointing
        self.tokenizer = CanonicalSpectralTokenizer(frequencies, subdivisions,
                                                     patch_dim)
        self.tokenizer.activation_checkpointing = activation_checkpointing
        self.token_projection = nn.Sequential(
            nn.Linear(4 * patch_dim + frequencies + 1, d), nn.LayerNorm(d))
        self.blocks = nn.ModuleList(CRSTBlock(d, heads, dropout) for _ in range(blocks))
        self.window_pool = nn.Linear(d, 1)
        self.record_pool = nn.Linear(d, 1)
        self.pma_seed = nn.Parameter(torch.zeros(1, 1, d))
        self.patient_pma = nn.MultiheadAttention(d, heads, dropout=dropout,
                                                  batch_first=True)
        self.classifier = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(),
                                        nn.Dropout(dropout), nn.Linear(d, 32),
                                        nn.GELU(), nn.Linear(32, 1))
        self.mask_decoder = nn.Linear(d, patch_dim)

    @staticmethod
    def _weighted_pool(x: Tensor, mask: Tensor, scorer: nn.Module, dim: int):
        logits = scorer(x).squeeze(-1).masked_fill(~mask, -1e4)
        weight = torch.softmax(logits, dim=dim) * mask
        weight = weight / weight.sum(dim=dim, keepdim=True).clamp_min(1e-6)
        return (x * weight.unsqueeze(-1)).sum(dim=dim)

    def forward(self, patches: Tensor, frequency_mask: Tensor, window_mask: Tensor,
                edges: Tensor, *, intervention: str | None = None,
                return_aux: bool = False):
        b, r, c, w, f, l = patches.shape
        if window_mask.shape != (b, r, c, w) or edges.shape != (b, r, c, c, 15):
            raise ValueError("CRST masks/connectivity have incompatible axes")
        u = self.tokenizer(patches, frequency_mask)
        a = u * window_mask.unsqueeze(-1)
        t = masked_median_mad(u, window_mask, dim=3)
        cc = masked_median_mad(u, window_mask, dim=2)
        # Channel indexing is canonical within a patient. Unobserved channels
        # have mask false and do not contribute to record-relative coordinates.
        rr = masked_median_mad(u, window_mask, dim=1)
        record_avail = (window_mask.sum(dim=1, keepdim=True) >= 2)
        rr = rr * record_avail.unsqueeze(-1)
        if intervention in {"A_ONLY", "T_ZERO"}:
            t = torch.zeros_like(t)
        if intervention in {"A_ONLY", "C_ZERO"}:
            cc = torch.zeros_like(cc)
        if intervention in {"A_ONLY", "R_ZERO"}:
            rr = torch.zeros_like(rr)
        availability = frequency_mask[:, None, None, None, :].expand(b, r, c, w, f)
        ra = record_avail.float().expand(b, r, c, w).unsqueeze(-1)
        x = self.token_projection(torch.cat((a, t, cc, rr, availability, ra), dim=-1))
        x = x * window_mask.unsqueeze(-1)
        for block in self.blocks:
            def run_block(value, block=block):
                return block(value, window_mask, edges,
                             channel_attention=intervention != "NO_CHANNEL_ATTENTION",
                             connectivity_on=intervention != "CONNECTIVITY_ZERO")
            if self.training and self.activation_checkpointing:
                x = torch.utils.checkpoint.checkpoint(run_block, x, use_reentrant=False)
            else:
                x = run_block(x)
        channel_record_mask = window_mask.any(dim=3)
        h_record = self._weighted_pool(x, window_mask, self.window_pool, dim=3)
        channel_mask = channel_record_mask.any(dim=1)
        h = self._weighted_pool(h_record, channel_record_mask, self.record_pool, dim=1)
        h_rel = masked_median_mad(h, channel_mask, dim=1)
        seed = self.pma_seed.expand(b, -1, -1)
        summary = self.patient_pma(seed, h, h,
                                   key_padding_mask=~channel_mask,
                                   need_weights=False)[0].expand(-1, c, -1)
        score = self.classifier(torch.cat((h, h_rel, summary), dim=-1)).squeeze(-1)
        score = score.masked_fill(~channel_mask, 0)
        if return_aux:
            record_q = torch.cat((h_record, h_rel[:, None].expand(-1, r, -1, -1),
                                  summary[:, None].expand(-1, r, -1, -1)), dim=-1)
            record_score = self.classifier(record_q).squeeze(-1)
            record_score = record_score.masked_fill(~channel_record_mask, 0)
            return {"logits": score, "token_embedding": u, "contextual": x,
                    "record_embedding": h_record, "record_logits": record_score,
                    "channel_embedding": h,
                    "channel_mask": channel_mask,
                    "relative_rms": torch.stack((t.square().mean().sqrt(),
                                                  cc.square().mean().sqrt(),
                                                  rr.square().mean().sqrt()))}
        return score


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
