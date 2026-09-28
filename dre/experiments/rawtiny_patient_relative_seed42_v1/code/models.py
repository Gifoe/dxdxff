"""Small raw-window and patient-relative models for the frozen A1 cohort.

Inputs follow the repository's validated RawAlignmentStore and patient-batch
layout.  There is no center or patient identifier in any forward pass.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from seizure_aggregator import CrossSeizureMILAggregator
from temporal_encoder import ChannelTemporalEncoder


def scores_from_nez_logits(logits: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    logits = logits.masked_fill(~mask, -1e9)
    nez = torch.sigmoid(logits).masked_fill(~mask, 0.0)
    ez = (1.0 - nez).masked_fill(~mask, 0.0)
    return {"logits": logits, "scores": nez, "score_nez": nez, "score_ez": ez}


class _Branch(nn.Module):
    def __init__(self, kernel: int, dilation: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv1d(16, 16, kernel, padding=dilation * (kernel - 1) // 2,
                                   dilation=dilation, groups=16, bias=False)
        self.pointwise = nn.Conv1d(16, 16, 1, bias=False)
        self.norm = nn.GroupNorm(4, 16)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.pointwise(self.depthwise(x)))).mean(dim=-1)


class RawTinyEncoder(nn.Module):
    """250 Hz, 2 s waveform to 32D; receptive fields 28/116/484 ms.

    No BatchNorm, global patient statistics or absolute center information.
    Valid windows are processed in bounded chunks with activation recompute.
    """

    def __init__(self, chunk_size: int = 256) -> None:
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 16, 15, stride=2, padding=7, bias=False),
            nn.GroupNorm(4, 16), nn.GELU(),
        )
        self.branches = nn.ModuleList((_Branch(7, 1), _Branch(15, 2), _Branch(31, 4)))
        self.projection = nn.Sequential(nn.Linear(48, 32), nn.LayerNorm(32))

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float()
        x = (x - x.mean(dim=-1, keepdim=True)) / x.std(dim=-1, unbiased=False, keepdim=True).clamp_min(1e-6)
        x = x.clamp(-8.0, 8.0)
        h = self.stem(x.unsqueeze(1))
        return self.projection(torch.cat([branch(h) for branch in self.branches], dim=-1))

    def forward(self, raw: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if raw.ndim != 5 or mask.shape != raw.shape[:-1]:
            raise ValueError("Expected raw [B,S,C,W,T] and mask [B,S,C,W]")
        shape = mask.shape
        flat = raw.reshape(-1, raw.shape[-1])
        device = next(self.parameters()).device
        valid = torch.nonzero(mask.detach().to("cpu").reshape(-1), as_tuple=False).flatten()
        out = torch.zeros((flat.shape[0], 32), device=device, dtype=torch.float32)
        for ids_cpu in valid.split(self.chunk_size):
            ids_source = ids_cpu.to(flat.device)
            x = flat.index_select(0, ids_source).to(device=device, dtype=torch.float32)
            if self.training and torch.is_grad_enabled():
                z = checkpoint(self._encode, x, use_reentrant=False)
            else:
                z = self._encode(x)
            out = out.index_copy(0, ids_cpu.to(device), z)
        return out.reshape(*shape, 32)


class PatientRelativeHead(nn.Module):
    def __init__(self, dim: int = 64, *, residual: bool = False) -> None:
        super().__init__()
        self.rank_scorer = nn.Linear(dim, 1, bias=False)
        self.input = nn.Linear(2 * dim + 1, dim)
        self.attn = nn.MultiheadAttention(dim, 2, dropout=0.0, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.classifier = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, 1))
        if residual:
            nn.init.zeros_(self.classifier[-1].weight)
            nn.init.zeros_(self.classifier[-1].bias)

    def relative_parts(self, u: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = mask.float().unsqueeze(-1)
        count = valid.sum(1, keepdim=True).clamp_min(1.0)
        mu = (u * valid).sum(1, keepdim=True) / count
        sigma = (((u - mu).square() * valid).sum(1, keepdim=True) / count + 1e-5).sqrt()
        z = ((u - mu) / sigma) * valid
        q = self.rank_scorer(u).squeeze(-1)
        pct = torch.zeros_like(q)
        for b in range(q.shape[0]):
            indices = torch.nonzero(mask[b], as_tuple=False).flatten()
            if indices.numel() > 1:
                order = torch.argsort(q[b, indices], stable=True)
                ranks = torch.empty_like(order)
                ranks[order] = torch.arange(len(indices), device=q.device)
                pct[b, indices] = ranks.to(q.dtype) / (len(indices) - 1)
        # Exact label-free percentile in the forward pass.  A smooth-rank
        # straight-through gradient makes the preliminary scorer learnable;
        # fully detached ranks would leave rank_scorer permanently random.
        pair = torch.sigmoid((q.unsqueeze(-1) - q.unsqueeze(-2)) / 0.1)
        pair = pair * mask.unsqueeze(1).to(pair.dtype)
        smooth = (pair.sum(dim=-1) - 0.5) / (mask.sum(dim=1, keepdim=True) - 1).clamp_min(1)
        smooth = smooth * mask.to(smooth.dtype)
        pct = pct + smooth - smooth.detach()
        return z, pct.unsqueeze(-1)

    def forward(self, u: torch.Tensor, mask: torch.Tensor,
                *, zero_z: bool = False, zero_rank: bool = False) -> torch.Tensor:
        z, pct = self.relative_parts(u, mask)
        if zero_z:
            z = torch.zeros_like(z)
        if zero_rank:
            pct = torch.zeros_like(pct)
        h = self.input(torch.cat([u, z, pct], dim=-1))
        padding = ~mask
        if padding.all(dim=1).any():
            padding = padding.clone()
            padding[padding.all(dim=1)] = False
        context, _ = self.attn(h, h, h, key_padding_mask=padding)
        return self.classifier(self.norm(h + context)).squeeze(-1).masked_fill(~mask, 0.0)


class RawTinyPatientModel(nn.Module):
    def __init__(self, *, patient_relative: bool) -> None:
        super().__init__()
        self.raw_encoder = RawTinyEncoder()
        self.temporal = ChannelTemporalEncoder(model_dim=32, pooling="mean")
        self.seizure = CrossSeizureMILAggregator(model_dim=32, pooling="mean")
        self.patient_relative = bool(patient_relative)
        self.head = (PatientRelativeHead(self.seizure.output_dim) if patient_relative else
                     nn.Sequential(nn.Linear(self.seizure.output_dim, 32), nn.GELU(), nn.Linear(32, 1)))

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        w = self.raw_encoder(batch["raw_windows"], batch["raw_window_mask"]).permute(0, 1, 3, 2, 4)
        seizure, _ = self.temporal(w, batch["seizure_channel_mask"], window_mask=batch["window_mask"])
        u, _ = self.seizure(seizure, batch["seizure_mask"], batch["raw_seizure_channel_mask"])
        mask = batch["channel_mask"]
        logits = self.head(u, mask) if self.patient_relative else self.head(u).squeeze(-1)
        out = scores_from_nez_logits(logits, mask)
        out["patient_channel_embedding"] = u
        out["raw_window_embedding"] = w
        return out


class HybridA1RawModel(nn.Module):
    """A1 plus zero-initialized raw-window residual and zero-output PR correction."""

    def __init__(self, a1: nn.Module) -> None:
        super().__init__()
        self.a1 = a1
        self.a1.eval()
        for parameter in self.a1.parameters():
            parameter.requires_grad_(False)
        self.raw_encoder = RawTinyEncoder()
        self.raw_projection = nn.Linear(32, 32, bias=False)
        self.alpha = nn.Parameter(torch.zeros(()))
        self.pr_residual = PatientRelativeHead(a1.seizure_aggregator.output_dim, residual=True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.a1.eval()
        return self

    def forward(self, batch: dict) -> dict[str, torch.Tensor]:
        raw = self.raw_encoder(batch["raw_windows"], batch["raw_window_mask"]).permute(0, 1, 3, 2, 4)
        delta = self.alpha * self.raw_projection(raw)

        def inject(_module, _inputs, output):
            if output.shape != delta.shape:
                raise ValueError(f"A1/raw window embedding mismatch: {output.shape} != {delta.shape}")
            return output + delta

        hook = self.a1.b0_encoder.register_forward_hook(inject)
        try:
            base = self.a1(batch)
        finally:
            hook.remove()
        mask = batch["channel_mask"]
        correction = self.pr_residual(base["patient_channel_embedding"], mask)
        out = dict(base)
        out.update(scores_from_nez_logits(base["logits"] + correction, mask))
        out["hybrid_alpha"] = self.alpha
        out["raw_residual_norm"] = delta.square().mean().sqrt()
        out["raw_window_embedding"] = raw
        return out


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

