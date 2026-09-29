"""Independent TF scorer and score-only A1/TF late fusion.

No A1 modules, embeddings, logits or descriptors enter TFScorer.forward.
The same scorer is used on ictal and interictal batches.
"""
from __future__ import annotations

import torch
from torch import nn

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "unified_a1_tf_two_benchmark_seed42_v1" / "code"))
from a1_tf import TimeFrequencyEncoder


def robust_channel_normalize(score: torch.Tensor, mask: torch.Tensor, eps: float = 1e-6):
    """Per patient median/MAD of active channel scores; padding remains zero."""
    if score.ndim != 2 or score.shape != mask.shape:
        raise ValueError("Expected matching [patient, channel] score/mask")
    normalized = torch.zeros_like(score)
    for p in range(score.shape[0]):
        active = mask[p].bool()
        if not active.any():
            raise ValueError("Patient without active channels")
        values = score[p, active]
        median = values.median()
        mad = (values - median).abs().median()
        normalized[p, active] = (values - median) / (1.4826 * mad + eps)
    return normalized


class TFScorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = TimeFrequencyEncoder()
        self.head = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Dropout(0.1), nn.Linear(64, 1))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        encoded = self.encoder(batch["tf_log_power"])  # [B,R,W,C,64]
        windows = batch["window_mask"].bool()[:, :, :, None, None]
        summed = (encoded * windows).sum(dim=2)
        count = windows.sum(dim=2).clamp_min(1)
        record_embedding = summed / count
        records = batch["seizure_channel_mask"].bool()[:, :, :, None]
        channel_embedding = (record_embedding * records).sum(dim=1) / records.sum(dim=1).clamp_min(1)
        channel_mask = batch["channel_mask"].bool()
        relative = torch.zeros_like(channel_embedding)
        for p in range(channel_embedding.shape[0]):
            active = channel_mask[p]
            if not active.any():
                raise ValueError("Patient without active channel")
            values = channel_embedding[p, active]
            median = values.median(dim=0).values
            mad = (values - median).abs().median(dim=0).values
            relative[p, active] = (values - median) / (1.4826 * mad + 1e-6)
        margin = self.head(torch.cat([channel_embedding, relative], dim=-1)).squeeze(-1)
        return {"margin_tf": margin, "embedding_tf": channel_embedding,
                "relative_embedding_tf": relative}


def fused_margin(margin_a1: torch.Tensor, margin_tf: torch.Tensor,
                 channel_mask: torch.Tensor, beta: float) -> torch.Tensor:
    if beta == 0:
        return margin_a1.clone()
    return margin_a1 + float(beta) * robust_channel_normalize(margin_tf, channel_mask)
