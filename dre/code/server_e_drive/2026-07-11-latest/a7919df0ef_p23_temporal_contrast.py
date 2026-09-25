"""Onset-aware, masked temporal contrast used by P23-TRN."""

from __future__ import annotations

import torch
from torch import nn


def _masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    weight = mask.to(values.dtype)
    count = weight.sum(dim=dim)
    # Multiplication is not a safe masking operation: NaN * 0 is NaN.
    # Caches may use NaN padding for invalid windows, so replace invalid
    # values before every reduction.
    masked_values = torch.where(mask.unsqueeze(-1), values, torch.zeros_like(values))
    mean = masked_values.sum(dim=dim) / count.clamp_min(1.0).unsqueeze(-1)
    valid = count > 0
    return mean.masked_fill(~valid.unsqueeze(-1), 0.0), valid


class P23TemporalContrastEncoder(nn.Module):
    """Adds a zero-initialized, bounded onset-aware seizure embedding delta."""

    def __init__(
        self, model_dim: int, *, onset_start_sec: float = 0.0, onset_end_sec: float = 10.0,
        spread_end_sec: float = 30.0, max_gate: float = 0.20, gate_init: float = -3.0,
    ) -> None:
        super().__init__()
        if not onset_start_sec < onset_end_sec < spread_end_sec:
            raise ValueError("P23 temporal boundaries must satisfy start < onset_end < spread_end")
        self.onset_start_sec = float(onset_start_sec)
        self.onset_end_sec = float(onset_end_sec)
        self.spread_end_sec = float(spread_end_sec)
        self.max_gate = float(max_gate)
        self.norm = nn.LayerNorm(5 * model_dim)
        self.head = nn.Sequential(
            nn.Linear(5 * model_dim + 5, model_dim), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(model_dim, model_dim), nn.Tanh(),
        )
        nn.init.zeros_(self.head[-2].weight)
        nn.init.zeros_(self.head[-2].bias)
        self.temporal_gate_raw = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(
        self, window_embeddings: torch.Tensor, window_centers: torch.Tensor,
        window_mask: torch.Tensor, seizure_channel_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if window_embeddings.ndim != 5:
            raise ValueError("window_embeddings must be [B,S,T,C,D]")
        bsz, seizures, windows, channels, _ = window_embeddings.shape
        if window_centers.shape != (bsz, seizures, windows):
            raise ValueError("window_centers must be [B,S,T]")
        # Match the established ChannelTemporalEncoder contract: a nominally
        # valid cache window is still excluded when its encoded representation
        # is non-finite. This can occur for sparse/padded channel-window cells.
        finite_window = torch.isfinite(window_embeddings).all(dim=-1)
        base = (
            window_mask.bool().unsqueeze(-1)
            & seizure_channel_mask.bool().unsqueeze(2)
            & finite_window
        )
        centers = window_centers.clamp(-120.0, 120.0).unsqueeze(-1)
        masks = {
            "pre": base & (centers < self.onset_start_sec),
            "onset": base & (centers >= self.onset_start_sec) & (centers <= self.onset_end_sec),
            "spread": base & (centers > self.onset_end_sec) & (centers <= self.spread_end_sec),
            "late": base & (centers > self.spread_end_sec),
        }
        h_all, _ = _masked_mean(window_embeddings, base, dim=2)
        h_pre, valid_pre = _masked_mean(window_embeddings, masks["pre"], dim=2)
        h_onset, valid_onset = _masked_mean(window_embeddings, masks["onset"], dim=2)
        h_spread, valid_spread = _masked_mean(window_embeddings, masks["spread"], dim=2)
        h_late, valid_late = _masked_mean(window_embeddings, masks["late"], dim=2)

        def difference(left: torch.Tensor, left_valid: torch.Tensor, right: torch.Tensor, right_valid: torch.Tensor) -> torch.Tensor:
            enabled = left_valid & right_valid
            return (left - right).masked_fill(~enabled.unsqueeze(-1), 0.0)

        delta_onset = difference(h_onset, valid_onset, h_pre, valid_pre)
        delta_spread = difference(h_spread, valid_spread, h_onset, valid_onset)
        delta_late = difference(h_late, valid_late, h_spread, valid_spread)
        slope_mask = base
        count = slope_mask.sum(dim=2)
        t = centers.expand(-1, -1, -1, channels)
        weights = slope_mask.to(window_embeddings.dtype)
        masked_t = torch.where(slope_mask, t, torch.zeros_like(t))
        masked_h = torch.where(slope_mask.unsqueeze(-1), window_embeddings, torch.zeros_like(window_embeddings))
        t_mean = masked_t.sum(dim=2) / count.clamp_min(1).to(t.dtype)
        h_mean = masked_h.sum(dim=2) / count.clamp_min(1).unsqueeze(-1).to(window_embeddings.dtype)
        tc = torch.where(slope_mask, t - t_mean.unsqueeze(2), torch.zeros_like(t))
        hc = torch.where(slope_mask.unsqueeze(-1), window_embeddings - h_mean.unsqueeze(2), torch.zeros_like(window_embeddings))
        denom = tc.square().sum(dim=2)
        slope_valid = count >= 2
        slope = (tc.unsqueeze(-1) * hc).sum(dim=2) / denom.clamp_min(1e-6).unsqueeze(-1)
        slope = slope.masked_fill(~slope_valid.unsqueeze(-1), 0.0)
        flags = torch.stack((valid_pre, valid_onset, valid_spread, valid_late, slope_valid), dim=-1).to(window_embeddings.dtype)
        features = torch.cat((h_all, delta_onset, delta_spread, delta_late, slope), dim=-1)
        raw = self.head(torch.cat((self.norm(features), flags), dim=-1))
        gate = self.max_gate * torch.sigmoid(self.temporal_gate_raw)
        delta = gate * raw
        valid_channel = base.any(dim=2)
        delta = delta.masked_fill(~valid_channel.unsqueeze(-1), 0.0)
        return {
            "temporal_delta": delta,
            "temporal_gate": torch.ones_like(valid_channel, dtype=window_embeddings.dtype) * gate,
            "delta_onset_norm": delta_onset.norm(dim=-1), "delta_spread_norm": delta_spread.norm(dim=-1),
            "delta_late_norm": delta_late.norm(dim=-1), "slope_norm": slope.norm(dim=-1),
            "valid_pre": valid_pre, "valid_onset": valid_onset, "valid_spread": valid_spread,
            "valid_late": valid_late, "slope_valid": slope_valid,
            "temporal_bin_valid_fraction": flags[..., :4].mean(dim=-1),
        }
