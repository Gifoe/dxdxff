from __future__ import annotations

import math
import torch


OUTSIDE_RESIDUAL_FEATURES = (
    "outside_nez_mean", "outside_nez_q05", "outside_nez_q10", "outside_nez_min",
    "outside_residual_mean", "outside_residual_max", "outside_residual_q90", "outside_residual_q95",
    "outside_residual_top05_mean", "outside_residual_top10_mean", "outside_residual_top20_mean",
    "outside_suspicious_fraction_03", "outside_suspicious_fraction_05", "outside_suspicious_fraction_07",
    "inside_nez_mean", "inside_abnormality_mean", "inside_abnormality_top10", "inside_outside_reliable_gap",
)


def _top_mean(value: torch.Tensor, fraction: float) -> torch.Tensor:
    k = min(value.numel(), max(1, math.ceil(value.numel() * fraction)))
    return torch.topk(value, k).values.mean()


def compute_outside_nez_residual(q_consensus_nez: torch.Tensor, reliable_abnormality: torch.Tensor, target: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
    rows = {name: [] for name in OUTSIDE_RESIDUAL_FEATURES}
    residual_channel = torch.where(channel_mask.bool() & ~target.bool(), reliable_abnormality, torch.zeros_like(reliable_abnormality))
    for index in range(q_consensus_nez.shape[0]):
        valid = channel_mask[index].bool(); outside = valid & ~target[index].bool(); inside = valid & target[index].bool()
        if not outside.any() or not inside.any(): raise ValueError("NVR requires at least one target and one outside channel per patient")
        q_out = q_consensus_nez[index][outside]; r_out = reliable_abnormality[index][outside]
        q_in = q_consensus_nez[index][inside]; r_in = reliable_abnormality[index][inside]
        values = {
            "outside_nez_mean": q_out.mean(), "outside_nez_q05": torch.quantile(q_out,.05), "outside_nez_q10": torch.quantile(q_out,.10), "outside_nez_min": q_out.min(),
            "outside_residual_mean": r_out.mean(), "outside_residual_max": r_out.max(), "outside_residual_q90": torch.quantile(r_out,.90), "outside_residual_q95": torch.quantile(r_out,.95),
            "outside_residual_top05_mean": _top_mean(r_out,.05), "outside_residual_top10_mean": _top_mean(r_out,.10), "outside_residual_top20_mean": _top_mean(r_out,.20),
            "outside_suspicious_fraction_03": (q_out < .3).float().mean(), "outside_suspicious_fraction_05": (q_out < .5).float().mean(), "outside_suspicious_fraction_07": (q_out < .7).float().mean(),
            "inside_nez_mean": q_in.mean(), "inside_abnormality_mean": r_in.mean(), "inside_abnormality_top10": _top_mean(r_in,.10), "inside_outside_reliable_gap": r_in.mean()-r_out.mean(),
        }
        for name in OUTSIDE_RESIDUAL_FEATURES: rows[name].append(values[name])
    output = {name: torch.stack(value) for name,value in rows.items()}; output["residual_channel"] = residual_channel
    return output


__all__ = ["OUTSIDE_RESIDUAL_FEATURES", "compute_outside_nez_residual"]
