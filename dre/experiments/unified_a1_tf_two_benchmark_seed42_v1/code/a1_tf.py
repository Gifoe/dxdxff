"""One A1-TF topology for ictal and interictal patient-channel batches.

The caller supplies 36-dimensional historical A1 views and 32-bin physical
frequency log powers on the *same* 2 s / 1 s window grid. Neither labels nor
patient identifiers enter this module. Reference construction is upstream.
"""

from __future__ import annotations

import torch
from torch import nn


class GatedTemporalBlock(nn.Module):
    def __init__(self, width: int, dilation: int, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.depthwise = nn.Conv1d(width, width, kernel_size=3,
                                   dilation=dilation, padding=dilation,
                                   groups=width)
        self.pointwise = nn.Conv1d(width, 2 * width, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch of patient-record-channels, frames, hidden]
        y = self.norm(x).transpose(1, 2)
        y = self.pointwise(self.depthwise(y)).transpose(1, 2)
        return x + self.dropout(nn.functional.glu(y, dim=-1))


class TimeFrequencyEncoder(nn.Module):
    def __init__(self, n_bins: int = 32, width: int = 64):
        super().__init__()
        self.spectral = nn.Sequential(nn.Linear(n_bins, width), nn.GELU(),
                                      nn.LayerNorm(width))
        self.temporal = nn.Sequential(*[
            GatedTemporalBlock(width, dilation) for dilation in (1, 2, 4, 8)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[-1] != 32:
            raise ValueError(f"Expected [B,R,W,C,32], got {tuple(x.shape)}")
        b, r, w, c, f = x.shape
        series = x.permute(0, 1, 3, 2, 4).reshape(b * r * c, w, f)
        encoded = self.temporal(self.spectral(series))
        return encoded.reshape(b, r, c, w, 64).permute(0, 1, 3, 2, 4)


class A1TFModel(nn.Module):
    """Wrap the *exact* historical NeuroEZCModel without changing its modules.

    `a1` is an initialized historical A1 model. Its modules are invoked in
    exactly their original order. At alpha=0, the classification path is
    algebraically identical to its forward method (eval-mode test required).
    """

    def __init__(self, a1: nn.Module):
        super().__init__()
        self.a1 = a1
        self.tf_encoder = TimeFrequencyEncoder()
        self.tf_projection = nn.Linear(64, a1.model_dim, bias=False)
        self.descriptor_reconstruction = nn.Linear(64, 9)
        self.alpha = nn.Parameter(torch.zeros(()))

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        h_tf = self.tf_encoder(batch["tf_log_power"])
        residual = self.alpha * self.tf_projection(h_tf)
        def inject(_module, _inputs, output):
            if output.shape != residual.shape:
                raise ValueError(f"A1/TF timeline mismatch: {output.shape} vs {residual.shape}")
            return output + residual

        # Calling the historical model's own forward avoids reimplementing
        # optional branches, masks, score sign, or aggregation semantics.
        hook = self.a1.b0_encoder.register_forward_hook(inject)
        try:
            result = self.a1(batch)
        finally:
            hook.remove()
        result.update({
            "tf_descriptor_prediction": self.descriptor_reconstruction(h_tf),
            "tf_residual_rms": residual.square().mean().sqrt(),
            "alpha": self.alpha,
        })
        return result


def freeze_for_ictal_stage(model: A1TFModel, stage: int) -> None:
    """Stage 1 protects all of A1; stage 2 releases only upper A1 modules."""
    if stage not in (1, 2):
        raise ValueError(stage)
    for parameter in model.a1.parameters():
        parameter.requires_grad_(False)
    if stage == 2:
        for module in (model.a1.temporal_encoder, model.a1.seizure_aggregator,
                       model.a1.channel_classifier):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
