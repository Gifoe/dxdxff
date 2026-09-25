"""Lightweight raw iEEG window encoder used by the N6 dual-view model."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _groups(channels: int) -> int:
    return 8 if channels % 8 == 0 else 1


class _DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, kernel_size: int, dilation: int, stride: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1) // 2
        self.depthwise = nn.Conv1d(in_channels, in_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=False)
        self.depth_norm = nn.GroupNorm(_groups(in_channels), in_channels)
        self.pointwise = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.point_norm = nn.GroupNorm(_groups(out_channels), out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Sequential(nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False), nn.GroupNorm(_groups(out_channels), out_channels))
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        residual = self.residual(value)
        value = self.activation(self.depth_norm(self.depthwise(value)))
        value = self.dropout(self.activation(self.point_norm(self.pointwise(value))))
        return self.activation(value + residual)


class RawWindowEncoder(nn.Module):
    """Encode aligned raw windows from ``[B,S,C,W,T]`` to ``[B,S,C,W,D]``."""

    def __init__(
        self,
        *,
        model_dim: int,
        dropout: float = 0.20,
        chunk_size: int = 4096,
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        self.chunk_size = max(1, int(chunk_size))
        self.gradient_checkpointing = bool(gradient_checkpointing)
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.GroupNorm(_groups(32), 32),
            nn.GELU(),
        )
        self.blocks = nn.Sequential(
            _DepthwiseSeparableBlock(32, 32, kernel_size=7, dilation=1, stride=2, dropout=dropout),
            _DepthwiseSeparableBlock(32, 64, kernel_size=7, dilation=2, stride=2, dropout=dropout),
            _DepthwiseSeparableBlock(64, 64, kernel_size=5, dilation=4, stride=1, dropout=dropout),
        )
        self.projection = nn.Sequential(nn.Linear(128, int(model_dim)), nn.LayerNorm(int(model_dim)))

    def _encode_valid(self, windows: torch.Tensor) -> torch.Tensor:
        normalized = windows.to(dtype=torch.float32)
        normalized = (normalized - normalized.mean(dim=-1, keepdim=True)) / torch.sqrt(normalized.var(dim=-1, unbiased=False, keepdim=True) + 1e-5)
        value = self.blocks(self.stem(normalized.unsqueeze(1)))
        pooled = torch.cat([value.mean(dim=-1), value.amax(dim=-1)], dim=-1)
        return self.projection(pooled)

    def forward(self, raw_windows: torch.Tensor, raw_window_mask: torch.Tensor) -> torch.Tensor:
        if raw_windows.ndim != 5 or raw_window_mask.ndim != 4:
            raise ValueError("RawWindowEncoder expects raw_windows [B,S,C,W,T] and raw_window_mask [B,S,C,W].")
        if tuple(raw_windows.shape[:-1]) != tuple(raw_window_mask.shape):
            raise ValueError("raw_window_mask must match raw_windows except for the sample axis.")
        shape = raw_windows.shape[:-1]
        encoder_device = next(self.parameters()).device
        flat_windows = raw_windows.reshape(-1, raw_windows.shape[-1])

        # Keep the large padded source tensor on CPU.  Only valid windows are
        # copied to the encoder device in bounded chunks, avoiding a full
        # [B,S,C,W,T] CUDA allocation for raw waveform batches.
        valid_indices_cpu = torch.nonzero(
            raw_window_mask.detach().to(device="cpu", dtype=torch.bool).reshape(-1),
            as_tuple=False,
        ).flatten()
        output = torch.zeros(
            (flat_windows.shape[0], self.projection[0].out_features),
            dtype=torch.float32,
            device=encoder_device,
        )
        for chunk_cpu in valid_indices_cpu.split(self.chunk_size):
            windows = flat_windows[chunk_cpu].to(
                device=encoder_device,
                dtype=torch.float32,
                non_blocking=bool(raw_windows.device.type == "cpu" and raw_windows.is_pinned()),
            )
            chunk_device = chunk_cpu.to(device=encoder_device)
            if self.training and torch.is_grad_enabled() and self.gradient_checkpointing:
                # Do not retain convolution activations for every raw chunk.
                # They are recomputed during backward, bounding GPU memory by
                # the configured chunk size instead of the full patient batch.
                encoded = checkpoint(self._encode_valid, windows, use_reentrant=False)
            else:
                encoded = self._encode_valid(windows)
            output[chunk_device] = encoded
        output = output.reshape(*shape, output.shape[-1])
        mask = raw_window_mask.to(device=encoder_device, dtype=torch.bool)
        return output.masked_fill(~mask.unsqueeze(-1), 0.0)


__all__ = ["RawWindowEncoder"]
