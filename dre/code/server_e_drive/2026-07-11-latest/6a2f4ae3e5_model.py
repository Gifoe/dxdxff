from __future__ import annotations

import torch
from torch import nn

from .fft_frontend import SEEGformerFFTFrontend
from .transformer import MaskedTransformerBlock


class SEEGformerBranch(nn.Module):
    def __init__(self, fft_dim: int, embed_dim: int, num_heads: int, num_blocks: int, mlp_ratio: float, dropout: float) -> None:
        super().__init__()
        self.projection = nn.Linear(fft_dim, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.blocks = nn.ModuleList([MaskedTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(num_blocks)])
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 32), nn.GELU(), nn.Dropout(dropout), nn.Linear(32, 1))

    def forward(self, frequency_features: torch.Tensor, channel_mask: torch.Tensor, *, return_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch = frequency_features.shape[0]
        values = self.projection(frequency_features)
        values = torch.cat([self.cls.expand(batch, -1, -1), values], dim=1)
        padding = torch.cat([torch.zeros((batch, 1), dtype=torch.bool, device=values.device), ~channel_mask], dim=1)
        attention = None
        for index, block in enumerate(self.blocks):
            values, weights = block(values, padding, return_attention=return_attention and index == len(self.blocks) - 1)
            if weights is not None:
                attention = weights
        logits = self.head(values[:, 1:]).squeeze(-1).masked_fill(~channel_mask, 0.0)
        return logits, attention


class SEEGformerTask1(nn.Module):
    model_name = "seegformer"
    output_type = "per_channel_logits"

    def __init__(self, *, sfreq: float = 200.0, n_times: int = 800, n_fft: int = 1600, freq_min: float = 0.5, freq_max: float = 80.0, embed_dim: int = 64, num_heads: int = 4, num_blocks: int = 2, mlp_ratio: float = 2.0, dropout: float = 0.2) -> None:
        super().__init__()
        self.frontend = SEEGformerFFTFrontend(sfreq=sfreq, n_times=n_times, n_fft=n_fft, freq_min=freq_min, freq_max=freq_max)
        args = (self.frontend.fft_dim, embed_dim, num_heads, num_blocks, mlp_ratio, dropout)
        self.real_branch, self.imag_branch, self.amplitude_branch = (SEEGformerBranch(*args) for _ in range(3))
        self.branch_weight_logits = nn.Parameter(torch.zeros(3))
        self.input_sfreq, self.n_times, self.fft_dim = sfreq, n_times, self.frontend.fft_dim

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())

    @property
    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def forward(self, waveform: torch.Tensor, channel_mask: torch.Tensor, *, return_attention: bool = False) -> dict[str, torch.Tensor | None]:
        real, imag, amplitude = self.frontend(waveform, channel_mask)
        real_logits, real_attention = self.real_branch(real, channel_mask, return_attention=return_attention)
        imag_logits, imag_attention = self.imag_branch(imag, channel_mask, return_attention=return_attention)
        amplitude_logits, amplitude_attention = self.amplitude_branch(amplitude, channel_mask, return_attention=return_attention)
        weights = torch.softmax(self.branch_weight_logits, dim=0)
        logits = (weights[0] * real_logits + weights[1] * imag_logits + weights[2] * amplitude_logits).masked_fill(~channel_mask, 0.0)
        return {"channel_logits": logits, "real_logits": real_logits, "imag_logits": imag_logits, "amplitude_logits": amplitude_logits, "branch_weights": weights, "real_attention": real_attention, "imag_attention": imag_attention, "amplitude_attention": amplitude_attention}
