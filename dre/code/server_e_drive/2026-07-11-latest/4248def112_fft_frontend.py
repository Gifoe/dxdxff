from __future__ import annotations

import math
import torch
from torch import nn


class SEEGformerFFTFrontend(nn.Module):
    def __init__(self, *, sfreq: float = 200.0, n_times: int = 800, n_fft: int = 1600, freq_min: float = 0.5, freq_max: float = 80.0, eps: float = 1e-6) -> None:
        super().__init__()
        if n_fft < n_times or not 0 <= freq_min < freq_max <= sfreq / 2:
            raise ValueError("Invalid FFT configuration.")
        frequencies = torch.fft.rfftfreq(n_fft, d=1.0 / sfreq)
        selected = torch.nonzero((frequencies >= freq_min) & (frequencies <= freq_max), as_tuple=False).squeeze(1)
        self.register_buffer("frequency_indices", selected, persistent=True)
        self.n_fft, self.eps, self.fft_dim = int(n_fft), float(eps), int(selected.numel())

    def _normalize(self, values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mean = values.mean(dim=-1, keepdim=True)
        std = values.std(dim=-1, keepdim=True).clamp_min(self.eps)
        return ((values - mean) / std).masked_fill(~mask.unsqueeze(-1), 0.0)

    def forward(self, x: torch.Tensor, channel_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        spectrum = torch.fft.rfft(x.float(), n=self.n_fft, dim=-1).index_select(-1, self.frequency_indices)
        real = spectrum.real / math.sqrt(self.n_fft)
        imag = spectrum.imag / math.sqrt(self.n_fft)
        amplitude = torch.log1p(spectrum.abs())
        return tuple(self._normalize(value, channel_mask) for value in (real, imag, amplitude))
