"""Raw-iEEG baseline architectures reported in the comparison table."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn
from scipy.signal import butter, filtfilt, iirnotch, resample_poly


def preprocess_raw_window(
    waveform: np.ndarray,
    sampling_rate: float,
    target_rate: int = 200,
    target_samples: int = 800,
) -> np.ndarray:
    """Apply the paper's raw-baseline preprocessing to [channel, sample] data."""
    signal = np.asarray(waveform, dtype=np.float64)
    if signal.ndim != 2:
        raise ValueError("waveform must be [channel, sample]")
    if sampling_rate <= 160:
        raise ValueError("sampling_rate must exceed twice the 80 Hz upper cutoff")
    nyquist = sampling_rate / 2.0
    band_b, band_a = butter(4, (0.5 / nyquist, 80.0 / nyquist), btype="bandpass")
    notch_b, notch_a = iirnotch(50.0 / nyquist, 30.0)
    signal = filtfilt(band_b, band_a, signal, axis=-1)
    signal = filtfilt(notch_b, notch_a, signal, axis=-1)
    common = int(np.gcd(int(round(sampling_rate)), target_rate))
    signal = resample_poly(
        signal,
        target_rate // common,
        int(round(sampling_rate)) // common,
        axis=-1,
    )
    if signal.shape[-1] < target_samples:
        signal = np.pad(signal, ((0, 0), (0, target_samples - signal.shape[-1])))
    signal = signal[..., :target_samples]
    median = np.median(signal, axis=-1, keepdims=True)
    lower, upper = np.quantile(signal, (0.25, 0.75), axis=-1, keepdims=True)
    signal = (signal - median) / np.maximum(upper - lower, 1e-6)
    return np.clip(signal, -8.0, 8.0).astype(np.float32)


class SEEGNet(nn.Module):
    """Three-scale convolution, bidirectional LSTM, and temporal attention."""

    def __init__(self):
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv1d(1, 24, kernel_size=kernel, stride=stride, padding=kernel // 2),
                    nn.BatchNorm1d(24),
                    nn.GELU(),
                )
                for kernel, stride in ((51, 4), (17, 2), (3, 1))
            ]
        )
        self.mixer = nn.Sequential(
            nn.Conv1d(72, 96, kernel_size=1),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.AdaptiveAvgPool1d(32),
        )
        self.recurrent = nn.LSTM(
            96, 64, batch_first=True, bidirectional=True
        )
        self.attention = nn.Linear(128, 1)
        self.head = nn.Linear(128, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        outputs = [branch(waveform) for branch in self.branches]
        length = min(value.shape[-1] for value in outputs)
        combined = torch.cat([value[..., :length] for value in outputs], dim=1)
        sequence = self.mixer(combined).transpose(1, 2)
        sequence, _ = self.recurrent(sequence)
        weights = torch.softmax(self.attention(sequence).squeeze(-1), dim=-1)
        pooled = torch.sum(sequence * weights.unsqueeze(-1), dim=1)
        return self.head(pooled).squeeze(-1)


class TimeConvCNN(nn.Module):
    """STFT image pathway with a locally initialized ResNet-18 backbone."""

    def __init__(self, local_resnet_state: str | Path | None = None):
        super().__init__()
        try:
            from torchvision.models import resnet18
        except ImportError as error:
            raise ImportError("TimeConv-CNN requires the local torchvision package") from error
        self.frontend = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.GELU(),
        )
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            32, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Linear(512, 1)
        if local_resnet_state:
            state = torch.load(local_resnet_state, map_location="cpu", weights_only=True)
            if "conv1.weight" in state and state["conv1.weight"].shape[1] == 3:
                state["conv1.weight"] = state["conv1.weight"].mean(
                    dim=1, keepdim=True
                ).repeat(1, 32, 1, 1)
            state.pop("fc.weight", None)
            state.pop("fc.bias", None)
            self.backbone.load_state_dict(state, strict=False)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        spectrum = torch.stft(
            waveform.squeeze(1),
            n_fft=128,
            hop_length=20,
            window=torch.hann_window(128, device=waveform.device),
            return_complex=True,
        ).abs()
        image = torch.log1p(spectrum)
        mean = image.mean(dim=(-2, -1), keepdim=True)
        scale = image.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        image = (image - mean) / scale
        return self.backbone(self.frontend(image.unsqueeze(1))).squeeze(-1)


class LocalCLAPAdapter(nn.Module):
    """Linear channel head over a locally supplied TorchScript audio encoder.

    The supplement intentionally performs no download or repository access.
    """

    def __init__(self, local_audio_encoder: str | Path, embedding_dimension: int):
        super().__init__()
        self.audio_encoder = torch.jit.load(str(local_audio_encoder), map_location="cpu")
        self.head = nn.Linear(embedding_dimension, 1)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        embedding = self.audio_encoder(waveform.squeeze(1))
        return self.head(embedding).squeeze(-1)


class MaskedTransformerBlock(nn.Module):
    def __init__(self, dimension: int = 64, heads: int = 4, dropout: float = 0.2):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            dimension, heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(dimension)
        self.mlp = nn.Sequential(
            nn.Linear(dimension, 2 * dimension),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * dimension, dimension),
        )
        self.norm2 = nn.LayerNorm(dimension)

    def forward(self, values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attention(
            values,
            values,
            values,
            key_padding_mask=~valid,
            need_weights=False,
        )
        values = self.norm1(values + attended)
        values = self.norm2(values + self.mlp(values))
        return values.masked_fill(~valid.unsqueeze(-1), 0.0)


class SEEGformer(nn.Module):
    """Tri-branch real/imaginary/amplitude frequency-domain Transformer."""

    def __init__(self, fft_bins: int = 641, dimension: int = 64):
        super().__init__()
        self.projections = nn.ModuleList([nn.Linear(fft_bins, dimension) for _ in range(3)])
        self.blocks = nn.ModuleList(
            [
                nn.ModuleList([MaskedTransformerBlock(dimension) for _ in range(2)])
                for _ in range(3)
            ]
        )
        self.heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dimension, 32), nn.GELU(), nn.Linear(32, 1)
                )
                for _ in range(3)
            ]
        )
        self.branch_weights = nn.Parameter(torch.zeros(3))

    def forward(self, waveform: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        # waveform: [batch, channel, 800], zero-padded on the channel axis only.
        spectrum = torch.fft.rfft(waveform, n=1600, dim=-1)
        branches = (spectrum.real[..., :641], spectrum.imag[..., :641], spectrum.abs()[..., :641])
        logits = []
        for branch, projection, blocks, head in zip(
            branches, self.projections, self.blocks, self.heads
        ):
            values = projection(branch)
            for block in blocks:
                values = block(values, channel_mask)
            logits.append(head(values).squeeze(-1))
        weights = torch.softmax(self.branch_weights, dim=0)
        combined = sum(weight * value for weight, value in zip(weights, logits))
        return combined.masked_fill(~channel_mask, 0.0)
