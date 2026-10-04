"""Pinned TimeConv morphology encoder and the audited A1 residual plugin.

The encoder is instantiated from the official Omni ``cnn.py`` source after a
hash check.  We deliberately do not load any Omni-trained checkpoint: the
only pretraining in the model constructor is torchvision's ImageNet ResNet18
initialization, which is external to Omni labels.
"""
from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn


OFFICIAL_CNN_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_official_module(path: Path):
    """Import exactly the audited official source, never a local substitute."""
    if sha256(path) != OFFICIAL_CNN_SHA256:
        raise RuntimeError("Pinned official cnn.py hash mismatch")
    spec = importlib.util.spec_from_file_location("omni_official_cnn_pinned", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import pinned official cnn.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TimeConvMorphology(nn.Module):
    """Official NeuralCNN split into 32-D embedding and its original head."""

    def __init__(self, source_module) -> None:
        super().__init__()
        self.model = source_module.NeuralCNN(in_channels=1, outputs=1)

    def embed_and_logit(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.model.feature_extractor(image)
        embedding = self.model.cnn(features)
        hidden = self.model.bn(self.model.relu(self.model.fc(embedding)))
        hidden = self.model.bn1(self.model.relu1(self.model.fc1(hidden)))
        return embedding, self.model.fc_out(hidden).squeeze(-1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.embed_and_logit(image)[1]


class FixedSegmentSpectrum:
    """Official Morlet-like transform for a fixed, label-blind 2-s segment."""

    def __init__(self, source_module) -> None:
        self.source = source_module

    def __call__(self, waves: torch.Tensor, sampling_rate_hz: float) -> torch.Tensor:
        if waves.ndim != 2:
            raise ValueError("Expected [channels, samples]")
        if int(sampling_rate_hz) != 1000:
            raise RuntimeError("Patient-CV cache must be 1000 Hz")
        # The official transform has no learnable parameters.  It remains out
        # of autograd exactly as in the previously audited implementation.
        with torch.no_grad():
            spectrum = self.source.compute_spectrum_batch(
                waves, ps_SampleRate=1000.0, ps_FreqSeg=224,
                ps_MinFreqHz=10.0, ps_MaxFreqHz=300.0,
                device=str(waves.device))
            image = self.source.normalize_img(spectrum.unsqueeze(1).float())
        if not torch.isfinite(image).all():
            raise RuntimeError("Non-finite official spectral image")
        return image


def patient_relative_z(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """The audited A1 channel-wise patient-relative z operation."""
    expanded = mask.float().unsqueeze(-1)
    count = expanded.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * expanded).sum(dim=1, keepdim=True) / count
    variance = (((x - mean) * expanded) ** 2).sum(dim=1, keepdim=True) / count
    return (x - mean) / torch.sqrt(variance + 1e-5) * expanded


class A1ContextResidual(nn.Module):
    """Two-head A1 attention as a bounded correction to TimeConv logits.

    The final projection is zero-initialized and the effective residual gate is
    nonzero (0.02) so the first forward exactly recovers the base TimeConv
    score while the residual branch can receive gradients on its first update.
    """

    def __init__(self, dim: int = 32, heads: int = 2, dropout: float = 0.25,
                 gate_initial: float = 0.02) -> None:
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        self.hidden = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Dropout(dropout))
        self.output = nn.Linear(dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        # effective_gate = 0.5*tanh(gate_parameter)
        value = max(min(float(gate_initial) / 0.5, 0.999999), -0.999999)
        self.gate_parameter = nn.Parameter(torch.atanh(torch.tensor(value)))

    def effective_gate(self) -> torch.Tensor:
        return 0.5 * torch.tanh(self.gate_parameter)

    def forward(self, anchors: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = patient_relative_z(anchors, mask)
        padding = ~mask
        if bool(padding.all(dim=1).any()):
            padding = padding.clone()
            padding[padding.all(dim=1)] = False
        context, _ = self.attention(z, z, z, key_padding_mask=padding, need_weights=False)
        context = self.norm(z + self.dropout(context))
        delta = self.output(self.hidden(context)).squeeze(-1)
        return (self.effective_gate() * delta).masked_fill(~mask, 0.0)
