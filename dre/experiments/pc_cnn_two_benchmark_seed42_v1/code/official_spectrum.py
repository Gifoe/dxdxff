"""Pinned Omni Morlet-like transform, with native-Nyquist ictal frequency cap."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch


OFFICIAL_CNN_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"


def load_official_module(path: Path):
    path = Path(path)
    if hashlib.sha256(path.read_bytes()).hexdigest() != OFFICIAL_CNN_SHA256:
        raise RuntimeError("Pinned official Omni cnn.py hash mismatch")
    spec = importlib.util.spec_from_file_location("pinned_omni_cnn", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Cannot import pinned official CNN")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class OfficialSpectrum:
    def __init__(self, source_module, channel_chunk: int = 4):
        self.source = source_module
        self.channel_chunk = channel_chunk

    @staticmethod
    def frequency_metadata(sampling_rate: float, device=None):
        max_hz = min(300.0, 0.45 * float(sampling_rate))
        if max_hz <= 10:
            raise ValueError("Sampling rate cannot represent the required 10-Hz minimum")
        # The pinned upstream transform indexes 224 frequencies high -> low.
        hz = torch.linspace(max_hz, 10.0, 224, device=device)
        coord = (hz - 10.0) / (max_hz - 10.0)
        mask = hz <= 0.45 * float(sampling_rate)
        return {"max_frequency_hz": max_hz,
                "normalized_frequency_coordinate": coord,
                "valid_frequency_mask": mask}

    def __call__(self, waves: torch.Tensor, sampling_rate: float) -> torch.Tensor:
        if waves.ndim != 2:
            raise ValueError("Expected channel batch [N,T]")
        metadata = self.frequency_metadata(sampling_rate, device=waves.device)
        pieces = []
        for chunk in waves.split(self.channel_chunk, dim=0):
            with torch.no_grad():
                spectrum = self.source.compute_spectrum_batch(
                    chunk, ps_SampleRate=float(sampling_rate), ps_FreqSeg=224,
                    ps_MinFreqHz=10, ps_MaxFreqHz=metadata["max_frequency_hz"],
                    device=str(chunk.device))
                image = self.source.normalize_img(spectrum.unsqueeze(1).float())
            if not torch.isfinite(image).all():
                raise RuntimeError("Nonfinite pinned Morlet-like spectral image")
            pieces.append(image)
        return torch.cat(pieces, dim=0)


def descriptor_frequency_mask(sampling_rate: float, device=None) -> torch.Tensor:
    """Map physical-frequency availability to the four 9-D A1 view blocks."""
    nyquist_safe = 0.45 * float(sampling_rate)
    # Five A1 bands: delta, theta, beta, low-gamma, high-gamma. The historical
    # descriptor implementation's exact edges are separately source-audited.
    upper_edges = (4.0, 8.0, 30.0, 80.0, 150.0)
    available = torch.tensor([nyquist_safe >= edge for edge in upper_edges]
                             + [True] * 4, dtype=torch.float32, device=device)
    return available.repeat(4)
