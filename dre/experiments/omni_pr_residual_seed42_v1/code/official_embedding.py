from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import torch
from torch import nn


OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"
CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"


def load_official(path: Path):
    observed = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if observed != OFFICIAL_SOURCE_SHA256:
        raise RuntimeError(f"Official CNN source mismatch: {observed}")
    spec = importlib.util.spec_from_file_location("frozen_official_cnn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class EmbeddingAdapter(nn.Module):
    """Expose the existing 32-D tensor without adding or changing parameters."""
    def __init__(self, raw: nn.Module):
        super().__init__()
        self.raw = raw

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        hidden = self.raw.feature_extractor(image)
        embedding = self.raw.cnn(hidden)
        if embedding.ndim != 2 or embedding.shape[1] != 32:
            raise RuntimeError("Frozen CNN no longer exposes a 32-D representation")
        value = self.raw.bn(self.raw.relu(self.raw.fc(embedding)))
        value = self.raw.bn1(self.raw.relu1(self.raw.fc1(value)))
        return {"logit": self.raw.fc_out(value), "embedding32": embedding}


class ResidualHead(nn.Module):
    """Literal fixed prompt topology, including its outer second tanh."""
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.LayerNorm(96), nn.Linear(96, 16), nn.GELU(), nn.Dropout(0.10),
            nn.Linear(16, 1), nn.Tanh())
        nn.init.zeros_(self.network[-2].weight)
        nn.init.zeros_(self.network[-2].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return 0.5 * torch.tanh(self.network(features)).squeeze(-1)
