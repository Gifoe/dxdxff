from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class AdapterSpec:
    name: str
    expected_sampling_rate: float
    input_duration_sec: float
    channel_handling: str
    normalization: str
    embedding_layer: str
    output_dim: int | None
    version: str


class FrozenFMAdapter:
    def __init__(self, spec: AdapterSpec) -> None:
        self.spec = spec
        self.model: torch.nn.Module | None = None
        self.loaded = False
        self.audit_info: dict[str, Any] = {**asdict(spec), "frozen": True, "fine_tuned": False}

    def freeze(self) -> None:
        if self.model is None:
            raise RuntimeError(f"{self.spec.name} adapter has no model to freeze.")
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False
        self.loaded = True

    def load(self, *, checkpoint_path: str | Path, external_repo_path: str | Path | None, device: str) -> "FrozenFMAdapter":
        raise NotImplementedError

    def encode_batch(self, waveforms: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    def checkpoint_sha256(path: str | Path) -> str:
        digest = hashlib.sha256()
        with Path(path).open("rb") as handle:
            for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()


__all__ = ["AdapterSpec", "FrozenFMAdapter"]
