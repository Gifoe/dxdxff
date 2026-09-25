from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .base import AdapterSpec, FrozenFMAdapter


class BrainBERTAdapter(FrozenFMAdapter):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(
            AdapterSpec(
                name="brainbert",
                expected_sampling_rate=float(config.get("target_sfreq", 200)),
                input_duration_sec=float(config.get("window_sec", 4.0)),
                channel_handling="single_seeg_contact",
                normalization="checkpoint_spectrogram_normalizer",
                embedding_layer="masked_patch_encoder_hidden_mean",
                output_dim=int(config.get("embedding_dim", 128)),
                version=str(config.get("version", "rawbrainbert-v1")),
            )
        )
        self.config = dict(config)

    def load(self, *, checkpoint_path: str | Path, external_repo_path: str | Path | None, device: str) -> "BrainBERTAdapter":
        del checkpoint_path, external_repo_path, device
        raise RuntimeError(
            "Official author BrainBERT loading is not implemented. The repository's RawBrainBERT checkpoint is a different model and cannot be used for this baseline."
        )

    def encode_batch(self, waveforms: np.ndarray) -> np.ndarray:
        del waveforms
        raise RuntimeError("Official author BrainBERT loading is unavailable; embeddings cannot be generated.")


__all__ = ["BrainBERTAdapter"]
