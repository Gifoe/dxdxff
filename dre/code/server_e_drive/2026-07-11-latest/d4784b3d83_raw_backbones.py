from __future__ import annotations

from typing import Any

import torch


class FrozenEmbeddingEncoder(torch.nn.Module):
    def __init__(self, dimension: int) -> None:
        super().__init__()
        self.output_dim = int(dimension)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = values.squeeze(1)
        if output.shape[-1] != self.output_dim:
            raise ValueError(f"Embedding dimension changed: expected {self.output_dim}, got {output.shape[-1]}.")
        return output


class BraindecodeTokenEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, output_dim: int) -> None:
        super().__init__()
        self.model = model
        self.output_dim = int(output_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = self.model(values)
        if isinstance(output, (tuple, list)):
            output = output[0]
        if output.ndim > 2:
            output = output.flatten(1)
        if output.shape[-1] != self.output_dim:
            raise RuntimeError(f"Braindecode encoder returned {tuple(output.shape)}, expected final dimension {self.output_dim}.")
        return output


def build_raw_token_encoder(name: str, *, n_times: int, sfreq: float, embedding_dim: int = 128) -> BraindecodeTokenEncoder:
    try:
        from braindecode import models
    except ImportError as exc:
        raise RuntimeError("braindecode is required for raw supervised baselines.") from exc
    normalized = str(name).lower().replace("-", "").replace("_", "")
    registry: dict[str, tuple[str, dict[str, Any]]] = {
        "eegnet": ("EEGNet", {}),
        "shallowfbcspnet": ("ShallowFBCSPNet", {}),
        "deep4net": ("Deep4Net", {}),
        "eegconformer": ("EEGConformer", {}),
    }
    if normalized not in registry:
        raise ValueError(f"Unknown raw backbone {name!r}.")
    class_name, extra = registry[normalized]
    architecture = getattr(models, class_name, None)
    if architecture is None:
        raise RuntimeError(f"Installed braindecode does not provide {class_name}.")
    kwargs = {"n_chans": 1, "n_outputs": int(embedding_dim), "n_times": int(n_times), "sfreq": float(sfreq), **extra}
    try:
        model = architecture(**kwargs)
    except TypeError as exc:
        raise RuntimeError(f"Braindecode {class_name} API is incompatible with required constructor {kwargs}.") from exc
    return BraindecodeTokenEncoder(model, int(embedding_dim))


__all__ = ["BraindecodeTokenEncoder", "FrozenEmbeddingEncoder", "build_raw_token_encoder"]
