from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


@dataclass(frozen=True)
class CheckpointState:
    epoch: int
    best_metric: float
    extra: dict[str, Any]


def _as_cpu_byte_tensor(value: Any) -> torch.Tensor:
    """Normalize a serialized RNG state for PyTorch RNG restoration.

    A checkpoint loaded with map_location="cuda" can move the serialized CPU
    RNG tensor onto CUDA. torch.random.set_rng_state and CUDA generator
    set_state expect a CPU uint8 tensor, so force the canonical representation.
    """
    if torch.is_tensor(value):
        tensor = value.detach()
    else:
        tensor = torch.as_tensor(value)
    return tensor.to(device="cpu", dtype=torch.uint8).contiguous()


def save_checkpoint_atomic(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    best_metric: float,
    extra: Mapping[str, Any] | None = None,
    scaler: Any | None = None,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    payload = {
        "epoch": int(epoch),
        "best_metric": float(best_metric),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.random.get_rng_state(),
        "cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "extra": dict(extra or {}),
    }
    torch.save(payload, temporary)
    os.replace(temporary, output)


def load_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: Any | None = None,
    map_location: str | torch.device = "cpu",
) -> CheckpointState:
    try:
        payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=map_location)

    model.load_state_dict(payload["model_state_dict"])

    if optimizer is not None and payload.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(payload["optimizer_state_dict"])

    if scaler is not None and payload.get("scaler_state_dict") is not None:
        scaler.load_state_dict(payload["scaler_state_dict"])

    if payload.get("random_state") is not None:
        random.setstate(payload["random_state"])

    if payload.get("numpy_random_state") is not None:
        np.random.set_state(payload["numpy_random_state"])

    if payload.get("torch_random_state") is not None:
        torch.random.set_rng_state(
            _as_cpu_byte_tensor(payload["torch_random_state"])
        )

    cuda_random_state = payload.get("cuda_random_state")
    if torch.cuda.is_available() and cuda_random_state is not None:
        if torch.is_tensor(cuda_random_state):
            cuda_random_state = [cuda_random_state]
        torch.cuda.set_rng_state_all(
            [_as_cpu_byte_tensor(state) for state in cuda_random_state]
        )

    return CheckpointState(
        int(payload["epoch"]),
        float(payload["best_metric"]),
        dict(payload.get("extra") or {}),
    )


__all__ = [
    "CheckpointState",
    "load_checkpoint",
    "save_checkpoint_atomic",
]
