"""Export frozen residual heads for inference in a NumPy-only runtime."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from scipy.special import erf

from common import atomic_json, sha256
from official_embedding import ResidualHead


def numpy_forward(state, features):
    value = np.asarray(features, dtype=np.float32)
    mean = value.mean(axis=1, keepdims=True)
    variance = ((value - mean) ** 2).mean(axis=1, keepdims=True)
    value = (value - mean) / np.sqrt(variance + np.float32(1e-5))
    value = value * state["network.0.weight"] + state["network.0.bias"]
    value = value @ state["network.1.weight"].T + state["network.1.bias"]
    value = np.float32(0.5) * value * (np.float32(1.0) + erf(value / np.float32(np.sqrt(2.0))))
    value = value @ state["network.4.weight"].T + state["network.4.bias"]
    return (np.float32(0.5) * np.tanh(np.tanh(value))).reshape(-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    arrays = {"freeze_sha256": np.asarray(sha256(args.freeze))}
    max_error = 0.0
    rng = np.random.default_rng(42)
    sample = rng.normal(size=(19, 96)).astype(np.float32)
    for name in ("ABS-ONLY", "PR-CNN"):
        selected = freeze["checkpoints"][name]
        checkpoint = Path(selected["private_path"])
        if sha256(checkpoint) != selected["sha256"]:
            raise RuntimeError("Frozen head checkpoint changed")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        head = ResidualHead()
        head.load_state_dict(payload["model_state_dict"])
        head.eval()
        state = {key: value.detach().cpu().numpy().astype(np.float32)
                 for key, value in head.state_dict().items()}
        arrays.update({f"{name}::{key}": value for key, value in state.items()})
        with torch.inference_mode():
            expected = head(torch.from_numpy(sample)).numpy()
        observed = numpy_forward(state, sample)
        max_error = max(max_error, float(np.max(np.abs(expected - observed))))
    if max_error >= 1e-6:
        raise RuntimeError(f"NumPy head export mismatch: {max_error}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, args.output)
    atomic_json(args.audit, {"status": "PASS", "max_abs_error": max_error,
                             "tolerance": 1e-6, "freeze_sha256": sha256(args.freeze),
                             "private_weight_export_sha256": sha256(args.output),
                             "scientific_change": False})


if __name__ == "__main__":
    main()
