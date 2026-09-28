"""Run before any A1-TF training or outcome access on the original server."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from a1_tf import A1TFModel, freeze_for_ictal_stage
from tf_preprocess import log_frequency_stft


SOURCE_HASH = "c3413ff6d3e7e226919b7b3b1c779c3b62cd65ba78900129540006006fd27e6c"


def parameter_names_and_shapes(model):
    return {name: tuple(value.shape) for name, value in model.named_parameters()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("E:/DRE-nips/new-pipeline/7-11"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    model_path = args.source / "neuroez_c/model.py"
    if hashlib.sha256(model_path.read_bytes()).hexdigest() != SOURCE_HASH:
        raise RuntimeError("Historical A1 model source hash changed")
    sys.path.insert(0, str(args.source))
    from neuroez_c.model import NeuroEZCModel

    settings = SimpleNamespace(model_dim=32, num_heads=2, dropout=0.4,
                               temporal_pooling="mean", record_pooling="mean",
                               use_patient_relative_z=True, positive_label="nez",
                               use_physics_dynamics=False, use_diffusion_residual=False,
                               use_channel_attention=True, use_edf_quality_weighting=False)
    torch.manual_seed(42)
    ictal_base = NeuroEZCModel(settings).eval()
    ictal = A1TFModel(ictal_base).eval()
    torch.manual_seed(43)
    interictal = A1TFModel(NeuroEZCModel(settings)).eval()
    batch = {
        "b0_features": torch.randn(1, 1, 59, 3, 36),
        "tf_log_power": torch.randn(1, 1, 59, 3, 32),
        "seizure_channel_mask": torch.ones(1, 1, 3, dtype=torch.bool),
        "window_mask": torch.ones(1, 1, 59, dtype=torch.bool),
        "seizure_mask": torch.ones(1, 1, dtype=torch.bool),
        "channel_mask": torch.ones(1, 3, dtype=torch.bool),
    }
    with torch.no_grad():
        original = ictal_base(batch)["logits"]
        restored = ictal(batch)["logits"]
        interictal(batch)  # initialize any historical LazyLinear modules
    error = float((original - restored).abs().max())
    mismatch = int((torch.sign(original) != torch.sign(restored)).sum())
    if error >= 1e-6 or mismatch:
        raise RuntimeError(f"alpha=0 exact A1 replay failed: {error}, {mismatch}")
    ictal_topology = parameter_names_and_shapes(ictal)
    interictal_topology = parameter_names_and_shapes(interictal)
    if ictal_topology != interictal_topology:
        raise RuntimeError("Ictal/interictal A1-TF parameter topology differs")
    n_params = sum(value.numel() for value in ictal.parameters())
    if n_params != sum(value.numel() for value in interictal.parameters()):
        raise RuntimeError("Ictal/interictal parameter count differs")
    for rate in (250, 300):
        signal = np.zeros((2, 60 * rate), dtype=np.float32)
        tf = log_frequency_stft(signal, rate)
        if tf.shape != (59, 2, 32) or not np.isfinite(tf).all():
            raise RuntimeError("Physical-bin STFT shape or finiteness failed")
    freeze_for_ictal_stage(ictal, 1)
    if any(p.requires_grad for p in ictal.a1.parameters()):
        raise RuntimeError("Stage 1 did not freeze the A1 core")
    freeze_for_ictal_stage(ictal, 2)
    if any(p.requires_grad for p in ictal.a1.b0_encoder.parameters()):
        raise RuntimeError("Stage 2 changed the historical A1 window encoder")
    result = {
        "pass": True,
        "original_A1_model_sha256": SOURCE_HASH,
        "alpha_zero_max_logit_difference": error,
        "alpha_zero_prediction_mismatch": mismatch,
        "I1_parameters": n_params,
        "N1_parameters": n_params,
        "parameter_topology_identical": True,
        "model_class": "A1TFModel",
        "benchmark_specific_module": False,
        "physical_stft_rates_tested_hz": [250, 300],
        "outcomes_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
