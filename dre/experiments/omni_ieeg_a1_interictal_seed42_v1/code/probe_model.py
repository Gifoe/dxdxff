"""Synthetic shape/memory check of the exact frozen A1 architecture."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch


SOURCE = Path("E:/DRE-nips/new-pipeline/7-11")
sys.path.insert(0, str(SOURCE))
from neuroez_c.model import NeuroEZCModel  # noqa: E402


def main():
    args = SimpleNamespace(model_dim=32, num_heads=2, dropout=0.4,
                           temporal_pooling="mean", record_pooling="mean",
                           use_patient_relative_z=True, positive_label="nez",
                           use_physics_dynamics=False, use_diffusion_residual=False,
                           use_channel_attention=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = NeuroEZCModel(args).to(device)
    b, s, t, c, f = 2, 4, 59, 100, 36
    batch = {
        "features": torch.randn((b, s, t, c, f), device=device),
        "b0_features": torch.randn((b, s, t, c, f), device=device),
        "window_mask": torch.ones((b, s, t), dtype=torch.bool, device=device),
        "seizure_mask": torch.ones((b, s), dtype=torch.bool, device=device),
        "seizure_channel_mask": torch.ones((b, s, c), dtype=torch.bool, device=device),
        "channel_mask": torch.ones((b, c), dtype=torch.bool, device=device),
        "center_id": torch.zeros((b,), dtype=torch.long, device=device),
    }
    output = model(batch)
    loss = output["logits"].square().mean()
    loss.backward()
    print(json.dumps({"pass": True, "shape": list(output["logits"].shape),
                      "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                      "gpu_peak_bytes": torch.cuda.max_memory_allocated() if device == "cuda" else None}))


if __name__ == "__main__":
    main()
