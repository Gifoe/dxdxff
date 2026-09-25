"""Honest environment and completed-run cost reporting; never fabricates latency."""
from __future__ import annotations

import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import pandas as pd


def collect_efficiency(*, output_root: str | Path) -> dict:
    root = Path(output_root); destination = root / "efficiency"; destination.mkdir(parents=True, exist_ok=True)
    hardware = {"os": platform.platform(), "python": sys.version, "cpu_count": os.cpu_count(), "status": "PARTIAL"}
    try:
        import torch
        hardware.update({"torch": torch.__version__, "cuda_available": torch.cuda.is_available(), "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    except Exception as exc:
        hardware["torch_error"] = repr(exc)
    try:
        hardware["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        hardware["git_commit"] = "UNAVAILABLE"
    (destination / "hardware_software.json").write_text(json.dumps(hardware, indent=2), encoding="utf-8")
    unavailable = pd.DataFrame([{"model": model, "status": "UNAVAILABLE_REQUIRES_MEASURED_COMPLETED_CHECKPOINTS", "reason": "Latency and peak memory must be measured on the target GPU after training."} for model in ("PRQ-Net", "BCR-Net", "CDEL")])
    unavailable.to_csv(destination / "model_forward_latency.csv", index=False)
    unavailable.to_csv(destination / "gpu_memory.csv", index=False)
    unavailable.to_csv(destination / "training_cost.csv", index=False)
    unavailable.to_csv(destination / "model_size.csv", index=False)
    return {"status": "partial", "directory": str(destination)}
