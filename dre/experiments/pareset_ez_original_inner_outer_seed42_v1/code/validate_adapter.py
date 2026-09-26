"""Pretraining gate: real-cache schema, fit-only transform, and one backward."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source_root.resolve()))
    sys.path.insert(0, str(args.model_root.resolve()))
    from epilens.data import load_records, load_partition_manifest, validate_protocol, select_records
    from epilens.features import fit_standardizer, four_view_expansion
    from pareset_ez import PaReSetEZ, patient_objective, parameter_count

    torch.set_num_threads(2)
    records = load_records(args.data)
    manifest = load_partition_manifest(args.manifest)
    validate_protocol(records, manifest, expected_patients=80)
    fit = select_records(records, manifest, 1, "fit")
    validation = select_records(records, manifest, 1, "validation")
    test = select_records(records, manifest, 1, "test")
    if (len(fit), len(validation), len(test)) != (51, 13, 16):
        raise ValueError("Fold-1 partition differs from frozen audit")
    standardizer = fit_standardizer(fit)
    if standardizer.mean.shape != (36,) or not np.isfinite(standardizer.mean).all() or not np.isfinite(standardizer.scale).all():
        raise ValueError("Invalid fit-only 36-dimensional standardizer")

    example = sorted(fit, key=lambda r: np.prod(r.descriptors.shape[:3]))[len(fit) // 2]
    expanded, mask = four_view_expansion(example.descriptors, example.window_times, example.valid)
    normalized = standardizer.apply(expanded, mask)
    if normalized.shape[-1] != 36 or not np.isfinite(normalized[mask]).all():
        raise ValueError("Invalid expanded valid features")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PaReSetEZ().to(device).train()
    x = torch.as_tensor(normalized, device=device)
    valid = torch.as_tensor(mask, device=device, dtype=torch.bool)
    times = torch.as_tensor(example.window_times, device=device)
    labels = torch.as_tensor(example.label_nez, device=device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    start = time.perf_counter()
    output = model(x, valid, times)
    loss, parts = patient_objective(output["logit_ez"], labels, output["channel_valid"])
    loss.backward()
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    if not bool(torch.isfinite(loss)) or not all(
        torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
    ):
        raise FloatingPointError("Nonfinite real-data loss or parameter gradient")
    result = {
        "status": "PASS", "fold": 1,
        "patients": len(records), "fit": len(fit), "validation": len(validation), "test_membership_only": len(test),
        "expanded_dimension": 36, "standardizer_fit_patients": len(fit),
        "model_parameters": parameter_count(model),
        "example_shape": list(example.descriptors.shape),
        "valid_channels": int(output["channel_valid"].sum().item()),
        "device": str(device), "forward_backward_seconds": elapsed,
        "peak_cuda_mb": round(torch.cuda.max_memory_allocated() / (1024 ** 2), 1) if device.type == "cuda" else None,
        "loss": float(loss.detach().cpu()),
        "bce": float(parts["bce"].detach().cpu()),
        "boundary": float(parts["boundary"].detach().cpu()),
        "gradient_finite": True,
        "test_outcome_evaluated": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
