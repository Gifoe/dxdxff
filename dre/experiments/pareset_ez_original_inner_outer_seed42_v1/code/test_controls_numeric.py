"""Numerical regression for the matched 36-D supplementary controls."""
from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


def main() -> None:
    source_root, control_root, output = map(Path, sys.argv[1:4])
    sys.path.insert(0, str(source_root))
    sys.path.insert(0, str(control_root))
    import epilens.models as models
    from train_repaired_control import repaired_mean_std

    torch.manual_seed(9)
    values = torch.randn(4, 5, 8)
    mask = torch.ones(4, 5, dtype=torch.bool)
    original_mean, original_std = models._masked_mean_std(values, mask, 0)
    fixed_mean, fixed_std = repaired_mean_std(values, mask, 0)
    difference = max(
        float((original_mean - fixed_mean).abs().max()),
        float((original_std - fixed_std).abs().max()),
    )
    if difference > 1e-6:
        raise RuntimeError(f"Repair altered nondegenerate forward output: {difference}")
    models._masked_mean_std = repaired_mean_std
    results = {}
    for name, factory in (("prq", models.PRQNet), ("bcr", models.BCRNet)):
        for seizures in (1, 3):
            model = factory(dropout=0.0).train()
            x = torch.zeros(seizures, 2, 3, 36, requires_grad=True)
            valid = torch.ones(seizures, 2, 3, dtype=torch.bool)
            output_data = model(x, valid)
            scalar = output_data["logit_nez"].sum() if name == "prq" else output_data["logit_ez"].sum()
            scalar.backward()
            results[f"{name}_{seizures}_seizure"] = {
                "output_finite": bool(torch.isfinite(scalar)),
                "input_gradient_finite": bool(torch.isfinite(x.grad).all()),
                "parameter_gradients_finite": all(
                    p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()
                ),
            }
    report = {
        "pass": all(all(row.values()) for row in results.values()),
        "nondegenerate_forward_max_abs_difference": difference,
        "cases": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
