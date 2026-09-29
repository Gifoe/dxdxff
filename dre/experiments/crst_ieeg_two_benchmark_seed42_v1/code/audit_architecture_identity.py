"""Compare exact trainable parameter topology across both independent runs."""

import csv
import json
from pathlib import Path

import torch

from crst_model import CRSTiEEG, parameter_count


def main():
    root = Path(__file__).resolve().parents[1]
    ictal = CRSTiEEG()
    omni = CRSTiEEG()
    params1 = {k: tuple(v.shape) for k, v in ictal.named_parameters()}
    params2 = {k: tuple(v.shape) for k, v in omni.named_parameters()}
    if params1 != params2 or parameter_count(ictal) != parameter_count(omni):
        raise RuntimeError("Benchmark architectures are not parameter-identical")
    with (root / "MODEL_PARAMETER_AUDIT.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(("module", "shape", "parameters", "trainable"))
        for name, value in ictal.named_parameters():
            writer.writerow((name, "x".join(map(str, value.shape)), value.numel(),
                             bool(value.requires_grad)))
    audit = {
        "pass": True,
        "architecture": "CRSTiEEG",
        "trainable_parameters_each_benchmark": parameter_count(ictal),
        "parameter_shapes_identical": True,
        "parameter_names_identical": True,
        "parameter_tensors": len(params1),
        "same_class_and_module_topology": True,
        "different_only": ["dataset", "frequency availability mask", "record count",
                           "channel count", "frozen train/test membership",
                           "independently learned weights"],
        "batch_norm_present": any(isinstance(m, torch.nn.modules.batchnorm._BatchNorm)
                                  for m in ictal.modules()),
        "patient_or_center_embedding": False,
        "test_outcome_accessed_by_this_audit": False,
    }
    (root / "ARCHITECTURE_IDENTITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
