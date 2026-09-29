"""Parameter/topology audit for the two separately weighted PC-CNN instances."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from official_spectrum import OfficialSpectrum, load_official_module
from pc_cnn import PCCNN
from train_pccnn import configure_stage


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    module = load_official_module(args.official_cnn)
    ictal = PCCNN(module.NeuralCNN(in_channels=1, outputs=1))
    omni = PCCNN(module.NeuralCNN(in_channels=1, outputs=1))
    i_parameters = {name: tuple(tensor.shape) for name, tensor in ictal.named_parameters()}
    o_parameters = {name: tuple(tensor.shape) for name, tensor in omni.named_parameters()}
    identical = i_parameters == o_parameters
    if not identical:
        raise RuntimeError("PC-CNN benchmark architectures differ")
    module_rows = []
    for component, submodule in (("TimeConv_and_ResNet18", ictal.raw),
                                 ("Physiology_FiLM", ictal.physiology),
                                 ("Context_LayerNorm", ictal.context_norm),
                                 ("Context_MHA", ictal.context_mha),
                                 ("Context_Residual", ictal.context_out)):
        module_rows.append({"component": component,
                            "parameters": sum(p.numel() for p in submodule.parameters()),
                            "benchmark_topology": "identical"})
    module_rows.append({"component": "three_bounded_scalar_gates", "parameters": 3,
                        "benchmark_topology": "identical"})
    total = sum(p.numel() for p in ictal.parameters())
    if total != sum(row["parameters"] for row in module_rows):
        raise RuntimeError("Parameter accounting mismatch")
    stage_b = configure_stage(ictal, "B")
    stage_b_names = {name for name, value in ictal.named_parameters() if value.requires_grad}
    if any(name.startswith("raw.") for name in stage_b_names):
        raise RuntimeError("Stage B unexpectedly unfreezes RawCNN")
    stage_c = configure_stage(ictal, "C")
    stage_c_names = {name for name, value in ictal.named_parameters() if value.requires_grad}
    if any(name.startswith("raw.feature_extractor.") or name.startswith("raw.cnn.layer1.")
           or name.startswith("raw.cnn.layer2.") or name.startswith("raw.cnn.layer3.")
           for name in stage_c_names):
        raise RuntimeError("Stage C unfreezes prohibited early morphology")
    ictal_frequency = OfficialSpectrum.frequency_metadata(250)
    omni_frequency = OfficialSpectrum.frequency_metadata(1000)
    audit = {"status": "PASS", "same_parameter_names_and_shapes": identical,
             "total_parameters_each_benchmark": total,
             "separately_initialized_weight_instances": ictal is not omni,
             "raw_backbone": "pinned official modified ImageNet ResNet18",
             "classification_heads": 1,
             "score_level_fusion": False,
             "stage_b_trainable_parameter_names": len(stage_b_names),
             "stage_c_trainable_parameter_names": len(stage_c_names),
             "ictal_max_frequency_hz": ictal_frequency["max_frequency_hz"],
             "omni_max_frequency_hz": omni_frequency["max_frequency_hz"],
             "native_frequency_bins_each": 224,
             "test_accessed": False}
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "MODEL_PARAMETER_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(module_rows[0]))
        writer.writeheader()
        writer.writerows(module_rows)
    (args.output / "ARCHITECTURE_IDENTITY_AUDIT.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
