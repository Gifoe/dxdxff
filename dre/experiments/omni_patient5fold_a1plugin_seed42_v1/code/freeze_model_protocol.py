#!/usr/bin/env python3
"""Bind the already-frozen cohort to the model/optimization protocol."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from modeling import OFFICIAL_CNN_SHA256, sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-protocol", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cohort = json.loads(args.cohort_protocol.read_text(encoding="utf-8"))
    manifest_sha = sha256(args.private_manifest)
    if cohort["patient_manifest_private_sha256"] != manifest_sha:
        raise RuntimeError("Cohort private manifest mismatch")
    if sha256(args.official_cnn) != OFFICIAL_CNN_SHA256:
        raise RuntimeError("Official TimeConv source mismatch")
    lock = {
        "experiment": "omni_patient5fold_a1plugin_seed42_v1",
        "cohort_protocol_sha256": sha256(args.cohort_protocol),
        "patient_manifest_private_sha256": manifest_sha,
        "dataset_revision": "73b9c5180a57828ab2a83c040e7e9d112e77b2cc",
        "official_timeconv_cnn_sha256": OFFICIAL_CNN_SHA256,
        "external_pretraining": "torchvision ImageNet ResNet18 only; no Omni-labelled checkpoint",
        "seed": 42, "folds": 5, "epochs": 30,
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-3, "gradient_clip": 1.0},
        "record_preprocessing": {"fixed_window_seconds": 60, "fixed_window_start_seconds": 1,
                                 "sampling_rate_hz": 1000, "notch_hz": 60},
        "temporal_schedule": {"segments_per_record": 30, "segment_seconds": 2,
                               "segment_selection": "epoch k uses non-overlapping segment k-1; all 60 seconds once"},
        "channel_batch": 16,
        "aggregation": {"within_edf_segments": "mean_logit", "across_edf_sessions": "mean_logit"},
        "baseline": "official TimeConv morphology plus original classifier; trained only on four fit folds",
        "plugin": {"only_difference": "A1 two-head patient-relative-z attention residual",
                   "attention_heads": 2, "attention_dropout": 0.25,
                   "output_projection_initialization": "zero", "effective_gate_initial": 0.02,
                   "residual_bound": "0.5*tanh(gate_parameter)"},
        "threshold_diagnostic": 0.5,
        "selection": "none; no validation split, early stopping, threshold tuning, or hyperparameter search",
        "test_fold_labels_used_for_optimization": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(lock, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, args.output)


if __name__ == "__main__":
    main()
