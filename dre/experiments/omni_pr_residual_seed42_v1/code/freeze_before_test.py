"""Hash-lock residual selections and validation thresholds before TEST I/O."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from common import atomic_json, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--baseline-audit", type=Path, required=True)
    parser.add_argument("--train-audit", type=Path, required=True)
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("Pre-test freeze already exists; refusing overwrite")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    baseline = json.loads(args.baseline_audit.read_text(encoding="utf-8"))
    train = json.loads(args.train_audit.read_text(encoding="utf-8"))
    selection = json.loads(args.selection.read_text(encoding="utf-8"))
    if baseline.get("status") != "PASS" or train.get("status") != "COMPLETE" or \
            selection.get("status") != "TRAIN_VALIDATION_COMPLETE" or \
            selection.get("official_test_accessed_by_trainer"):
        raise RuntimeError("Development artifacts are not eligible for freezing")
    if selection["protocol_sha256"] != sha256(args.protocol):
        raise RuntimeError("Selection/protocol mismatch")
    checkpoints = {}
    for name in ("ABS-ONLY", "PR-CNN"):
        selected = selection["selections"][name]
        path = Path(selected["checkpoint"])
        if sha256(path) != selected["checkpoint_sha256"]:
            raise RuntimeError("Selected residual checkpoint changed")
        checkpoints[name] = {"private_path": str(path), "sha256": selected["checkpoint_sha256"],
                             "selected_epoch": selected["selected_epoch"],
                             "validation_auroc": selected["validation"]["auroc"],
                             "validation_ap": selected["validation"]["pooled_ap"],
                             "validation_threshold": selection["thresholds"][name]}
    value = {
        "status": "FROZEN_BEFORE_OFFICIAL_TEST",
        "model_frozen_before_final_test": True,
        "protocol_sha256": sha256(args.protocol),
        "baseline_audit_sha256": sha256(args.baseline_audit),
        "train_embedding_audit_sha256": sha256(args.train_audit),
        "selection_sha256": sha256(args.selection),
        "frozen_cnn_checkpoint_sha256": protocol["frozen_checkpoint_sha256"],
        "checkpoints": checkpoints,
        "thresholds": selection["thresholds"],
        "official_test_accessed_by_head_training_or_selection": False,
    }
    atomic_json(args.output, value)
    print(json.dumps({key: value[key] for key in ("status", "model_frozen_before_final_test", "protocol_sha256")}, sort_keys=True))


if __name__ == "__main__":
    main()
