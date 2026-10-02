"""Hard-gate the historical full-record TRAIN frozen-CNN artifact.

This program intentionally does not touch the historical five-clip TRAIN NPZ
cache.  It validates the separate full-record artifact created by
``omni_bag_mismatch_audit_seed42_v1`` before any representation extraction or
geometry calculation is permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"
OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"
EXPECTED = {
    "edfs": 296,
    "total_segments": 316364,
    "labeled_segments": 145052,
    "labeled_edf_channel_units": 13350,
    "train_full_auroc": 0.9586782931,
    "tolerance": 1e-5,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, suffix=".tmp") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def manifest_digest(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(sha256(path).encode("ascii"))
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True,
                        help="Full-record train_full_predictions directory only.")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    files = sorted(args.artifact.glob("*.npz"))
    total_segments = labeled_segments = labeled_units = 0
    y, score = [], []
    marker_failures = []
    for path in files:
        marker = path.with_suffix(".json")
        if not marker.is_file():
            marker_failures.append(f"missing_marker:{path.name}")
        else:
            payload = json.loads(marker.read_text(encoding="utf-8"))
            if payload.get("output_sha256") != sha256(path):
                marker_failures.append(f"output_sha_mismatch:{path.name}")
            if payload.get("checkpoint_sha256") != CHECKPOINT_SHA256:
                marker_failures.append(f"checkpoint_mismatch:{path.name}")
            if payload.get("official_cnn_source_sha256") != OFFICIAL_SOURCE_SHA256:
                marker_failures.append(f"source_mismatch:{path.name}")
        with np.load(path, allow_pickle=False) as z:
            channels = z["channel_names"].astype(str)
            labels = np.asarray(z["pathological_labels"], dtype=np.int8)
            starts = np.asarray(z["starts"], dtype=np.int64)
            probabilities = np.asarray(z["segment_pathological_probs"], dtype=np.float64)
            expected_starts = 1000 + np.arange(len(starts), dtype=np.int64) * 60000
            if probabilities.shape != (len(channels), len(starts)):
                raise RuntimeError(f"Invalid full-record score shape: {path.name}")
            if not np.array_equal(starts, expected_starts):
                raise RuntimeError(f"Noncanonical full-record starts: {path.name}")
            if not np.isfinite(probabilities).all() or (probabilities < 0).any() or (probabilities > 1).any():
                raise RuntimeError(f"Invalid frozen probabilities: {path.name}")
            valid = np.isin(labels, [0, 1])
            total_segments += int(probabilities.size)
            labeled_segments += int(valid.sum() * len(starts))
            labeled_units += int(valid.sum())
            y.extend(labels[valid].tolist())
            score.extend(probabilities[valid].mean(axis=1).tolist())

    auroc = float(roc_auc_score(np.asarray(y, dtype=np.int8), np.asarray(score, dtype=np.float64)))
    checks = {
        "edf_count": len(files) == EXPECTED["edfs"],
        "total_segments": total_segments == EXPECTED["total_segments"],
        "labeled_segments": labeled_segments == EXPECTED["labeled_segments"],
        "labeled_edf_channel_units": labeled_units == EXPECTED["labeled_edf_channel_units"],
        "frozen_cnn_auroc": abs(auroc - EXPECTED["train_full_auroc"]) < EXPECTED["tolerance"],
        "per_edf_provenance": not marker_failures,
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FULL_RECORD_TRAIN_RECOVERY_FAILED",
        "artifact": "omni_bag_mismatch_audit_seed42_v1/train_full_predictions",
        "artifact_manifest_sha256": manifest_digest(files),
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
        "expected": EXPECTED,
        "observed": {
            "edfs": len(files), "total_segments": total_segments,
            "labeled_segments": labeled_segments,
            "labeled_edf_channel_units": labeled_units,
            "train_full_auroc": auroc,
        },
        "checks": checks,
        "marker_failure_count": len(marker_failures),
        "marker_failure_examples": marker_failures[:10],
        "historical_five_clip_train_npzs_read": False,
    }
    atomic_json(args.out, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if result["status"] != "PASS":
        raise RuntimeError("FULL_RECORD_TRAIN_RECOVERY_FAILED")


if __name__ == "__main__":
    main()
