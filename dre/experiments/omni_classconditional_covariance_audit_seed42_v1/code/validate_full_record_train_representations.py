"""Hard-validate the regenerated full-record TRAIN R4/P16 cache.

This is deliberately a pre-analysis gate.  It reads the frozen
``omni_bag_mismatch_audit_seed42_v1`` full-record artifact and the newly
regenerated private representations, never the historical five-clip TRAIN
feature NPZs.  It produces only aggregate provenance/counts for publication.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score


CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"
OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"
TRAIN_AUROC = 0.9586782931
TOLERANCE = 1e-5
EXPECTED = {"edfs": 296, "total_segments": 316364, "labelled_segments": 145052,
            "labelled_edf_channel_units": 13350}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def output_key(edf: str) -> str:
    return hashlib.sha256(edf.encode("utf-8")).hexdigest()[:24]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--recovery-gate", type=Path, required=True)
    parser.add_argument("--representation-cache", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    recovery = json.loads(args.recovery_gate.read_text(encoding="utf-8"))
    if recovery.get("status") != "PASS" or recovery.get("historical_five_clip_train_npzs_read") is not False:
        raise RuntimeError("FULL_RECORD_ARTIFACT_RECOVERY_GATE_REQUIRED")
    if recovery.get("checkpoint_sha256") != CHECKPOINT_SHA256 or recovery.get("official_cnn_source_sha256") != OFFICIAL_SOURCE_SHA256:
        raise RuntimeError("FULL_RECORD_ARTIFACT_PROVENANCE_MISMATCH")
    recovery_sha = sha256(args.recovery_gate)
    artifacts = sorted(args.artifact.glob("*.npz"))
    outputs = sorted(args.representation_cache.glob("*.npz"))
    if len(artifacts) != EXPECTED["edfs"] or len(outputs) != EXPECTED["edfs"]:
        raise RuntimeError("FULL_RECORD_REPRESENTATION_EDF_COUNT_MISMATCH")
    total = labeled_segments = labeled_units = 0
    labels_all, scores_all = [], []
    maximum_probability_error = 0.0
    output_digests = hashlib.sha256()
    for artifact in artifacts:
        with np.load(artifact, allow_pickle=False) as source:
            edf = str(source["edf"])
            channels = source["channel_names"].astype(str)
            labels = np.asarray(source["pathological_labels"], dtype=np.int8)
            starts = np.asarray(source["starts"], dtype=np.int64)
            frozen_probs = np.asarray(source["segment_pathological_probs"], dtype=np.float32)
        destination = args.representation_cache / f"{output_key(edf)}.npz"
        marker = args.representation_cache / f"{output_key(edf)}.json"
        if not destination.is_file() or not marker.is_file():
            raise RuntimeError(f"Missing full-record representation for artifact EDF: {artifact.name}")
        metadata = json.loads(marker.read_text(encoding="utf-8"))
        binding = {"full_artifact_sha256": sha256(artifact), "recovery_gate_sha256": recovery_sha,
                   "checkpoint_sha256": CHECKPOINT_SHA256, "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256}
        if any(metadata.get(key) != value for key, value in binding.items()) or metadata.get("output_sha256") != sha256(destination):
            raise RuntimeError(f"Representation provenance/digest mismatch: {destination.name}")
        with np.load(destination, allow_pickle=False) as result:
            if (str(result["edf"]) != edf or not np.array_equal(result["channel_names"].astype(str), channels) or
                    not np.array_equal(np.asarray(result["pathological_labels"], np.int8), labels) or
                    not np.array_equal(np.asarray(result["starts"], np.int64), starts)):
                raise RuntimeError(f"Representation content identity mismatch: {destination.name}")
            offsets = np.asarray(result["segment_offsets"], dtype=np.int64)
            logits = np.asarray(result["segment_logits"], dtype=np.float64)
            r4, p16 = np.asarray(result["r4_mean"]), np.asarray(result["p16_mean"])
        if r4.shape != (len(channels), 32) or p16.shape != (len(channels), 16) or len(offsets) != len(channels) + 1:
            raise RuntimeError(f"Representation tensor shape mismatch: {destination.name}")
        lengths = np.diff(offsets)
        if not np.all(lengths == len(starts)):
            raise RuntimeError(f"Representation segment count mismatch: {destination.name}")
        reconstructed = np.vstack([1.0 - sigmoid(logits[offsets[i]:offsets[i + 1]]) for i in range(len(channels))])
        maximum_probability_error = max(maximum_probability_error, float(np.max(np.abs(reconstructed - frozen_probs))))
        if maximum_probability_error > 1e-6:
            raise RuntimeError("Frozen full-record segment-probability parity failed")
        labeled = np.isin(labels, [0, 1])
        total += int(len(channels) * len(starts)); labeled_segments += int(labeled.sum() * len(starts)); labeled_units += int(labeled.sum())
        labels_all.extend(labels[labeled].tolist())
        scores_all.extend(reconstructed[labeled].mean(axis=1).tolist())
        output_digests.update(destination.name.encode("utf-8")); output_digests.update(sha256(destination).encode("utf-8"))
    observed = {"edfs": len(artifacts), "total_segments": total, "labelled_segments": labeled_segments,
                "labelled_edf_channel_units": labeled_units,
                "train_full_auroc": float(roc_auc_score(np.asarray(labels_all), np.asarray(scores_all))),
                "max_frozen_probability_abs_error": maximum_probability_error}
    if any(observed[key] != value for key, value in EXPECTED.items()) or abs(observed["train_full_auroc"] - TRAIN_AUROC) >= TOLERANCE:
        raise RuntimeError("FULL_RECORD_TRAIN_REPRESENTATION_GATE_FAILED")
    payload = {"status": "PASS", "expected": {**EXPECTED, "train_full_auroc": TRAIN_AUROC, "tolerance": TOLERANCE},
               "observed": observed, "checkpoint_sha256": CHECKPOINT_SHA256,
               "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256, "full_record_artifact_recovery_gate_sha256": recovery_sha,
               "representation_cache_digest": output_digests.hexdigest(),
               "historical_five_clip_train_npzs_read": False}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "observed": observed}, sort_keys=True))


if __name__ == "__main__":
    main()
