"""Resume-safe frozen-CNN embedding extraction for official TRAIN or TEST.

Private outputs contain channel identities and never belong in Git. The
public audit contains counts, hashes, identity error, and aggregate metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from common import atomic_json, atomic_npz, sha256, sigmoid
from official_embedding import (CHECKPOINT_SHA256, OFFICIAL_SOURCE_SHA256,
                                EmbeddingAdapter, load_official)


def load_model(source_path: Path, checkpoint_path: Path, device):
    if sha256(checkpoint_path) != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen CNN checkpoint SHA-256 mismatch")
    source = load_official(source_path)
    raw = source.NeuralCNN(in_channels=1, outputs=1).to(device)
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    raw.load_state_dict(payload["model_state_dict"])
    raw.eval()
    adapter = EmbeddingAdapter(raw).to(device).eval()
    preprocessor = source.NeuralCNNPreProcessing(
        image_size=224, frequency=1000, freq_range_hz=[10, 300],
        event_length=60000, selected_window_size_ms=30000,
        selected_freq_range_hz=[10, 300], random_shift_ms=0)
    if sum(p.numel() for p in raw.parameters()) != sum(p.numel() for p in adapter.parameters()):
        raise RuntimeError("Embedding adapter changed the CNN parameter count")
    return raw, adapter, preprocessor


def infer(data, raw, adapter, prep, device, batch_size, identity):
    logits, embeddings = [], []
    with torch.inference_mode():
        for start in range(0, len(data), batch_size):
            wave = torch.from_numpy(np.asarray(data[start:start + batch_size], np.float32)).to(device)
            image = prep(wave)
            result = adapter(image)
            if identity["real_batches_checked"] == 0:
                old = raw(image)
                error = float((old - result["logit"]).abs().max().cpu())
                identity["real_max_abs_error"] = error
                identity["real_batches_checked"] = 1
                if error >= 1e-7:
                    raise RuntimeError(f"Embedding-forward identity failed: {error}")
            logits.append(result["logit"].squeeze(1).detach().cpu().numpy().astype(np.float32))
            embeddings.append(result["embedding32"].detach().cpu().numpy().astype(np.float32))
    return np.concatenate(logits), np.concatenate(embeddings)


def aggregate(names, pathological_labels, logits, embeddings):
    ordered = list(dict.fromkeys(map(str, names)))
    channel_logits, channel_embeddings, labels, clips = [], [], [], []
    offsets, flat_logits = [0], []
    names = np.asarray(names).astype(str)
    pathological_labels = np.asarray(pathological_labels, dtype=np.int8)
    for name in ordered:
        selected = np.flatnonzero(names == name)
        current_labels = np.unique(pathological_labels[selected])
        if len(current_labels) != 1:
            raise RuntimeError("Conflicting labels within EDF-channel")
        current_logits = np.asarray(logits[selected], dtype=np.float32)
        channel_logits.append(float(current_logits.mean()))
        channel_embeddings.append(np.asarray(embeddings[selected], dtype=np.float64).mean(axis=0))
        labels.append(int(current_labels[0]))
        clips.append(len(selected))
        flat_logits.extend(current_logits.tolist())
        offsets.append(len(flat_logits))
    return {
        "channel_names": np.asarray(ordered),
        "pathological_labels": np.asarray(labels, dtype=np.int8),
        "embeddings": np.asarray(channel_embeddings, dtype=np.float32),
        "mean_logits": np.asarray(channel_logits, dtype=np.float32),
        "segment_logits": np.asarray(flat_logits, dtype=np.float32),
        "segment_offsets": np.asarray(offsets, dtype=np.int64),
        "clips": np.asarray(clips, dtype=np.int32),
    }


def read_train(path: Path):
    with np.load(path, allow_pickle=False) as source:
        waves = np.asarray(source["waveforms"], dtype=np.float32)
        if waves.ndim != 3 or waves.shape[-1] != 60000:
            raise RuntimeError("Invalid synchronized TRAIN waveform cache")
        clips, channels = waves.shape[:2]
        names = np.tile(np.asarray(source["channel_names"]).astype(str), clips)
        pathological = np.tile(np.asarray(source["labels"], dtype=np.int8), clips)
        data = waves.reshape(clips * channels, 60000)
        return data, names, pathological, str(source["patient"]), str(source["edf"])


def read_test(path: Path):
    with np.load(path, allow_pickle=False) as source:
        data = np.asarray(source["data"], dtype=np.float32)
        names = np.asarray(source["name"]).astype(str)
        normal = np.asarray(source["labels"], dtype=np.int8)
        pathological = np.where(normal >= 0, 1 - normal, -1).astype(np.int8)
        return data, names, pathological, str(source["patient"]), str(source["edf_name"])


def output_name(path: Path, root: Path) -> str:
    import hashlib
    relative = path.relative_to(root).as_posix()
    return hashlib.sha256(relative.encode()).hexdigest()[:24]


def validate_resume(dest, marker, binding):
    if not dest.exists() and not marker.exists():
        return False
    if not dest.is_file() or not marker.is_file():
        raise RuntimeError(f"Partial embedding cache: {dest.name}")
    old = json.loads(marker.read_text(encoding="utf-8"))
    if any(old.get(key) != value for key, value in binding.items()) or old.get("output_sha256") != sha256(dest):
        raise RuntimeError(f"Embedding cache provenance mismatch: {dest.name}")
    return True


def load_output(path: Path):
    with np.load(path, allow_pickle=False) as source:
        offsets = np.asarray(source["segment_offsets"], dtype=np.int64)
        flat = np.asarray(source["segment_logits"], dtype=np.float64)
        logits = [flat[offsets[i]:offsets[i + 1]] for i in range(len(offsets) - 1)]
        return {key: np.asarray(source[key]) for key in
                ("channel_names", "pathological_labels", "embeddings", "mean_logits", "clips")}, logits, str(source["patient"]), str(source["edf"])


def finalize(args, lock, identity, elapsed):
    y, official_score, mean_logit_score = [], [], []
    patients, labeled_patients, edfs, channels, clips = set(), set(), set(), 0, 0
    fallback_files = 0
    for path in sorted(args.output.glob("*.npz")):
        data, logits, patient, edf = load_output(path)
        patients.add(patient); edfs.add(edf); channels += len(data["channel_names"])
        clips += int(data["clips"].sum())
        labels = data["pathological_labels"].astype(np.int8)
        for label, values, mean_value in zip(labels, logits, data["mean_logits"]):
            if label in (0, 1):
                labeled_patients.add(patient)
                y.append(int(label))
                official_score.append(float(1.0 - sigmoid(values).mean()))
                mean_logit_score.append(float(1.0 - sigmoid(mean_value)))
    audit = {
        "status": "COMPLETE", "mode": args.mode,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
        "protocol_sha256": sha256(args.protocol),
        "source_files": len(list(args.source.rglob("*.npz"))) if args.mode == "test" else len(list(args.source.glob("*.npz"))),
        "output_files": len(list(args.output.glob("*.npz"))),
            "patients": len(patients), "labeled_patients": len(labeled_patients),
            "edfs": len(edfs), "all_channels": channels,
        "segments": clips, "labeled_edf_channel_pairs": len(y),
        "identity_real_max_abs_error": identity["real_max_abs_error"],
        "cnn_parameter_count_unchanged": True,
        "elapsed_seconds": elapsed,
        "individual_records_private": True,
    }
    if args.mode == "test":
        y = np.asarray(y, dtype=np.int8)
        official_auc = float(roc_auc_score(y, official_score))
        mean_logit_auc = float(roc_auc_score(y, mean_logit_score))
        audit.update({
            "normal_pairs": int((y == 0).sum()), "pathological_pairs": int((y == 1).sum()),
            "official_sigmoid_then_mean_auroc": official_auc,
            "mean_logit_then_sigmoid_diagnostic_auroc": mean_logit_auc,
            "expected_auroc": lock["expected_baseline_auroc"],
            "absolute_error": abs(official_auc - lock["expected_baseline_auroc"]),
            "baseline_replay_pass": abs(official_auc - lock["expected_baseline_auroc"]) < lock["baseline_absolute_tolerance"],
        })
        if len(y) != lock["expected_test_pairs"] or len(labeled_patients) != lock["expected_test_patients"] or \
                int((y == 0).sum()) != lock["expected_test_normal"] or int((y == 1).sum()) != lock["expected_test_pathological"]:
            raise RuntimeError("Official TEST cohort counts differ")
        if not audit["baseline_replay_pass"]:
            raise RuntimeError("BASELINE_REPLAY_FAILED")
        atomic_json(args.public_audit / "BASELINE_REPLAY_AUDIT.json", audit)
    else:
        if len(patients) != 141:
            raise RuntimeError(f"Expected 141 frozen TRAIN patients, got {len(patients)}")
        atomic_json(args.public_audit / "TRAIN_EMBEDDING_CACHE_AUDIT.json", audit)
    atomic_json(args.public_audit / "EMBEDDING_EXTRACTION_AUDIT.json", audit)
    print(json.dumps(audit, sort_keys=True), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-audit", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha256(args.official_cnn) != lock["official_cnn_source_sha256"] or sha256(args.checkpoint) != lock["frozen_checkpoint_sha256"]:
        raise RuntimeError("Frozen source/checkpoint differs from protocol")
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid shard")
    files = sorted(args.source.rglob("*.npz")) if args.mode == "test" else sorted(args.source.glob("*.npz"))
    if args.mode == "test" and len(files) != 237:
        raise RuntimeError(f"Expected 237 official TEST NPZs, got {len(files)}")
    if args.mode == "train" and len(files) != 296:
        raise RuntimeError(f"Expected 296 synchronized TRAIN NPZs, got {len(files)}")
    files = files[args.shard_index::args.shards]
    args.output.mkdir(parents=True, exist_ok=True)
    args.public_audit.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    raw, adapter, prep = load_model(args.official_cnn, args.checkpoint, device)
    identity = {"real_batches_checked": 0, "real_max_abs_error": None}
    started = time.monotonic()
    for ordinal, path in enumerate(files, 1):
        key = output_name(path, args.source)
        dest, marker = args.output / f"{key}.npz", args.output / f"{key}.json"
        binding = {"mode": args.mode, "relative_source": path.relative_to(args.source).as_posix(),
                   "source_bytes": path.stat().st_size, "checkpoint_sha256": CHECKPOINT_SHA256,
                   "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
                   "protocol_sha256": sha256(args.protocol)}
        if validate_resume(dest, marker, binding):
            print(json.dumps({"ordinal": ordinal, "of": len(files), "reused": True}), flush=True)
            continue
        data, names, labels, patient, edf = read_train(path) if args.mode == "train" else read_test(path)
        logits, embeddings = infer(data, raw, adapter, prep, device, args.batch_size, identity)
        aggregated = aggregate(names, labels, logits, embeddings)
        atomic_npz(dest, patient=patient, edf=edf, **aggregated)
        atomic_json(marker, {**binding, "output_sha256": sha256(dest),
                             "channels": len(aggregated["channel_names"]), "segments": len(data)})
        print(json.dumps({"ordinal": ordinal, "of": len(files), "reused": False,
                          "channels": len(aggregated["channel_names"]), "segments": len(data)}), flush=True)
    # A shard cannot finalize global counts until every shard is present.
    expected = 237 if args.mode == "test" else 296
    if len(list(args.output.glob("*.npz"))) == expected:
        if identity["real_batches_checked"] == 0:
            # All files were resumed. Identity was already enforced when each
            # cache was first generated; public audit must already exist.
            prior = args.public_audit / ("BASELINE_REPLAY_AUDIT.json" if args.mode == "test" else "TRAIN_EMBEDDING_CACHE_AUDIT.json")
            if not prior.is_file():
                raise RuntimeError("All cache files resumed but identity audit is absent")
        else:
            finalize(args, lock, identity, time.monotonic() - started)


if __name__ == "__main__":
    main()
