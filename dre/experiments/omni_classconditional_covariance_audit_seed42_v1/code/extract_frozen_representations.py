"""Extract frozen CNN R4/P16 channel means from existing feature NPZs.

This is deliberately inference-only.  It reads the already extracted 60-s
feature NPZs, not EDF files; it never writes a checkpoint and it never
optimizes a parameter.  Output caches are private because they contain EDF /
channel identities and labels.  Public artifacts are produced by run_audit.py.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch


CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"
OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", delete=False,
                                     dir=path.parent, suffix=".tmp") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent,
                                     suffix=".tmp") as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_official(path: Path):
    if sha256(path) != OFFICIAL_SOURCE_SHA256:
        raise RuntimeError("Official CNN source SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("frozen_official_cnn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def load_model(source: Path, checkpoint: Path, device: torch.device):
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen CNN checkpoint SHA-256 mismatch")
    module = load_official(source)
    raw = module.NeuralCNN(in_channels=1, outputs=1).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    raw.load_state_dict(payload["model_state_dict"])
    raw.eval()
    preprocessing = module.NeuralCNNPreProcessing(
        image_size=224, frequency=1000, freq_range_hz=[10, 300],
        event_length=60000, selected_window_size_ms=30000,
        selected_freq_range_hz=[10, 300], random_shift_ms=0)
    return raw, preprocessing


def forward_representations(raw, preprocessing, waves: np.ndarray, device, batch_size: int):
    r4_out, p16_out, logits = [], [], []
    max_identity_error = 0.0
    with torch.inference_mode():
        for start in range(0, len(waves), batch_size):
            wave = torch.from_numpy(np.asarray(waves[start:start + batch_size], np.float32)).to(device)
            image = preprocessing(wave)
            r4 = raw.cnn(raw.feature_extractor(image))
            p32 = raw.bn(raw.relu(raw.fc(r4)))
            p16 = raw.bn1(raw.relu1(raw.fc1(p32)))
            derived = raw.fc_out(p16)
            direct = raw(image)
            max_identity_error = max(max_identity_error, float((derived - direct).abs().max().cpu()))
            r4_out.append(r4.cpu().numpy().astype(np.float32))
            p16_out.append(p16.cpu().numpy().astype(np.float32))
            logits.append(derived.squeeze(1).cpu().numpy().astype(np.float32))
    if max_identity_error >= 1e-7:
        raise RuntimeError(f"Frozen forward identity failed: {max_identity_error}")
    return (np.concatenate(r4_out), np.concatenate(p16_out), np.concatenate(logits),
            max_identity_error)


def read_source(mode: str, path: Path):
    with np.load(path, allow_pickle=False) as source:
        if mode == "train":
            waves = np.asarray(source["waveforms"], dtype=np.float32)
            if waves.ndim != 3 or waves.shape[-1] != 60000:
                raise RuntimeError("Invalid synchronized TRAIN waveform feature cache")
            clips, channels = waves.shape[:2]
            names = np.tile(np.asarray(source["channel_names"]).astype(str), clips)
            labels = np.tile(np.asarray(source["labels"], dtype=np.int8), clips)
            return (waves.reshape(clips * channels, 60000), names, labels,
                    str(source["patient"]), str(source["edf"]))
        waves = np.asarray(source["data"], dtype=np.float32)
        names = np.asarray(source["name"]).astype(str)
        normal = np.asarray(source["labels"], dtype=np.int8)
        labels = np.where(normal >= 0, 1 - normal, -1).astype(np.int8)
        return waves, names, labels, str(source["patient"]), str(source["edf_name"])


def aggregate(names, labels, r4, p16, logits):
    names = np.asarray(names).astype(str)
    labels = np.asarray(labels, dtype=np.int8)
    ordered = list(dict.fromkeys(names.tolist()))
    channel_r4, channel_p16, channel_labels, offsets, flat_logits, counts = [], [], [], [0], [], []
    for name in ordered:
        index = np.flatnonzero(names == name)
        unique = np.unique(labels[index])
        if len(unique) != 1:
            raise RuntimeError("Conflicting labels within EDF-channel")
        channel_r4.append(np.asarray(r4[index], dtype=np.float64).mean(axis=0))
        channel_p16.append(np.asarray(p16[index], dtype=np.float64).mean(axis=0))
        current = np.asarray(logits[index], dtype=np.float32)
        flat_logits.extend(current.tolist())
        offsets.append(len(flat_logits))
        counts.append(len(index))
        channel_labels.append(int(unique[0]))
    return {
        "channel_names": np.asarray(ordered),
        "pathological_labels": np.asarray(channel_labels, dtype=np.int8),
        "r4_mean": np.asarray(channel_r4, dtype=np.float32),
        "p16_mean": np.asarray(channel_p16, dtype=np.float32),
        "segment_logits": np.asarray(flat_logits, dtype=np.float32),
        "segment_offsets": np.asarray(offsets, dtype=np.int64),
        "clips": np.asarray(counts, dtype=np.int32),
    }


def output_name(source: Path, root: Path) -> str:
    relative = source.relative_to(root).as_posix()
    return hashlib.sha256(relative.encode("utf-8")).hexdigest()[:24]


def is_reusable(dest: Path, marker: Path, binding: dict) -> bool:
    if not dest.exists() and not marker.exists():
        return False
    if not dest.is_file() or not marker.is_file():
        raise RuntimeError(f"Partial private representation cache: {dest.name}")
    recorded = json.loads(marker.read_text(encoding="utf-8"))
    return all(recorded.get(key) == value for key, value in binding.items()) and recorded.get("output_sha256") == sha256(dest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid shard selection")
    all_files = sorted(args.source.rglob("*.npz")) if args.mode == "test" else sorted(args.source.glob("*.npz"))
    expected = 237 if args.mode == "test" else 296
    if len(all_files) != expected:
        raise RuntimeError(f"Expected {expected} {args.mode} source feature NPZs, got {len(all_files)}")
    files = all_files[args.shard_index::args.shards]
    args.out.mkdir(parents=True, exist_ok=True)
    extractor_hash = sha256(Path(__file__))
    device = torch.device("cuda")
    raw, preprocessing = load_model(args.official_cnn, args.checkpoint, device)
    for ordinal, source in enumerate(files, 1):
        dest = args.out / f"{output_name(source, args.source)}.npz"
        marker = dest.with_suffix(".json")
        binding = {
            "mode": args.mode, "relative_source": source.relative_to(args.source).as_posix(),
            "source_bytes": source.stat().st_size, "checkpoint_sha256": CHECKPOINT_SHA256,
            "official_source_sha256": OFFICIAL_SOURCE_SHA256, "extractor_sha256": extractor_hash,
        }
        if is_reusable(dest, marker, binding):
            print(json.dumps({"mode": args.mode, "ordinal": ordinal, "of": len(files), "reused": True}), flush=True)
            continue
        waves, names, labels, patient, edf = read_source(args.mode, source)
        r4, p16, logits, error = forward_representations(raw, preprocessing, waves, device, args.batch_size)
        values = aggregate(names, labels, r4, p16, logits)
        atomic_npz(dest, patient=np.asarray(patient), edf=np.asarray(edf), **values)
        atomic_json(marker, {**binding, "output_sha256": sha256(dest),
                             "channels": len(values["channel_names"]), "segments": len(waves),
                             "r4_dimension": 32, "p16_dimension": 16,
                             "frozen_forward_max_abs_error": error})
        print(json.dumps({"mode": args.mode, "ordinal": ordinal, "of": len(files), "reused": False,
                          "channels": len(values["channel_names"]), "segments": len(waves),
                          "identity_error": error}), flush=True)


if __name__ == "__main__":
    main()
