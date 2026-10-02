"""Build private R4/P16 means from the validated full-record TRAIN artifact.

The input population, labels, EDF identities, and deterministic 60-second
starts come exclusively from ``omni_bag_mismatch_audit_seed42_v1``'s frozen
full-record prediction artifact.  The old five-clip training feature NPZs are
not opened.  Native-digital HDF5 is replayed only to recover the frozen CNN
intermediate tensors at those already validated starts.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import tempfile
import time
from pathlib import Path

import h5py
import mne
import numpy as np
import torch


CHECKPOINT_SHA256 = "442b6a01b60994197fbc0e146f3abab7897b0b71e794259014d3eca663de3852"
OFFICIAL_SOURCE_SHA256 = "c493b43ec025f74153f0ada3041e6640bfa07beac58f058139ad5122da043f95"
DATASET_REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"
FULL_RECORD_BATCH_SIZE = 8
EXPECTED_EDFS = 296


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


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent,
                                     suffix=".tmp") as handle:
        np.savez_compressed(handle, **arrays)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def load_source(path: Path):
    if sha256(path) != OFFICIAL_SOURCE_SHA256:
        raise RuntimeError("Official CNN source SHA-256 mismatch")
    spec = importlib.util.spec_from_file_location("frozen_official_cnn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def model_and_preprocessor(source_path: Path, checkpoint: Path, device: torch.device):
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise RuntimeError("Frozen checkpoint SHA-256 mismatch")
    source = load_source(source_path)
    model = source.NeuralCNN(in_channels=1, outputs=1).to(device)
    model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=False)["model_state_dict"], strict=True)
    model.eval()
    prep = source.NeuralCNNPreProcessing(
        image_size=224, frequency=1000, freq_range_hz=[10, 300],
        event_length=60000, selected_window_size_ms=30000,
        selected_freq_range_hz=[10, 300], random_shift_ms=0)
    return model, prep


def physical_signal(h5: h5py.File, index: int, header: dict) -> np.ndarray:
    if str(header["dimension"]).lower() not in ("uv", "µv", "μv"):
        raise ValueError("Unexpected physical unit")
    digital = np.asarray(h5[f"digital/ch{index:04d}"][:], dtype=np.float64)
    dmin, dmax = float(header["digital_min"]), float(header["digital_max"])
    pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
    if dmax <= dmin:
        raise ValueError("Invalid digital range")
    signal = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
    source_rate = float(header["sample_frequency"])
    if source_rate < 900:
        raise ValueError("Official cohort source rate below 900 Hz")
    signal = mne.filter.notch_filter(signal, Fs=source_rate, freqs=[60], notch_widths=2,
                                     n_jobs=1, verbose=False)
    if source_rate != 1000:
        signal = mne.filter.resample(signal, up=1000, down=source_rate, npad="auto", n_jobs=1,
                                     verbose=False)
    return np.asarray(signal, dtype=np.float32)


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values >= 0, 1.0 / (1.0 + np.exp(-values)), np.exp(values) / (1.0 + np.exp(values)))


def output_key(edf: str) -> str:
    return hashlib.sha256(edf.encode("utf-8")).hexdigest()[:24]


def is_reusable(dest: Path, marker: Path, binding: dict) -> bool:
    if not dest.exists() and not marker.exists():
        return False
    if not dest.is_file() or not marker.is_file():
        raise RuntimeError(f"Partial representation cache: {dest.name}")
    old = json.loads(marker.read_text(encoding="utf-8"))
    if not all(old.get(key) == value for key, value in binding.items()):
        raise RuntimeError(f"Resume provenance mismatch: {dest.name}")
    if old.get("output_sha256") != sha256(dest):
        raise RuntimeError(f"Resume output digest mismatch: {dest.name}")
    return True


def forward_channel(signal: np.ndarray, starts: np.ndarray, model, prep, device: torch.device):
    r4_values, p16_values, logits = [], [], []
    captured: dict[str, torch.Tensor] = {}

    def capture(name: str):
        def hook(_module, _inputs, output):
            captured[name] = output
        return hook

    # Run the exact single ``model(image)`` call used to create the frozen
    # full-record artifact. Hooks only observe its intermediate tensors; they
    # do not introduce a second convolution/BN path with a different kernel
    # selection or numerical trace.
    r4_hook = model.cnn.register_forward_hook(capture("r4"))
    p16_hook = model.bn1.register_forward_hook(capture("p16"))
    try:
        with torch.inference_mode():
            for begin in range(0, len(starts), FULL_RECORD_BATCH_SIZE):
                current = starts[begin:begin + FULL_RECORD_BATCH_SIZE]
                waves = np.stack([signal[int(start):int(start) + 60000] for start in current])
                if waves.shape != (len(current), 60000):
                    raise RuntimeError("Short full-record window")
                captured.clear()
                image = prep(torch.from_numpy(waves).to(device))
                direct = model(image)
                r4, p16 = captured.get("r4"), captured.get("p16")
                if r4 is None or p16 is None or r4.shape != (len(current), 32) or p16.shape != (len(current), 16):
                    raise RuntimeError("Frozen intermediate hook shape mismatch")
                r4_values.append(r4.cpu().numpy().astype(np.float32))
                p16_values.append(p16.cpu().numpy().astype(np.float32))
                logits.append(direct.squeeze(1).cpu().numpy().astype(np.float32))
    finally:
        r4_hook.remove(); p16_hook.remove()
    return (np.concatenate(r4_values), np.concatenate(p16_values), np.concatenate(logits), 0.0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-artifact", type=Path, required=True)
    parser.add_argument("--recovery-gate", type=Path, required=True)
    parser.add_argument("--signal-cache", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard selection")
    gate = json.loads(args.recovery_gate.read_text(encoding="utf-8"))
    if gate.get("status") != "PASS" or gate.get("observed", {}).get("total_segments") != 316364:
        raise RuntimeError("FULL_RECORD_TRAIN_RECOVERY_GATE_REQUIRED")
    if gate.get("checkpoint_sha256") != CHECKPOINT_SHA256 or gate.get("official_cnn_source_sha256") != OFFICIAL_SOURCE_SHA256:
        raise RuntimeError("Recovery gate provenance mismatch")
    all_files = sorted(args.full_artifact.glob("*.npz"))
    if len(all_files) != EXPECTED_EDFS:
        raise RuntimeError(f"Expected {EXPECTED_EDFS} full-record EDF artifacts, got {len(all_files)}")
    files = all_files[args.shard_index::args.num_shards]
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for frozen full-record representation extraction")
    model, prep = model_and_preprocessor(args.official_cnn, args.checkpoint, device)
    extractor_sha = sha256(Path(__file__))
    gate_sha = sha256(args.recovery_gate)
    total_segments = labeled_segments = labeled_units = new_count = 0
    score_error = identity_error = 0.0
    started = time.monotonic()
    for ordinal, source in enumerate(files, 1):
        with np.load(source, allow_pickle=False) as z:
            patient, edf = str(z["patient"]), str(z["edf"])
            channels = z["channel_names"].astype(str)
            labels = np.asarray(z["pathological_labels"], dtype=np.int8)
            starts = np.asarray(z["starts"], dtype=np.int64)
            frozen_probs = np.asarray(z["segment_pathological_probs"], dtype=np.float32)
        if frozen_probs.shape != (len(channels), len(starts)):
            raise RuntimeError(f"Artifact shape mismatch: {source.name}")
        canonical = 1000 + np.arange(len(starts), dtype=np.int64) * 60000
        if not np.array_equal(starts, canonical):
            raise RuntimeError(f"Artifact has noncanonical starts: {source.name}")
        dest, marker = args.out / f"{output_key(edf)}.npz", args.out / f"{output_key(edf)}.json"
        binding = {
            "full_artifact_sha256": sha256(source), "recovery_gate_sha256": gate_sha,
            "checkpoint_sha256": CHECKPOINT_SHA256, "official_cnn_source_sha256": OFFICIAL_SOURCE_SHA256,
            "extractor_sha256": extractor_sha, "batch_size": FULL_RECORD_BATCH_SIZE,
        }
        if is_reusable(dest, marker, binding):
            print(json.dumps({"ordinal": ordinal, "of": len(files), "reused": True}), flush=True)
            continue
        h5_path = args.signal_cache / Path(edf).with_suffix(".edf.h5")
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        r4_mean, p16_mean, flat_logits, offsets, clips = [], [], [], [0], []
        local_probability_error = local_identity_error = 0.0
        with h5py.File(h5_path, "r") as h5:
            if str(h5.attrs["dataset_revision"]) != DATASET_REVISION:
                raise RuntimeError("Dataset revision mismatch")
            metadata = json.loads(h5["metadata_json"][()].decode("utf-8"))
            headers = metadata["signal_headers"]
            index = {str(header["label"]): i for i, header in enumerate(headers)}
            absent = [name for name in channels if name not in index]
            if absent:
                raise RuntimeError(f"Selected channels absent from native cache: {len(absent)}")
            for channel_index, channel in enumerate(channels):
                signal = physical_signal(h5, index[channel], headers[index[channel]])
                if int(starts[-1]) + 60000 > len(signal):
                    raise RuntimeError("Full-record artifact exceeds reconstructed signal")
                r4, p16, logits, error = forward_channel(signal, starts, model, prep, device)
                reconstructed_probs = (1.0 - sigmoid(logits)).astype(np.float32)
                local_probability_error = max(local_probability_error, float(np.max(np.abs(reconstructed_probs - frozen_probs[channel_index]))))
                local_identity_error = max(local_identity_error, error)
                r4_mean.append(np.asarray(r4, np.float64).mean(axis=0))
                p16_mean.append(np.asarray(p16, np.float64).mean(axis=0))
                flat_logits.extend(logits.tolist()); offsets.append(len(flat_logits)); clips.append(len(logits))
        if local_probability_error > 1e-6:
            raise RuntimeError(f"Full-record frozen-score parity failed: {local_probability_error}")
        valid = np.isin(labels, [0, 1])
        total_segments += int(len(channels) * len(starts))
        labeled_segments += int(valid.sum() * len(starts)); labeled_units += int(valid.sum())
        score_error = max(score_error, local_probability_error); identity_error = max(identity_error, local_identity_error)
        atomic_npz(dest, patient=np.asarray(patient), edf=np.asarray(edf), channel_names=channels,
                   pathological_labels=labels, r4_mean=np.asarray(r4_mean, dtype=np.float32),
                   p16_mean=np.asarray(p16_mean, dtype=np.float32), segment_logits=np.asarray(flat_logits, dtype=np.float32),
                   segment_offsets=np.asarray(offsets, dtype=np.int64), clips=np.asarray(clips, dtype=np.int32), starts=starts)
        atomic_json(marker, {**binding, "output_sha256": sha256(dest), "channels": len(channels),
                             "segments": int(len(channels) * len(starts)), "r4_dimension": 32, "p16_dimension": 16,
                             "max_frozen_probability_abs_error": local_probability_error,
                             "max_forward_identity_abs_error": local_identity_error})
        new_count += 1
        print(json.dumps({"ordinal": ordinal, "of": len(files), "reused": False, "channels": len(channels),
                          "segments": int(len(channels) * len(starts)), "probability_error": local_probability_error}), flush=True)
    complete_files = len(list(args.out.glob("*.npz")))
    atomic_json(args.out / f"SHARD_{args.shard_index}_OF_{args.num_shards}.json", {
        "status": "COMPLETE", "shard_index": args.shard_index, "num_shards": args.num_shards,
        "source_edfs_in_shard": len(files), "new_edfs": new_count,
    })
    atomic_json(args.out / "FULL_RECORD_REPRESENTATION_STATE.json", {
        "status": "COMPLETE" if complete_files == EXPECTED_EDFS else "PARTIAL", "completed_edfs": complete_files,
        "expected_edfs": EXPECTED_EDFS, "shard_index": args.shard_index, "num_shards": args.num_shards,
        "new_edfs": new_count, "segments_this_shard": total_segments,
        "labeled_segments_this_shard": labeled_segments, "labeled_units_this_shard": labeled_units,
        "max_frozen_probability_abs_error_this_shard": score_error,
        "max_forward_identity_abs_error_this_shard": identity_error,
        "wall_seconds": time.monotonic() - started, "full_artifact_recovery_gate_sha256": gate_sha,
        "historical_five_clip_train_npzs_read": False,
    })


if __name__ == "__main__":
    main()
