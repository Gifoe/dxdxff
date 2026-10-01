"""Resume-safe full-record frozen-CNN inference for official TRAIN EDFs.

The historical synchronized TRAIN cache is used only as the authoritative
EDF/channel/label population. Signals come from the lossless native-digital
HDF5 migration and are reconstructed with the same physical scaling, notch
filter, 1000-Hz rate, TEST windows, wavelet preprocessing, and frozen CNN.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import time
from pathlib import Path

import h5py
import mne
import numpy as np
import torch

from audit_core import atomic_json, atomic_npz, sha256


def load_source(path: Path):
    spec = importlib.util.spec_from_file_location("frozen_official_cnn", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def model_and_preprocessor(source_path: Path, checkpoint: Path, device):
    source = load_source(source_path)
    model = source.NeuralCNN(in_channels=1, outputs=1).to(device)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.eval()
    prep = source.NeuralCNNPreProcessing(
        image_size=224, frequency=1000, freq_range_hz=[10, 300],
        event_length=60000, selected_window_size_ms=30000,
        selected_freq_range_hz=[10, 300], random_shift_ms=0)
    return model, prep


def physical_signal(h5: h5py.File, index: int, header: dict) -> np.ndarray:
    if str(header["dimension"]).lower() not in ("uv", "µv", "μv"):
        raise ValueError(f"Unexpected physical unit: {header['dimension']}")
    digital = np.asarray(h5[f"digital/ch{index:04d}"][:], dtype=np.float64)
    dmin, dmax = float(header["digital_min"]), float(header["digital_max"])
    pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
    if dmax <= dmin:
        raise ValueError("Invalid digital range")
    signal = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
    source_rate = float(header["sample_frequency"])
    if source_rate < 900:
        raise ValueError(f"Official cohort source rate below 900 Hz: {source_rate}")
    signal = mne.filter.notch_filter(signal, Fs=source_rate, freqs=[60], notch_widths=2,
                                     n_jobs=1, verbose=False)
    if source_rate != 1000:
        signal = mne.filter.resample(signal, up=1000, down=source_rate,
                                     npad="auto", n_jobs=1, verbose=False)
    return np.asarray(signal, dtype=np.float32)


def starts_for(length: int) -> np.ndarray:
    count = (int(length) - 2000) // 60000
    if count < 1:
        raise ValueError(f"Signal too short after TEST margins: {length}")
    return 1000 + np.arange(count, dtype=np.int64) * 60000


def infer_channel(signal, starts, model, prep, device, batch_size):
    logits = []
    with torch.inference_mode():
        for begin in range(0, len(starts), batch_size):
            selected = starts[begin:begin + batch_size]
            waves = np.stack([signal[int(s):int(s) + 60000] for s in selected])
            tensor = torch.from_numpy(waves).to(device)
            normal_logits = model(prep(tensor)).squeeze(1)
            logits.append(normal_logits.detach().cpu().numpy().astype(np.float32))
    normal = np.concatenate(logits)
    return (1.0 - 1.0 / (1.0 + np.exp(-normal.astype(np.float64)))).astype(np.float32)


def output_key(edf: str) -> str:
    import hashlib
    return hashlib.sha256(edf.encode("utf-8")).hexdigest()[:24]


def validate_resume(dest: Path, marker: Path, binding: dict) -> bool:
    if not dest.exists() and not marker.exists():
        return False
    if not dest.is_file() or not marker.is_file():
        raise RuntimeError(f"Partial output: {dest.name}")
    old = json.loads(marker.read_text(encoding="utf-8"))
    if any(old.get(k) != v for k, v in binding.items()) or old.get("output_sha256") != sha256(dest):
        raise RuntimeError(f"Resume provenance mismatch: {dest.name}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-waveforms", type=Path, required=True)
    parser.add_argument("--signal-cache", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--max-new-edfs", type=int, default=0,
                        help="0 processes all; supervisor can use 1 to isolate native exits")
    args = parser.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha256(args.checkpoint) != lock["frozen_checkpoint_sha256"]:
        raise RuntimeError("Frozen checkpoint SHA-256 mismatch")
    if sha256(args.official_cnn) != lock["official_cnn_source_sha256"]:
        raise RuntimeError("Official CNN source SHA-256 mismatch")
    files = sorted(args.train_waveforms.glob("*.npz"))
    if len(files) != 296:
        raise RuntimeError(f"Expected 296 TRAIN EDF caches, got {len(files)}")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    files = files[args.shard_index::args.num_shards]
    args.output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the locked frozen-CNN extraction")
    print(json.dumps({"stage": "loading_frozen_model", "device": str(device)}), flush=True)
    model, prep = model_and_preprocessor(args.official_cnn, args.checkpoint, device)
    print(json.dumps({"stage": "frozen_model_loaded"}), flush=True)
    protocol_sha = sha256(args.protocol)
    started = time.monotonic(); inference_seconds = 0.0; new_count = 0
    for ordinal, train_path in enumerate(files, 1):
        with np.load(train_path, allow_pickle=False) as old:
            patient, edf = str(old["patient"]), str(old["edf"])
            channels = old["channel_names"].astype(str)
            labels = np.asarray(old["labels"], dtype=np.int8)
            historical_starts = np.asarray(old["starts"], dtype=np.int64)
        key = output_key(edf)
        dest, marker = args.output / f"{key}.npz", args.output / f"{key}.json"
        binding = {
            "edf": edf, "source_train_sha256": sha256(train_path),
            "checkpoint_sha256": lock["frozen_checkpoint_sha256"],
            "official_cnn_source_sha256": lock["official_cnn_source_sha256"],
            "protocol_sha256": protocol_sha,
        }
        if validate_resume(dest, marker, binding):
            print(json.dumps({"ordinal": ordinal, "of": len(files), "edf": edf, "reused": True}), flush=True)
            continue
        print(json.dumps({"stage": "starting_edf", "ordinal": ordinal, "of": len(files), "edf": edf,
                          "channels": len(channels)}), flush=True)
        h5_path = args.signal_cache / Path(edf).with_suffix(".edf.h5")
        if not h5_path.is_file():
            raise FileNotFoundError(h5_path)
        edf_started = time.monotonic()
        with h5py.File(h5_path, "r") as h5:
            if str(h5.attrs["dataset_revision"]) != lock["dataset_revision"]:
                raise RuntimeError(f"Dataset revision mismatch: {edf}")
            metadata = json.loads(h5["metadata_json"][()].decode("utf-8"))
            headers = metadata["signal_headers"]
            index = {str(header["label"]): i for i, header in enumerate(headers)}
            absent = [name for name in channels if name not in index]
            if absent:
                raise RuntimeError(f"{len(absent)} selected channels absent: {edf}")
            common_starts = None; probabilities = []
            local_inference = 0.0
            for channel in channels:
                signal = physical_signal(h5, index[channel], headers[index[channel]])
                starts = starts_for(len(signal))
                if common_starts is None:
                    common_starts = starts
                elif not np.array_equal(starts, common_starts):
                    raise RuntimeError(f"Channel lengths differ within EDF: {edf}")
                tic = time.monotonic()
                probabilities.append(infer_channel(signal, starts, model, prep, device, args.batch_size))
                local_inference += time.monotonic() - tic
        inference_seconds += local_inference
        duration = float(metadata["file_duration_seconds"])
        atomic_npz(dest, patient=patient, edf=edf, channel_names=channels,
                   pathological_labels=labels, starts=common_starts,
                   segment_pathological_probs=np.stack(probabilities),
                   duration_seconds=duration, historical_starts=historical_starts)
        atomic_json(marker, {**binding, "output_sha256": sha256(dest),
                             "channels": int(len(channels)), "segments_per_channel": int(len(common_starts)),
                             "duration_seconds": duration, "inference_seconds": local_inference,
                             "processing_seconds": time.monotonic() - edf_started,
                             "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated())})
        new_count += 1
        print(json.dumps({"ordinal": ordinal, "of": len(files), "edf": edf, "reused": False,
                          "channels": len(channels), "segments_per_channel": len(common_starts),
                          "inference_seconds": local_inference}), flush=True)
        if args.max_new_edfs and new_count >= args.max_new_edfs:
            break
    if not args.max_new_edfs or new_count < args.max_new_edfs:
        atomic_json(args.output / f"SHARD_{args.shard_index}_OF_{args.num_shards}.json",
                    {"status": "COMPLETE", "shard_index": args.shard_index,
                     "num_shards": args.num_shards, "source_edfs_in_shard": len(files)})
    completed = len(list(args.output.glob("*.npz")))
    state = {
        "status": "COMPLETE" if completed == 296 else "PARTIAL",
        "expected_edfs": 296, "completed_edfs": completed,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "new_edfs_this_process": new_count, "wall_seconds_this_process": time.monotonic() - started,
        "cnn_inference_seconds_this_process": inference_seconds,
        "peak_gpu_memory_bytes_this_process": int(torch.cuda.max_memory_allocated()),
        "checkpoint_sha256": lock["frozen_checkpoint_sha256"],
        "official_cnn_source_sha256": lock["official_cnn_source_sha256"],
        "protocol_sha256": protocol_sha,
    }
    atomic_json(args.output / "EXTRACTION_STATE.json", state)
    print(json.dumps(state, indent=2), flush=True)


if __name__ == "__main__":
    main()
