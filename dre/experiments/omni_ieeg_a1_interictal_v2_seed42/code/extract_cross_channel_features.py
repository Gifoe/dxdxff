"""Frozen Omni signal cache -> original A1 descriptors and v2 cross-channel views.

Every good, signal-present channel contributes to the reference even when its
official pathological/normal label is unknown. Only officially labeled
channels are emitted for supervised A1 training or evaluation. This is a
representation change, not a label-driven channel-reference selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

from audit_official_labels import official_pathology_label


REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"
DESCRIPTOR_SHA256 = "18ec360e9d9ec7e5b56b05208c0ef6575c13769ab6866ef241e4d65e8a554c72"
NAMES = (
    "log_bp_delta", "log_bp_theta", "log_bp_beta", "log_bp_low_gamma",
    "log_bp_high_gamma", "rms", "variance", "line_length_per_sec",
    "spectral_entropy",
)
RATE = 300
EPS = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one(value) -> bool:
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def starts(n: int, split: str, seed: int) -> list[int]:
    """Identical to the frozen v1 Omni 60-s segment count/start algorithm."""
    if n <= 2 * RATE:
        return []
    usable = n - 2 * RATE
    sample = 60 * RATE
    if usable < sample:
        return []
    if split == "test":
        return [RATE + i * sample for i in range(usable // sample)]
    if split != "train":
        raise ValueError(split)
    count = min(5, max(1, usable // (sample // 2)))
    valid_range = usable - sample
    rng = random.Random(seed)
    if valid_range > count:
        indices = sorted(rng.sample(range(valid_range), count))
    else:
        indices = [i * valid_range // max(1, count - 1) for i in range(count)]
    return [RATE + i for i in indices]


def cross_views(features_all_good: np.ndarray) -> np.ndarray:
    """[59, all-good-C, 9] -> [59, all-good-C, 36], channel reference."""
    f = np.asarray(features_all_good, dtype=np.float32)
    if f.ndim != 3 or f.shape[0] != 59 or f.shape[1] < 1 or f.shape[2] != 9:
        raise ValueError(f"Expected [59,C>=1,9], got {f.shape}")
    reference = np.median(f, axis=1, keepdims=True)
    mad = np.median(np.abs(f - reference), axis=1, keepdims=True)
    scale = 1.4826 * mad + EPS
    delta = f - reference
    ratio = np.log((np.abs(f) + EPS) / (np.abs(reference) + EPS))
    result = np.concatenate([f, delta, delta / scale, ratio], axis=-1)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite cross-channel A1 view")
    return result.astype(np.float32, copy=False)


def convert(row, source: Path, cache: Path, output: Path, descriptors, protocol_sha: str):
    relative = Path(row.edf)
    destination = output / relative.with_suffix(".npz")
    marker = destination.with_suffix(".json")
    if marker.exists() and destination.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["protocol_sha256"] != protocol_sha or previous["edf"] != relative.as_posix():
            raise RuntimeError(f"Resume identity mismatch: {relative}")
        if destination.stat().st_size == previous["bytes"] and sha256(destination) == previous["sha256"]:
            return {"status": "reused", "segments": previous["segments"]}
        raise RuntimeError(f"Feature cache differs from frozen marker: {relative}")
    if marker.exists() or destination.exists():
        raise RuntimeError(f"Partial feature state needs inspection: {relative}")
    h5_path = cache / relative.with_suffix(".edf.h5")
    sidecar = source / Path(str(relative).replace("_ieeg.edf", "_channels.tsv"))
    with h5py.File(h5_path, "r") as h5:
        if str(h5.attrs["dataset_revision"]) != REVISION:
            raise RuntimeError(f"Frozen cache revision mismatch: {relative}")
        meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
        headers = meta["signal_headers"]
        name_to_index = {str(header["label"]): idx for idx, header in enumerate(headers)}
        channels = pd.read_csv(sidecar, sep="\t")
        if not {"name", "good", "soz", "resection"} <= set(channels):
            raise RuntimeError(f"Missing official channel fields: {relative}")
        good = channels.loc[channels["good"].map(one) &
                            channels["name"].astype(str).isin(name_to_index)].copy()
        if good["name"].duplicated().any():
            raise RuntimeError(f"Duplicate good signal channel: {relative}")
        good_names = good["name"].astype(str).tolist()
        pathology = [official_pathology_label(row.outcome, channel.resection,
                                              channel.soz, channel.good)
                     for channel in good.itertuples(index=False)]
        labeled_indices = [idx for idx, label in enumerate(pathology) if label is not None]
        if len(good_names) != row.good_signal_channels or len(labeled_indices) != row.official_labeled_channels:
            raise RuntimeError(f"Official label/reference channel audit changed: {relative}")
        if not labeled_indices:
            raise RuntimeError(f"No official labeled channel: {relative}")
        rates = {float(headers[name_to_index[name]]["sample_frequency"]) for name in good_names}
        if len(rates) != 1:
            raise RuntimeError(f"Mixed sampling rates among good channels: {relative}")
        original_rate = rates.pop()
        output_length = int(round(float(meta["file_duration_seconds"]) * RATE))
        signals = np.empty((len(good_names), output_length), dtype=np.float32)
        for out_idx, name in enumerate(good_names):
            header_idx = name_to_index[name]
            header = headers[header_idx]
            if header["dimension"] != "uV":
                raise RuntimeError(f"Unexpected physical unit: {relative} {name}")
            digital = np.asarray(h5[f"digital/ch{header_idx:04d}"][:], dtype=np.float64)
            dmin, dmax = float(header["digital_min"]), float(header["digital_max"])
            pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
            microvolts = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
            filtered = mne.filter.notch_filter(microvolts, Fs=original_rate,
                                               freqs=[60], notch_widths=2,
                                               n_jobs=1, verbose=False)
            resampled = mne.filter.resample(filtered, up=RATE, down=original_rate,
                                            npad="auto", n_jobs=1, verbose=False)
            if len(resampled) != output_length:
                raise RuntimeError(f"Resampling length mismatch: {relative}")
            signals[out_idx] = resampled.astype(np.float32)
    seed = int.from_bytes(hashlib.sha256(("42|" + relative.as_posix()).encode()).digest()[:8], "big")
    segment_starts = starts(signals.shape[1], row.official_split, seed)
    if not segment_starts:
        raise RuntimeError(f"No official 60-s segment: {relative}")
    feature_idx = [descriptors.BASE_SPECTRAL_FEATURE_NAMES.index(name) for name in NAMES]
    tensors = np.empty((len(segment_starts), 59, len(labeled_indices), 36), dtype=np.float32)
    for clip_idx, start in enumerate(segment_starts):
        clip = signals[:, start : start + 60 * RATE]
        if clip.shape != (len(good_names), 60 * RATE):
            raise RuntimeError(f"Short official 60-s clip: {relative}")
        windows = np.empty((59, len(good_names), 9), dtype=np.float32)
        for window_idx in range(59):
            window = clip[:, window_idx * RATE : (window_idx + 2) * RATE]
            windows[window_idx] = descriptors.compute_spectral_channel_features(
                window, float(RATE))[:, feature_idx]
        tensors[clip_idx] = cross_views(windows)[:, labeled_indices]
    names = np.asarray([good_names[idx] for idx in labeled_indices])
    labels = np.asarray([pathology[idx] for idx in labeled_indices], dtype=np.int8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".npz.partial")
    with partial.open("wb") as stream:
        np.savez(stream, features=tensors, channel_names=names, pathology=labels,
                 good_reference_channels=len(good_names), starts=np.asarray(segment_starts),
                 patient=str(row.patient), dataset=str(row.dataset), edf=relative.as_posix())
    os.replace(partial, destination)
    result = {"edf": relative.as_posix(), "split": row.official_split,
              "segments": len(segment_starts), "labeled_channels": len(labeled_indices),
              "good_reference_channels": len(good_names), "bytes": destination.stat().st_size,
              "sha256": sha256(destination), "protocol_sha256": protocol_sha,
              "effective_rate_hz": RATE, "source_h5_bytes": h5_path.stat().st_size}
    temp = marker.with_suffix(".json.tmp")
    temp.write_text(json.dumps(result, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, marker)
    return {"status": "built", "segments": len(segment_starts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("F:/Omni-iEEG/data"))
    parser.add_argument("--cache", type=Path, default=Path("F:/Omni-iEEG/signal_cache"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, default=Path("E:/DRE-nips/new-pipeline/7-11"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--test-freeze", type=Path)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    descriptor_path = args.source_code / "ez_features.py"
    if sha256(descriptor_path) != DESCRIPTOR_SHA256:
        raise RuntimeError("Historical A1 descriptor source hash mismatch")
    sys.path.insert(0, str(args.source_code))
    import ez_features  # noqa: E402
    protocol_sha = sha256(args.protocol)
    if args.split == "test":
        if args.test_freeze is None or not args.test_freeze.is_file():
            raise RuntimeError("Official test requires checkpoint+threshold freeze")
        freeze = json.loads(args.test_freeze.read_text(encoding="utf-8"))
        if not freeze.get("model_frozen_before_official_test") or \
                freeze.get("protocol_sha256") != protocol_sha or \
                not freeze.get("threshold_frozen_before_official_test"):
            raise RuntimeError("Official test freeze content invalid")
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    cohort = pd.read_csv(args.cohort)
    cohort = cohort.loc[cohort["official_split"] == args.split].iloc[
        args.shard_index::args.num_shards]
    if args.limit:
        cohort = cohort.head(args.limit)
    started = time.monotonic()
    for ordinal, row in enumerate(cohort.itertuples(index=False), start=1):
        if int(row.official_labeled_channels) == 0:
            print(f"{args.split} {ordinal}/{len(cohort)} no_official_label edf={row.edf}", flush=True)
            continue
        result = convert(row, args.source, args.cache, args.output, ez_features, protocol_sha)
        print(f"{args.split} {ordinal}/{len(cohort)} {result['status']} segments={result['segments']} "
              f"edf={row.edf} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
