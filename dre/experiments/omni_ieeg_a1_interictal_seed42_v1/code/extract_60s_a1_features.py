"""Official-split 60-s clips -> exact A1 descriptors and 36-D interictal views.

Uses native digital HDF5, reconstructs physical microvolts, applies the Omni
channel benchmark's full-record 60-Hz notch and 300-Hz anti-aliased resampling,
then computes the original A1 spectral/classical descriptor function.  No
labels enter waveform preprocessing.  A single shared segment start per EDF is
needed for A1's cross-channel attention (the official independent channel
classifier samples starts separately per channel).
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


OMNI_REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"
SOURCE_HASH = "18ec360e9d9ec7e5b56b05208c0ef6575c13769ab6866ef241e4d65e8a554c72"
NAMES = (
    "log_bp_delta", "log_bp_theta", "log_bp_beta", "log_bp_low_gamma",
    "log_bp_high_gamma", "rms", "variance", "line_length_per_sec",
    "spectral_entropy",
)
TARGET_RATE = 300
EPS = 1e-5


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def starts(n: int, split: str, seed: int) -> list[int]:
    """Exact Omni train/test segment-count/start algorithm at 300 Hz.

    The training random seed is now explicitly frozen per EDF.  Starts are
    shared by its channels so patient-level attention sees synchronous data.
    """
    rate = TARGET_RATE
    if n <= 2 * rate:
        return []
    usable = n - 2 * rate
    sample = 60 * rate
    if usable < sample:
        return []
    if split == "test":
        return [rate + i * sample for i in range(usable // sample)]
    if split != "train":
        raise ValueError(split)
    n_samples = min(5, max(1, usable // (sample // 2)))
    valid_range = usable - sample
    rng = random.Random(seed)
    if valid_range > n_samples:
        indices = sorted(rng.sample(range(valid_range), n_samples))
    else:
        indices = [i * valid_range // max(1, n_samples - 1) for i in range(n_samples)]
    return [rate + i for i in indices]


def views(features: np.ndarray) -> np.ndarray:
    """[59,C,9] -> A1 ABS/DELTA/ZDELTA/ratio [59,C,36]."""
    f = np.asarray(features, dtype=np.float32)
    if f.ndim != 3 or f.shape[0] != 59 or f.shape[2] != 9:
        raise ValueError(f"Expected [59,C,9], got {f.shape}")
    reference = np.median(f, axis=0, keepdims=True)
    mad = np.median(np.abs(f - reference), axis=0, keepdims=True)
    scale = 1.4826 * mad + EPS
    delta = f - reference
    ratio = np.log((np.abs(f) + EPS) / (np.abs(reference) + EPS))
    out = np.concatenate([f, delta, delta / scale, ratio], axis=-1)
    if not np.isfinite(out).all():
        raise RuntimeError("Nonfinite interictal views")
    return out.astype(np.float32, copy=False)


def convert_edf(row, cache: Path, output: Path, source_module, protocol_sha: str) -> dict:
    relative = Path(row.edf)
    h5path = cache / relative.with_suffix(".edf.h5")
    destination = output / relative.with_suffix(".npz")
    marker = destination.with_suffix(".json")
    if marker.exists() and destination.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous["protocol_sha256"] != protocol_sha or previous["edf"] != relative.as_posix():
            raise RuntimeError(f"Resume marker identity mismatch: {relative}")
        if destination.stat().st_size == previous["bytes"] and sha256(destination) == previous["sha256"]:
            return {"edf": relative.as_posix(), "status": "reused", "segments": previous["segments"]}
        raise RuntimeError(f"Existing feature file differs from marker: {relative}")
    if destination.exists() or marker.exists():
        raise RuntimeError(f"Partial feature state needs inspection: {relative}")
    with h5py.File(h5path, "r") as h5:
        if h5.attrs["dataset_revision"] != OMNI_REVISION:
            raise RuntimeError(f"Revision mismatch: {relative}")
        meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
        headers = meta["signal_headers"]
        names = [str(h["label"]) for h in headers]
        name_to_idx = {name: idx for idx, name in enumerate(names)}
        sidecar = pd.read_csv(Path(row.source_root) / str(relative).replace("_ieeg.edf", "_channels.tsv"), sep="\t")
        soz_numeric = pd.to_numeric(sidecar["soz"], errors="coerce")
        valid = sidecar.loc[(pd.to_numeric(sidecar["good"], errors="coerce") == 1)
                            & soz_numeric.isin([0, 1])
                            & sidecar["name"].astype(str).isin(name_to_idx)]
        if len(valid) != row.valid_channels:
            raise RuntimeError(f"Cohort channel count changed: {relative}")
        channel_names = valid["name"].astype(str).tolist()
        labels = valid["soz"].astype(np.int8).to_numpy()
        if not channel_names:
            raise RuntimeError(f"No valid supervised channels: {relative}")
        rates = {float(headers[name_to_idx[name]]["sample_frequency"]) for name in channel_names}
        if len(rates) != 1:
            raise RuntimeError(f"Mixed sampling rates: {relative}")
        rate = rates.pop()
        out_length = int(round(float(meta["file_duration_seconds"]) * TARGET_RATE))
        signals = np.empty((len(channel_names), out_length), dtype=np.float32)
        for out_idx, name in enumerate(channel_names):
            idx = name_to_idx[name]
            header = headers[idx]
            if header["dimension"] != "uV":
                raise RuntimeError(f"Unexpected physical unit: {relative} {name}")
            digital = np.asarray(h5[f"digital/ch{idx:04d}"][:], dtype=np.float64)
            dmin, dmax = float(header["digital_min"]), float(header["digital_max"])
            pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
            microvolts = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
            # MNE's filtering and anti-aliased resampling are channel-separable.
            filtered = mne.filter.notch_filter(microvolts, Fs=rate, freqs=[60],
                                               notch_widths=2, n_jobs=1, verbose=False)
            resampled = mne.filter.resample(filtered, up=TARGET_RATE, down=rate,
                                            npad="auto", n_jobs=1, verbose=False)
            if len(resampled) != out_length:
                raise RuntimeError(f"Unexpected resample length: {relative} {len(resampled)} vs {out_length}")
            signals[out_idx] = resampled.astype(np.float32)
    seed = int.from_bytes(hashlib.sha256(("42|" + relative.as_posix()).encode()).digest()[:8], "big")
    segment_starts = starts(signals.shape[1], row.official_split, seed)
    if not segment_starts:
        raise RuntimeError(f"No 60-s segment after official preprocessing: {relative}")
    feature_idx = [source_module.BASE_SPECTRAL_FEATURE_NAMES.index(name) for name in NAMES]
    tensors = np.empty((len(segment_starts), 59, len(channel_names), 36), dtype=np.float32)
    for segment_idx, start in enumerate(segment_starts):
        clip = signals[:, start : start + 60 * TARGET_RATE]
        if clip.shape != (len(channel_names), 60 * TARGET_RATE):
            raise RuntimeError(f"Short 60-s clip: {relative}")
        sequence = np.empty((59, len(channel_names), 9), dtype=np.float32)
        for window_idx in range(59):
            window = clip[:, window_idx * TARGET_RATE : (window_idx + 2) * TARGET_RATE]
            sequence[window_idx] = source_module.compute_spectral_channel_features(
                window, float(TARGET_RATE))[:, feature_idx]
        tensors[segment_idx] = views(sequence)
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(".npz.partial")
    with partial.open("wb") as handle:
        np.savez(handle, features=tensors, channel_names=np.asarray(channel_names),
                 soz=labels, starts=np.asarray(segment_starts, dtype=np.int64),
                 patient=str(row.patient), dataset=str(row.dataset), edf=relative.as_posix())
    os.replace(partial, destination)
    digest = sha256(destination)
    payload = {"edf": relative.as_posix(), "split": row.official_split,
               "segments": len(segment_starts), "channels": len(channel_names),
               "bytes": destination.stat().st_size, "sha256": digest,
               "protocol_sha256": protocol_sha, "effective_rate_hz": TARGET_RATE,
               "source_h5_bytes": h5path.stat().st_size}
    temp = marker.with_suffix(".json.tmp")
    temp.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, marker)
    return {"edf": relative.as_posix(), "status": "built", "segments": len(segment_starts)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["train", "test"], required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path("F:/Omni-iEEG/signal_cache"))
    parser.add_argument("--source", type=Path, default=Path("F:/Omni-iEEG/data"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-code", type=Path, default=Path("E:/DRE-nips/new-pipeline/7-11"))
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--test-freeze", type=Path)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.split == "test" and (args.test_freeze is None or not args.test_freeze.is_file()):
        raise RuntimeError("Official test features require a frozen model-selection marker")
    source_file = args.source_code / "ez_features.py"
    if sha256(source_file) != SOURCE_HASH:
        raise RuntimeError("Historical A1 descriptor source hash mismatch")
    sys.path.insert(0, str(args.source_code))
    import ez_features  # noqa: E402
    protocol_sha = sha256(args.protocol)
    if args.split == "test":
        freeze = json.loads(args.test_freeze.read_text(encoding="utf-8"))
        if not freeze.get("model_frozen_before_official_test") or freeze.get("protocol_sha256") != protocol_sha:
            raise RuntimeError("Official test feature extraction needs a matching frozen protocol/model")
    cohort = pd.read_csv(args.cohort)
    cohort = cohort.loc[cohort["official_split"] == args.split].copy()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid disjoint shard index/count")
    cohort = cohort.iloc[args.shard_index::args.num_shards]
    if args.limit:
        cohort = cohort.head(args.limit)
    started = time.monotonic()
    for i, row in enumerate(cohort.itertuples(index=False), start=1):
        if int(row.valid_channels) == 0:
            print(f"{args.split} {i}/{len(cohort)} skipped_unknown_soz edf={row.edf}", flush=True)
            continue
        enriched = row._asdict()
        enriched["source_root"] = str(args.source)
        result = convert_edf(type("Row", (), enriched), args.cache, args.output,
                             ez_features, protocol_sha)
        print(f"{args.split} {i}/{len(cohort)} {result['status']} segments={result['segments']} "
              f"edf={result['edf']} elapsed={time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
