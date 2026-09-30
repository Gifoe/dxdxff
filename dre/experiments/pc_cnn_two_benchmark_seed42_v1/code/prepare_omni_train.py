"""Private synchronized Omni TRAIN records for the single-record PC-CNN.

Test EDFs are rejected. A separate, post-freeze test entry point is required.
Each train EDF is read from the frozen native-digital HDF5 plus original
sidecar. The same 60-s starts apply to *all* good channels in a record.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd


DATASET_REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"
DESCRIPTOR_SHA256 = "18ec360e9d9ec7e5b56b05208c0ef6575c13769ab6866ef241e4d65e8a554c72"
NAMES = ("log_bp_delta", "log_bp_theta", "log_bp_beta", "log_bp_low_gamma",
         "log_bp_high_gamma", "rms", "variance", "line_length_per_sec",
         "spectral_entropy")
EPS = 1e-5


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def exact_descriptor_module(path: Path):
    if sha(path) != DESCRIPTOR_SHA256:
        raise RuntimeError("Historical A1 descriptor code differs")
    spec = importlib.util.spec_from_file_location("pinned_a1_ez_features", path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def flag(value) -> bool:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("true", "false"):
            return lowered == "true"
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def official_pathology_label(outcome, resection, soz, good) -> int:
    if not flag(good):
        return -1
    # Frozen official ordering: normal if good outcome and not resected;
    # otherwise pathological if SOZ. Unknown channels remain context only.
    if flag(outcome) and not flag(resection):
        return 0
    if flag(soz):
        return 1
    return -1


def train_starts_300(n: int, seed: int) -> list[int]:
    """Exact frozen v2 synchronized start algorithm at 300 Hz."""
    rate = 300
    usable = n - 2 * rate
    sample = 60 * rate
    if usable < sample:
        return []
    count = min(5, max(1, usable // (sample // 2)))
    valid = usable - sample
    rng = random.Random(seed)
    if valid > count:
        indices = sorted(rng.sample(range(valid), count))
    else:
        indices = [i * valid // max(1, count - 1) for i in range(count)]
    return [rate + i for i in indices]


def cross_views(features: np.ndarray) -> np.ndarray:
    """Exact v2 contemporaneous all-good-channel reference [59,C,9]->36."""
    reference = np.median(features, axis=1, keepdims=True)
    mad = np.median(np.abs(features - reference), axis=1, keepdims=True)
    delta = features - reference
    scale = 1.4826 * mad + EPS
    logratio = np.log((np.abs(features) + EPS) / (np.abs(reference) + EPS))
    result = np.concatenate([features, delta, delta / scale, logratio], axis=-1)
    if not np.isfinite(result).all():
        raise RuntimeError("Nonfinite Omni descriptor view")
    return result.astype(np.float32)


def physical_signal(h5: h5py.File, idx: int, header: dict) -> np.ndarray:
    if str(header["dimension"]).lower() not in ("uv", "µv", "μv"):
        raise RuntimeError("Unexpected EDF physical unit")
    digital = np.asarray(h5[f"digital/ch{idx:04d}"][:], dtype=np.float64)
    dmin, dmax = float(header["digital_min"]), float(header["digital_max"])
    pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
    if dmax <= dmin:
        raise RuntimeError("Invalid EDF scaling")
    physical = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
    rate = float(header["sample_frequency"])
    if rate < 900:
        raise RuntimeError("Source rate outside official Task-2 scope")
    physical = mne.filter.notch_filter(physical, Fs=rate, freqs=[60],
                                       notch_widths=2, n_jobs=1, verbose=False)
    if rate != 1000:
        physical = mne.filter.resample(physical, up=1000, down=rate,
                                       npad="auto", n_jobs=1, verbose=False)
    # Preserve the float64 full-record signal until the validated 300-Hz
    # descriptor resample; MNE's resampler rejects float32 here. The stored
    # RawCNN waveform is converted to float32 only on assignment below.
    return np.asarray(physical, dtype=np.float64)


def convert(row, source: Path, signal_cache: Path, destination_root: Path,
            descriptor_module, protocol_sha: str) -> dict:
    rel = Path(str(row.edf_name))
    if rel.is_absolute() or ".." in rel.parts or rel.suffix.lower() != ".edf":
        raise RuntimeError("Unsafe EDF path")
    dest = destination_root / ("edf_" + hashlib.sha256(rel.as_posix().encode()).hexdigest()[:24] + ".npz")
    marker = dest.with_suffix(".json")
    if dest.exists() and marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != protocol_sha or old.get("sha256") != sha(dest):
            raise RuntimeError("Private Omni cache resume mismatch")
        return {"reused": True, "clips": old["clips"],
                "labeled_channels": old["labeled_channels"]}
    if dest.exists() or marker.exists():
        raise RuntimeError("Partial private Omni cache needs manual inspection")
    sidecar = source / Path(str(rel).replace("_ieeg.edf", "_channels.tsv"))
    h5path = signal_cache / rel.with_suffix(".edf.h5")
    if not sidecar.is_file() or not h5path.is_file():
        raise RuntimeError("Frozen Omni signal/sidecar missing")
    channels = pd.read_csv(sidecar, sep="\t")
    with h5py.File(h5path, "r") as h5:
        if str(h5.attrs["dataset_revision"]) != DATASET_REVISION:
            raise RuntimeError("Frozen dataset revision mismatch")
        meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
        headers = meta["signal_headers"]
        index = {str(header["label"]): i for i, header in enumerate(headers)}
        good = channels.loc[channels["good"].map(flag) &
                            channels["name"].astype(str).isin(index)].copy()
        if good.empty or good["name"].duplicated().any():
            raise RuntimeError("No unique good channels")
        names = good["name"].astype(str).tolist()
        rates = {float(headers[index[name]]["sample_frequency"]) for name in names}
        if len(rates) != 1:
            raise RuntimeError("Mixed channel rates within EDF")
        labels = np.asarray([official_pathology_label(row.outcome, ch.resection,
                                                       ch.soz, ch.good)
                             for ch in good.itertuples(index=False)], dtype=np.int8)
        if not np.any(labels >= 0):
            return {"reused": False, "clips": 0, "labeled_channels": 0}
        seed = int.from_bytes(hashlib.sha256(("42|" + rel.as_posix()).encode()).digest()[:8], "big")
        estimated_300 = int(round(float(meta["file_duration_seconds"]) * 300))
        starts300 = train_starts_300(estimated_300, seed)
        if not starts300:
            raise RuntimeError("No synchronized 60-s train clip")
        starts1000 = [int(round(s * 1000 / 300)) for s in starts300]
        waves = np.empty((len(starts300), len(names), 60000), dtype=np.float32)
        down_clips = np.empty((len(starts300), len(names), 18000), dtype=np.float32)
        # Channel-wise full-record filtering uses bounded memory even for a
        # long EDF. Starts remain shared across all channels.
        for channel, name in enumerate(names):
            full = physical_signal(h5, index[name], headers[index[name]])
            down = mne.filter.resample(full, up=300, down=1000,
                                       npad="auto", n_jobs=1, verbose=False)
            for clip, (s1000, s300) in enumerate(zip(starts1000, starts300)):
                wave = full[s1000:s1000 + 60000]
                reduced = down[s300:s300 + 18000]
                if wave.shape != (60000,) or reduced.shape != (18000,):
                    raise RuntimeError("Short aligned train clip")
                waves[clip, channel] = wave
                down_clips[clip, channel] = reduced
    idx9 = [descriptor_module.BASE_SPECTRAL_FEATURE_NAMES.index(name) for name in NAMES]
    descriptors = []
    for down in down_clips:
        feat = np.stack([descriptor_module.compute_spectral_channel_features(
            down[:, j * 300:(j + 2) * 300], 300.0)[:, idx9]
            for j in range(59)])
        descriptors.append(np.transpose(cross_views(feat), (1, 0, 2)))
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".npz.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, waveforms=waves,
                            descriptors=np.asarray(descriptors, dtype=np.float32),
                            labels=labels, channel_names=np.asarray(names),
                            starts=np.asarray(starts1000, dtype=np.int64),
                            patient=str(row.patient_name), edf=rel.as_posix(),
                            sampling_rate_hz=1000.0)
    os.replace(partial, dest)
    status = {"protocol_sha256": protocol_sha, "sha256": sha(dest),
              "bytes": dest.stat().st_size, "clips": len(starts300),
              "channels": len(names), "labeled_channels": int(np.sum(labels >= 0))}
    temp = marker.with_suffix(".json.tmp")
    temp.write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, marker)
    return {"reused": False, "clips": len(starts300),
            "labeled_channels": status["labeled_channels"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--official-split", type=Path, required=True)
    p.add_argument("--train-val-split", type=Path, required=True)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--signal-cache", type=Path, required=True)
    p.add_argument("--descriptor-source", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    if sha(args.official_split) != lock["omni_official_split_sha256"] or \
            sha(args.train_val_split) != lock["omni_inner_train_val_split_sha256"]:
        raise RuntimeError("Official/train-val split differs from lock")
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid shard")
    module = exact_descriptor_module(args.descriptor_source)
    roles = pd.read_csv(args.train_val_split).set_index("patient")["role"].to_dict()
    if len(roles) != 141 or set(roles.values()) != {"inner_train", "inner_val"}:
        raise RuntimeError("Frozen Omni 141-patient roles differ")
    rows = pd.read_csv(args.official_split)
    rows = rows.loc[(rows["split"] == "train") & (rows["dataset"] != "Multicenter") &
                    (pd.to_numeric(rows["frequency"], errors="coerce") > 900) &
                    rows["interictal"].map(flag) &
                    (pd.to_numeric(rows["length"], errors="coerce") >= 62)]
    rows = rows.loc[rows["patient_name"].astype(str).isin(roles)]
    if rows["patient_name"].nunique() != 141:
        raise RuntimeError("Eligible Omni train cohort differs")
    args.output.mkdir(parents=True, exist_ok=True)
    subset = rows.iloc[args.shard_index::args.shards]
    if args.limit:
        subset = subset.iloc[:args.limit]
    totals = {"edfs": 0, "clips": 0, "labeled_channels": 0,
              "test_split_accessed": False}
    protocol_sha = sha(args.protocol)
    for i, row in enumerate(subset.itertuples(index=False), 1):
        result = convert(row, args.source, args.signal_cache, args.output,
                         module, protocol_sha)
        totals["edfs"] += int(result["clips"] > 0)
        totals["clips"] += result["clips"]
        totals["labeled_channels"] += result["labeled_channels"]
        print(json.dumps({"ordinal": i, "of": len(subset),
                          "reused": result["reused"], "clips": result["clips"]}), flush=True)
    print(json.dumps({"status": "TRAIN_EXTRACTION_COMPLETE", **totals}), flush=True)


if __name__ == "__main__":
    main()
