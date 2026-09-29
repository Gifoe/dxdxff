"""Reconstruct pinned Omni channel-model NPZ waveforms from lossless HDF5.

This reproduces the official channel selection and 60-s clip rules without
requiring the EDF files that were safely migrated to native-digital HDF5.
The paper/config's 1000-Hz main benchmark rate overrides the published
features.py function's inconsistent 300-Hz default. No test model result is
read here. Runtime NPZ files and audit JSON stay private on the server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"
RATE = 1000
SECONDS = 60


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def flag(value: object) -> bool:
    return str(value).lower() in ("1", "1.0", "true")


def number(value: object) -> float:
    try:
        return float(value)
    except (ValueError, TypeError):
        return float("nan")


def windows(length: int, *, train: bool, rng: random.Random) -> list[int]:
    """Exact logic of official extract_random/uniform_samples at 1000 Hz."""
    usable = length - 2 * RATE
    segment = SECONDS * RATE
    if usable < segment:
        return []
    if not train:
        return [RATE + i * segment for i in range(usable // segment)]
    count = min(5, max(1, usable // (segment // 2)))
    valid = usable - segment
    if valid > count:
        indices = sorted(rng.sample(range(valid), count))
    else:
        indices = [i * valid // max(1, count - 1) for i in range(count)]
    return [RATE + i for i in indices]


def physical_signal(h5: h5py.File, idx: int, header: dict) -> np.ndarray:
    if str(header["dimension"]).lower() not in ("uv", "µv", "μv"):
        raise ValueError(f"Unexpected EDF physical unit: {header['dimension']}")
    digital = np.asarray(h5[f"digital/ch{idx:04d}"][:], dtype=np.float64)
    dmin, dmax = float(header["digital_min"]), float(header["digital_max"])
    pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
    if dmax <= dmin:
        raise ValueError("Invalid EDF channel digital range")
    microvolts = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
    source_rate = float(header["sample_frequency"])
    if source_rate < 900:
        raise ValueError(f"Official cohort source rate <900 Hz: {source_rate}")
    filtered = mne.filter.notch_filter(microvolts, Fs=source_rate, freqs=[60],
                                       notch_widths=2, n_jobs=1, verbose=False)
    if source_rate != RATE:
        filtered = mne.filter.resample(filtered, up=RATE, down=source_rate,
                                       npad="auto", n_jobs=1, verbose=False)
    return np.asarray(filtered, dtype=np.float32)


def labels_for(row: pd.Series, channels: pd.DataFrame, *, split: str):
    outcome = number(row["outcome"])
    has_resection = flag(row["has_resection"])
    has_soz = flag(row["has_soz"])
    for channel in channels.itertuples(index=False):
        if not flag(getattr(channel, "good")):
            continue
        name = str(getattr(channel, "name"))
        resection = number(getattr(channel, "resection"))
        soz = number(getattr(channel, "soz"))
        if split == "test":
            label = 1 if outcome == 1 and resection == 0 else 0 if soz == 1 else -1
            yield name, label, "test"
        else:
            # Exact official features.py branch criteria. One channel may be
            # in both branches; the branches are intentionally separate NPZs.
            if outcome == 1 and has_resection and resection == 0:
                yield name, 1, "positive"
            if has_soz and soz == 1 and (outcome != 1 or resection != 0):
                yield name, 0, "negative"


def convert(row: pd.Series, *, split: str, source: Path, cache: Path,
            output: Path, protocol_sha: str) -> dict:
    relative = Path(str(row["edf_name"]))
    if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
        raise ValueError(f"Unsafe EDF path: {relative}")
    sidecar = source / Path(str(relative).replace("_ieeg.edf", "_channels.tsv"))
    h5_path = cache / relative.with_suffix(".edf.h5")
    if not sidecar.is_file() or not h5_path.is_file():
        raise FileNotFoundError(f"Missing frozen-manifest input: {relative}")
    channels = pd.read_csv(sidecar, sep="\t")
    selected = list(labels_for(row, channels, split=split))
    by_branch: dict[str, list[tuple[str, int]]] = {}
    for name, label, branch in selected:
        by_branch.setdefault(branch, []).append((name, label))
    if not by_branch:
        return {"edf": relative.as_posix(), "samples": 0, "reason": "no branch labels"}
    # The official extractor does not seed its parallel random sampling.
    # A per-EDF seed makes interruption/resume invariant to processing order.
    seed_material = f"42|{relative.as_posix()}".encode("utf-8")
    rng = random.Random(int.from_bytes(hashlib.sha256(seed_material).digest()[:8], "big"))
    with h5py.File(h5_path, "r") as h5:
        if str(h5.attrs["dataset_revision"]) != REVISION:
            raise RuntimeError(f"HDF5 revision differs: {relative}")
        meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
        headers = meta["signal_headers"]
        index = {str(header["label"]): i for i, header in enumerate(headers)}
        absent = [name for name, _label, _branch in selected if name not in index]
        if absent:
            raise ValueError(f"Official good channels absent in HDF5: {relative}, n={len(absent)}")
        expected = int(round(float(meta["file_duration_seconds"]) * RATE))
        written = 0
        for branch, named_labels in by_branch.items():
            dest = output / branch / relative.with_suffix(".npz").name
            marker = dest.with_suffix(".json")
            if marker.is_file() and dest.is_file():
                old = json.loads(marker.read_text(encoding="utf-8"))
                if old.get("protocol_sha256") == protocol_sha and old.get("sha256") == digest(dest):
                    written += int(old["samples"])
                    continue
                raise RuntimeError(f"Resume hash/protocol mismatch: {dest}")
            if marker.exists() or dest.exists():
                raise RuntimeError(f"Partial output requires inspection: {dest}")
            data, names, labels, starts, ends = [], [], [], [], []
            for name, label in named_labels:
                idx = index[name]
                signal = physical_signal(h5, idx, headers[idx])
                if abs(len(signal) - expected) > 1:
                    raise ValueError(f"Reconstructed sample count mismatch: {relative}/{name}")
                for start in windows(len(signal), train=(split == "train"), rng=rng):
                    wave = signal[start:start + SECONDS * RATE]
                    if wave.shape != (SECONDS * RATE,):
                        raise RuntimeError("Short channel segment")
                    data.append(wave)
                    names.append(name)
                    labels.append(label)
                    starts.append(start)
                    ends.append(start + SECONDS * RATE)
            if not data:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            partial = dest.with_suffix(".npz.partial")
            with partial.open("wb") as stream:
                np.savez(stream, data=np.stack(data), name=np.asarray(names),
                         labels=np.asarray(labels, dtype=np.int8),
                         start_indices=np.asarray(starts, dtype=np.int64),
                         end_indices=np.asarray(ends, dtype=np.int64),
                         patient=str(row["patient_name"]), edf_name=relative.as_posix())
            os.replace(partial, dest)
            payload = {"edf": relative.as_posix(), "branch": branch,
                       "samples": len(labels), "protocol_sha256": protocol_sha,
                       "sha256": digest(dest), "bytes": dest.stat().st_size,
                       "sample_rate_hz": RATE}
            temporary = marker.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
            os.replace(temporary, marker)
            written += len(labels)
    return {"edf": relative.as_posix(), "samples": written, "branches": sorted(by_branch)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=("train", "test"), required=True)
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    rows = pd.read_csv(args.official_split)
    rows = rows.loc[(rows["dataset"] != "Multicenter") &
                    (rows["split"] == args.split) &
                    (pd.to_numeric(rows["frequency"], errors="coerce") > 900) &
                    (rows["interictal"].map(flag)) &
                    (pd.to_numeric(rows["length"], errors="coerce") >= 62)]
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Invalid shard index/count")
    if args.limit:
        rows = rows.head(args.limit)
    rows = rows.iloc[args.shard_index::args.num_shards]
    protocol_sha = digest(args.protocol)
    total = 0
    for ordinal, (_idx, row) in enumerate(rows.iterrows(), start=1):
        result = convert(row, split=args.split, source=args.source,
                         cache=args.cache, output=args.output,
                         protocol_sha=protocol_sha)
        total += int(result["samples"])
        print(json.dumps({"ordinal": ordinal, "total_edfs": len(rows),
                          "edf": result["edf"], "samples": result["samples"],
                          "cumulative_samples": total}), flush=True)
    audit = {"split": args.split, "edfs": len(rows), "samples": total,
             "sample_rate_hz": RATE, "source_dataset_revision": REVISION,
             "protocol_sha256": protocol_sha, "completed": True,
             "num_shards": args.num_shards, "shard_index": args.shard_index}
    (args.output / f"EXTRACTION_{args.split.upper()}_SHARD{args.shard_index}OF{args.num_shards}.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
