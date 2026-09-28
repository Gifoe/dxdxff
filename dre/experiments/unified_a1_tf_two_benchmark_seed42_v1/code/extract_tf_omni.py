"""Official Omni EDF HDF5 -> label-free A1-aligned TF features, one split at a time.

This deliberately reuses the v2 physical-unit, notch, resample, clip-start,
good-channel and labeled-channel procedures. It verifies clip starts and
channel order against the frozen v2 36D feature cache before writing output.
No outcomes or performance metrics are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd

from tf_preprocess import log_frequency_stft

V2_CODE = Path(__file__).resolve().parents[2] / "omni_ieeg_a1_interictal_v2_seed42/code"
if not V2_CODE.is_dir():
    V2_CODE = Path("E:/DRE-nips/new-pipeline/7-11/omni_a1_v2/code")
sys.path.insert(0, str(V2_CODE))
from extract_cross_channel_features import RATE, REVISION, one, starts  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def convert(row, source: Path, cache: Path, v2_features: Path, output: Path,
            protocol_sha: str, split: str) -> str:
    relative = Path(row.edf)
    target = output / relative.with_suffix(".npz")
    marker = target.with_suffix(".json")
    if target.exists() and marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if (old.get("protocol_sha256") == protocol_sha and old.get("split") == split and
                old.get("sha256") == digest(target) and old.get("bytes") == target.stat().st_size):
            return "reused"
        raise RuntimeError(f"TF cache marker mismatch: {relative}")
    if target.exists() or marker.exists():
        raise RuntimeError(f"Partial TF cache needs inspection: {relative}")
    v2_path = v2_features / relative.with_suffix(".npz")
    v2_marker = v2_path.with_suffix(".json")
    if not v2_path.is_file() or not v2_marker.is_file():
        raise RuntimeError(f"Missing exact v2 descriptor source: {relative}")
    if json.loads(v2_marker.read_text(encoding="utf-8"))["sha256"] != digest(v2_path):
        raise RuntimeError(f"V2 source feature hash changed: {relative}")
    with np.load(v2_path, allow_pickle=False) as base:
        expected_names = [str(name) for name in base["channel_names"]]
        expected_starts = [int(value) for value in base["starts"]]
        expected_segments = int(base["features"].shape[0])

    sidecar = source / Path(str(relative).replace("_ieeg.edf", "_channels.tsv"))
    with h5py.File(cache / relative.with_suffix(".edf.h5"), "r") as h5:
        if str(h5.attrs["dataset_revision"]) != REVISION:
            raise RuntimeError(f"Wrong native signal revision: {relative}")
        meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
        headers = meta["signal_headers"]
        by_name = {str(header["label"]): idx for idx, header in enumerate(headers)}
        channels = pd.read_csv(sidecar, sep="\t")
        good_names = channels.loc[channels["good"].map(one) &
                                   channels["name"].astype(str).isin(by_name),
                                   "name"].astype(str).tolist()
        labeled_names = [name for name in good_names if name in set(expected_names)]
        if labeled_names != expected_names:
            raise RuntimeError(f"Labeled-channel ordering differs from v2: {relative}")
        rates = {float(headers[by_name[name]]["sample_frequency"]) for name in good_names}
        if len(rates) != 1:
            raise RuntimeError(f"Mixed good-channel sample rates: {relative}")
        original_rate = rates.pop()
        length = int(round(float(meta["file_duration_seconds"]) * RATE))
        signals = np.empty((len(expected_names), length), dtype=np.float32)
        for out_idx, name in enumerate(expected_names):
            idx = by_name[name]
            header = headers[idx]
            if header["dimension"] != "uV":
                raise RuntimeError(f"Non-uV source channel: {relative}/{name}")
            digital = np.asarray(h5[f"digital/ch{idx:04d}"][:], dtype=np.float64)
            dmin, dmax = float(header["digital_min"]), float(header["digital_max"])
            pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
            physical = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
            filtered = mne.filter.notch_filter(physical, Fs=original_rate,
                                               freqs=[60], notch_widths=2,
                                               n_jobs=1, verbose=False)
            resampled = mne.filter.resample(filtered, up=RATE, down=original_rate,
                                            npad="auto", n_jobs=1, verbose=False)
            if len(resampled) != length:
                raise RuntimeError(f"Resampling length mismatch: {relative}")
            signals[out_idx] = resampled.astype(np.float32)
    seed = int.from_bytes(hashlib.sha256(("42|" + relative.as_posix()).encode()).digest()[:8], "big")
    actual_starts = starts(signals.shape[1], split, seed)
    if actual_starts != expected_starts or len(actual_starts) != expected_segments:
        raise RuntimeError(f"Clip timeline differs from v2: {relative}")
    data = np.stack([log_frequency_stft(signals[:, start:start + 60 * RATE], RATE)
                     for start in actual_starts]).astype(np.float32)
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_suffix(".npz.partial")
    with partial.open("wb") as stream:
        np.savez(stream, tf_log_power=data, channel_names=np.asarray(expected_names),
                 starts=np.asarray(actual_starts), edf=relative.as_posix())
    os.replace(partial, target)
    payload = {"edf": relative.as_posix(), "split": split,
               "protocol_sha256": protocol_sha, "v2_source_sha256": digest(v2_path),
               "bytes": target.stat().st_size, "sha256": digest(target),
               "shape": list(data.shape)}
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    return "built"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--cohort", type=Path, required=True)
    p.add_argument("--source", type=Path, default=Path("F:/Omni-iEEG/data"))
    p.add_argument("--cache", type=Path, default=Path("F:/Omni-iEEG/signal_cache"))
    p.add_argument("--v2-features", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--test-freeze", type=Path)
    p.add_argument("--num-shards", type=int, default=1)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    if a.split == "test":
        if not a.test_freeze or not a.test_freeze.is_file():
            raise RuntimeError("New official test TF extraction requires an A1-TF freeze file")
        frozen = json.loads(a.test_freeze.read_text(encoding="utf-8"))
        if not frozen.get("model_frozen_before_official_test") or not frozen.get("threshold_frozen_before_official_test"):
            raise RuntimeError("A1-TF model/threshold not frozen before official test")
    protocol_sha = digest(a.protocol)
    if not 0 <= a.shard_index < a.num_shards:
        raise ValueError("Invalid shard")
    rows = pd.read_csv(a.cohort)
    rows = rows.loc[rows["official_split"] == a.split].iloc[a.shard_index::a.num_shards]
    if a.limit:
        rows = rows.head(a.limit)
    for ordinal, row in enumerate(rows.itertuples(index=False), start=1):
        if int(row.official_labeled_channels) == 0:
            continue
        status = convert(row, a.source, a.cache, a.v2_features, a.output,
                         protocol_sha, a.split)
        print(f"{a.split} {ordinal}/{len(rows)} {status} edf={row.edf}", flush=True)


if __name__ == "__main__":
    main()
