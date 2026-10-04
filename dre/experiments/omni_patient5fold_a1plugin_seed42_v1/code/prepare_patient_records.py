#!/usr/bin/env python3
"""Create a private, label-audited, one-60-s-record-per-EDF Omni cache.

The cache is label-complete for cohort construction but remains private.  It
contains every good channel (including ``-1`` context-only channels) from all
eligible EDFs belonging to the frozen both-class cohort.  The single 60-s
window starts one second after recording onset, a fixed label-blind rule that
is identical for Baseline and Plugin and covers every official ``length>=62``
recording.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import mne
import numpy as np
import pandas as pd


REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"
RATE, SECONDS, START = 1000, 60, 1000


def flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "1.0", "true"}
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def label(outcome: object, resection: object, soz: object, good: object) -> int:
    if not flag(good):
        return -1
    if flag(outcome) and not flag(resection):
        return 0
    if flag(soz):
        return 1
    return -1


def physical_signal(h5: h5py.File, index: int, header: dict) -> np.ndarray:
    if str(header["dimension"]).casefold() not in {"uv", "µv", "μv"}:
        raise RuntimeError("Unexpected physical unit")
    digital = np.asarray(h5[f"digital/ch{index:04d}"][:], dtype=np.float64)
    dmin, dmax = float(header["digital_min"]), float(header["digital_max"])
    pmin, pmax = float(header["physical_min"]), float(header["physical_max"])
    if dmax <= dmin:
        raise RuntimeError("Invalid EDF digital range")
    signal = (digital - dmin) * ((pmax - pmin) / (dmax - dmin)) + pmin
    source_rate = float(header["sample_frequency"])
    if source_rate < 900:
        raise RuntimeError("Task2 source rate below 900Hz")
    signal = mne.filter.notch_filter(signal, Fs=source_rate, freqs=[60], notch_widths=2,
                                     n_jobs=1, verbose=False)
    if source_rate != RATE:
        signal = mne.filter.resample(signal, up=RATE, down=source_rate, npad="auto", n_jobs=1,
                                     verbose=False)
    return np.asarray(signal, dtype=np.float32)


def eligible_rows(split: pd.DataFrame, patients: set[str]) -> pd.DataFrame:
    rows = split.loc[(split.dataset.astype(str) != "Multicenter") & split.interictal.map(flag) &
                     (pd.to_numeric(split.frequency, errors="coerce") > 900) &
                     (pd.to_numeric(split.length, errors="coerce") >= 62)].copy()
    rows = rows.loc[rows.patient_name.astype(str).isin(patients)].copy()
    return rows.sort_values(["patient_name", "edf_name"]).reset_index(drop=True)


def convert(row, *, source: Path, signal_cache: Path, output: Path, protocol_sha: str) -> dict:
    rel = Path(str(row.edf_name))
    if rel.is_absolute() or ".." in rel.parts or rel.suffix.casefold() != ".edf":
        raise RuntimeError(f"Unsafe EDF path {rel}")
    dest = output / ("edf_" + hashlib.sha256(rel.as_posix().encode()).hexdigest()[:24] + ".npz")
    marker = dest.with_suffix(".json")
    if dest.exists() and marker.exists():
        previous = json.loads(marker.read_text(encoding="utf-8"))
        if previous.get("protocol_sha256") != protocol_sha or previous.get("sha256") != sha256(dest):
            raise RuntimeError("Private record resume mismatch")
        return {"reused": True, "channels": int(previous["channels"]), "labeled": int(previous["labeled"]),
                "unknown_context": int(previous["unknown_context"])}
    if dest.exists() or marker.exists():
        raise RuntimeError("Partial private record cache")
    sidecar = source / Path(str(rel).replace("_ieeg.edf", "_channels.tsv"))
    h5path = signal_cache / rel.with_suffix(".edf.h5")
    if not sidecar.is_file() or not h5path.is_file():
        raise FileNotFoundError("Missing frozen source/cache")
    channels = pd.read_csv(sidecar, sep="\t")
    needed = {"name", "good", "resection", "soz"}
    if not needed.issubset(channels.columns) or channels.name.astype(str).duplicated().any():
        raise RuntimeError("Invalid channel metadata")
    with h5py.File(h5path, "r") as h5:
        if str(h5.attrs["dataset_revision"]) != REVISION:
            raise RuntimeError("Frozen signal-cache revision mismatch")
        metadata = json.loads(h5["metadata_json"][()].decode("utf-8"))
        headers = metadata["signal_headers"]
        channel_index = {str(header["label"]): i for i, header in enumerate(headers)}
        selected = channels.loc[channels.good.map(flag)].copy()
        if selected.empty or not selected.name.astype(str).isin(channel_index).all():
            raise RuntimeError("Good channel unavailable in HDF5")
        # NPZ must remain safe to load with ``allow_pickle=False``.  Pandas'
        # default object dtype would otherwise create a pickle-backed array.
        names = selected.name.astype(str).to_numpy(dtype=str)
        targets = np.asarray([label(row.outcome, item.resection, item.soz, item.good)
                              for item in selected.itertuples(index=False)], dtype=np.int8)
        waves = np.empty((len(names), RATE * SECONDS), dtype=np.float32)
        for position, name in enumerate(names):
            signal = physical_signal(h5, channel_index[name], headers[channel_index[name]])
            wave = signal[START:START + RATE * SECONDS]
            if wave.shape != (RATE * SECONDS,):
                raise RuntimeError("Eligibility length does not contain fixed 60-s window")
            waves[position] = wave
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(".npz.partial")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, waveforms=waves, labels=targets, channel_names=names,
                            patient_name=str(row.patient_name), edf_name=rel.as_posix(),
                            fixed_start_sample=START, sampling_rate_hz=float(RATE))
    os.replace(partial, dest)
    status = {"protocol_sha256": protocol_sha, "sha256": sha256(dest), "channels": len(names),
              "labeled": int(np.sum(targets >= 0)), "unknown_context": int(np.sum(targets < 0)),
              "bytes": dest.stat().st_size, "fixed_start_sample": START}
    temporary = marker.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, marker)
    return {"reused": False, "channels": len(names), "labeled": status["labeled"],
            "unknown_context": status["unknown_context"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--signal-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--shards", type=int, required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    args = parser.parse_args()
    if args.shards < 1 or not 0 <= args.shard_index < args.shards:
        raise ValueError("Invalid shard selection")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if protocol["patient_manifest_private_sha256"] != sha256(args.private_manifest):
        raise RuntimeError("Private manifest differs from frozen cohort lock")
    manifest = pd.read_csv(args.private_manifest)
    if not (manifest.n_pathological_channels.gt(0) & manifest.n_normal_channels.gt(0)).all():
        raise RuntimeError("Non-both-class patient in private manifest")
    rows = eligible_rows(pd.read_csv(args.official_split), set(manifest.patient_name.astype(str)))
    if rows.patient_name.nunique() != len(manifest) or len(rows) != int(manifest.n_eligible_edfs.sum()):
        raise RuntimeError("Record set differs from frozen manifest")
    args.output.mkdir(parents=True, exist_ok=True)
    subset = rows.iloc[args.shard_index::args.shards]
    totals = {"edfs": 0, "channels": 0, "labeled": 0, "unknown_context": 0}
    protocol_sha = sha256(args.protocol)
    for ordinal, row in enumerate(subset.itertuples(index=False), 1):
        result = convert(row, source=args.sidecar_root, signal_cache=args.signal_cache,
                         output=args.output, protocol_sha=protocol_sha)
        totals["edfs"] += 1; totals["channels"] += result["channels"]; totals["labeled"] += result["labeled"]
        totals["unknown_context"] += result["unknown_context"]
        print(json.dumps({"ordinal": ordinal, "of": len(subset), "reused": result["reused"], **totals}), flush=True)
    marker = args.output / f"SHARD_{args.shard_index:02d}_OF_{args.shards:02d}.json"
    marker.write_text(json.dumps({"status": "COMPLETE", "shard": args.shard_index, "shards": args.shards,
                                  "protocol_sha256": protocol_sha, **totals}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
