#!/usr/bin/env python3
"""Losslessly repair a private NPZ string-dtype serialization error.

No source HDF5/EDF is reread.  Every waveform and label array is SHA-256
compared before the repaired file is atomically promoted into a new cache root.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


def bytes_hash(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-records", type=int, required=True)
    args = parser.parse_args()
    source = sorted(args.input.glob("edf_*.npz"))
    if len(source) != args.expected_records:
        raise RuntimeError("Source cache record count mismatch")
    args.output.mkdir(parents=True, exist_ok=True)
    completed = 0
    for ordinal, path in enumerate(source, 1):
        destination = args.output / path.name
        marker = destination.with_suffix(".json")
        if destination.exists() and marker.exists():
            completed += 1; continue
        if destination.exists() or marker.exists():
            raise RuntimeError("Partial migrated record")
        # This is trusted self-produced cache content.  The output is repaired
        # to a safe non-pickle string array and all later consumers use
        # allow_pickle=False.
        with np.load(path, allow_pickle=True) as data:
            waves = data["waveforms"]
            labels = data["labels"]
            names = np.asarray(data["channel_names"], dtype=str)
            patient = np.asarray(data["patient_name"])
            edf = np.asarray(data["edf_name"])
            start = np.asarray(data["fixed_start_sample"])
            rate = np.asarray(data["sampling_rate_hz"])
        before = {"waveforms": bytes_hash(waves), "labels": bytes_hash(labels),
                  "channel_names_utf8": hashlib.sha256("\0".join(names.tolist()).encode("utf-8")).hexdigest()}
        partial = destination.with_suffix(".npz.partial")
        with partial.open("wb") as stream:
            np.savez_compressed(stream, waveforms=waves, labels=labels, channel_names=names,
                                patient_name=patient, edf_name=edf,
                                fixed_start_sample=start, sampling_rate_hz=rate)
        with np.load(partial, allow_pickle=False) as repaired:
            after = {"waveforms": bytes_hash(repaired["waveforms"]), "labels": bytes_hash(repaired["labels"]),
                     "channel_names_utf8": hashlib.sha256("\0".join(repaired["channel_names"].astype(str).tolist()).encode("utf-8")).hexdigest()}
        if before != after:
            partial.unlink(missing_ok=True)
            raise RuntimeError("Lossless migration hash mismatch")
        os.replace(partial, destination)
        temporary = marker.with_suffix(".json.tmp")
        temporary.write_text(json.dumps({"status": "MIGRATED_LOSSLESS", "source_file_sha256": file_hash(path),
                                         "sha256": file_hash(destination), "array_hashes": after}, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, marker)
        completed += 1
        print(json.dumps({"ordinal": ordinal, "of": len(source), "complete": completed}), flush=True)
    # The source shard markers certify the record membership.  They are copied
    # verbatim only after every migrated EDF has passed its array hashes.
    for marker in sorted(args.input.glob("SHARD_*_OF_04.json")):
        destination = args.output / marker.name
        if not destination.exists():
            destination.write_bytes(marker.read_bytes())
    (args.output / "MIGRATION_COMPLETE.json").write_text(json.dumps({"status": "PASS", "records": completed}) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
