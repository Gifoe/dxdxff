"""Reconcile disjoint waveform extraction shards before CNN training."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def checksum(path: Path) -> str:
    hashobj = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            hashobj.update(block)
    return hashobj.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=("train", "test"), required=True)
    p.add_argument("--official-split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--num-shards", type=int, required=True)
    args = p.parse_args()
    protocol_sha = checksum(args.protocol)
    cohort = pd.read_csv(args.official_split)
    expected = cohort.loc[(cohort["dataset"] != "Multicenter") &
                          (cohort["split"] == args.split) &
                          (pd.to_numeric(cohort["frequency"], errors="coerce") > 900) &
                          (cohort["interictal"].astype(str).str.lower().isin(["true", "1", "1.0"])) &
                          (pd.to_numeric(cohort["length"], errors="coerce") >= 62)]
    audits = []
    for idx in range(args.num_shards):
        file = args.output / f"EXTRACTION_{args.split.upper()}_SHARD{idx}OF{args.num_shards}.json"
        audit = json.loads(file.read_text(encoding="utf-8"))
        if (audit["protocol_sha256"] != protocol_sha or not audit["completed"] or
                audit["num_shards"] != args.num_shards or audit["shard_index"] != idx):
            raise RuntimeError(f"Shard provenance mismatch: {file}")
        audits.append(audit)
    if sum(audit["edfs"] for audit in audits) != len(expected):
        raise RuntimeError("Shard EDF count differs from official filtered split")
    samples = 0
    marker_count = 0
    for marker in args.output.rglob("*.json"):
        if marker.name.startswith("EXTRACTION_"):
            continue
        row = json.loads(marker.read_text(encoding="utf-8"))
        if "sha256" not in row:
            continue
        file = marker.with_suffix(".npz")
        if (row["protocol_sha256"] != protocol_sha or not file.is_file() or
                file.stat().st_size != row["bytes"] or checksum(file) != row["sha256"]):
            raise RuntimeError(f"Corrupt waveform cache: {file}")
        samples += int(row["samples"])
        marker_count += 1
    if samples != sum(audit["samples"] for audit in audits):
        raise RuntimeError("Shard sample sum differs from verified waveform files")
    summary = {"split": args.split, "edfs": len(expected), "samples": samples,
               "npz_files": marker_count, "source_dataset_revision":
               "73b9c5180a57828ab2a83c040e7e9d112e77b2cc",
               "sample_rate_hz": 1000, "protocol_sha256": protocol_sha,
               "num_shards": args.num_shards, "completed": True}
    dest = args.output / f"EXTRACTION_{args.split.upper()}_AUDIT.json"
    dest.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
