"""Private per-patient CRST inputs from audited historical ictal raw cache.

Runs only after the exact Phase-0 alignment audit passes. No patient/channel
metadata or spectral arrays produced here may be committed to GitHub.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from spectral_cache import connectivity_edges, spectral_patches


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--phase0", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    phase0 = json.loads(args.phase0.read_text(encoding="utf-8"))
    if phase0["status"] not in ("PASS", "PASS_WITH_MASKED_PARTIAL_RECORDS"):
        raise RuntimeError("Ictal raw alignment gate not passed")
    if (digest(args.raw_cache) != phase0["input_sha256"]["raw"] or
            digest(args.feature_cache) != phase0["input_sha256"]["feature"] or
            digest(args.manifest) != phase0["input_sha256"]["manifest"]):
        raise RuntimeError("Phase-0 input hash mismatch")
    protocol_sha = digest(args.protocol)
    with args.manifest.open(newline="", encoding="utf-8-sig") as f:
        subjects = {row["subject_id"] for row in csv.DictReader(f)}
    sys.path.insert(0, str(args.source_root))
    from neuroez_c.dual_view_data import _channel_names, _record_key  # noqa: E402
    with args.feature_cache.open("rb") as f:
        feature = pickle.load(f)
    with args.raw_cache.open("rb") as f:
        raw = pickle.load(f)
    patient_index = feature["patient_index"]
    grouped = defaultdict(list)
    for record in raw["run_records"]:
        subject = str(record["subject_id"])
        if subject in subjects:
            grouped[subject].append(record)
    if len(grouped) != 80 or sum(map(len, grouped.values())) != 256:
        raise RuntimeError("Ictal raw records no longer match audited cohort")
    args.output.mkdir(parents=True, exist_ok=True)
    for ordinal, subject in enumerate(sorted(subjects), 1):
        patient_hash = hashlib.sha256(subject.encode()).hexdigest()[:20]
        path = args.output / f"patient_{patient_hash}.npz"
        marker = args.output / f"patient_{patient_hash}.json"
        if path.exists() and marker.exists():
            old = json.loads(marker.read_text(encoding="utf-8"))
            if old.get("protocol_sha256") != protocol_sha or old.get("sha256") != digest(path):
                raise RuntimeError("Private patient cache resume hash mismatch")
            print(f"ictal {ordinal}/80 reused", flush=True)
            continue
        if path.exists() or marker.exists():
            raise RuntimeError("Partial private patient cache needs inspection")
        canonical = list(patient_index[subject]["canonical_channels"])
        labels = np.asarray(patient_index[subject]["labels"], dtype=np.int8)
        if len(canonical) != len(labels) or len(canonical) != len(set(canonical)):
            raise RuntimeError("Invalid historical canonical labels")
        record_list = sorted(grouped[subject], key=lambda record: str(_record_key(record)))
        r, c = len(record_list), len(canonical)
        patches = np.zeros((r, c, 59, 64, 8), dtype=np.float16)
        windows = np.zeros((r, c, 59), dtype=bool)
        edges = np.zeros((r, c, c, 15), dtype=np.float16)
        present = np.zeros((r, c), dtype=bool)
        band_availability = np.zeros((r, 5), dtype=np.uint8)
        for ri, record in enumerate(record_list):
            sample = record["sample"]
            names = _channel_names(record)
            idx = {name: i for i, name in enumerate(canonical)}
            local = [idx[name] for name in names]
            if len(local) != len(set(local)):
                raise RuntimeError("Duplicate canonical channel in source record")
            raw_wave = np.asarray(sample["raw_waveform"], dtype=np.float32)
            valid = int(sample["raw_valid_samples"])
            start = int(sample["raw_valid_start_sample"])
            fs = float(sample["raw_temporal_sfreq"])
            spectral, fmask, wmask = spectral_patches(raw_wave, fs, valid, start)
            relation, bands = connectivity_edges(raw_wave, fs, valid, start)
            patches[ri, local] = spectral.astype(np.float16)
            windows[ri, local] = wmask
            edges[ri][np.ix_(local, local)] = relation.astype(np.float16)
            present[ri, local] = True
            band_availability[ri] = bands.astype(np.uint8)
        tmp = path.with_suffix(".npz.partial")
        with tmp.open("wb") as f:
            np.savez(f, patches=patches, window_mask=windows, edges=edges,
                     channel_present=present, frequency_mask=fmask,
                     connectivity_band_mask=band_availability, labels=labels,
                     channel_names=np.asarray(canonical), subject_id=subject)
        os.replace(tmp, path)
        status = {"protocol_sha256": protocol_sha, "sha256": digest(path),
                  "bytes": path.stat().st_size, "records": r, "channels": c,
                  "valid_window_count": int(windows.sum())}
        marker.write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
        print(f"ictal {ordinal}/80 converted records={r} channels={c}", flush=True)


if __name__ == "__main__":
    main()
