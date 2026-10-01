"""Convert private NPZ caches to standard gzip CSV for ABI-isolated statistics."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""): h.update(block)
    return h.hexdigest()


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64); out = np.empty_like(x); pos = x >= 0
    out[pos] = 1 / (1 + np.exp(-x[pos])); e = np.exp(x[~pos]); out[~pos] = e / (1 + e)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train5", type=Path, required=True)
    p.add_argument("--train-waveforms", type=Path, required=True)
    p.add_argument("--full-cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(parents=True, exist_ok=True)
    counts = {}
    train_path = a.output / "train5_segments.csv.gz"; rows = 0
    with gzip.open(train_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["patient", "edf", "channel", "y", "segment_index", "score"])
        for path in sorted(a.train5.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                offsets = np.asarray(z["segment_offsets"], dtype=np.int64)
                scores = 1.0 - sigmoid(z["segment_logits"])
                for i, (channel, y) in enumerate(zip(z["channel_names"].astype(str), z["pathological_labels"])):
                    if int(y) not in (0, 1): continue
                    for j, score in enumerate(scores[offsets[i]:offsets[i + 1]]):
                        w.writerow([str(z["patient"]), str(z["edf"]), channel, int(y), j, repr(float(score))]); rows += 1
    counts["train5_segment_rows"] = rows
    full_path = a.output / "train_full_segments.csv.gz"; rows = 0
    with gzip.open(full_path, "wt", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["patient", "edf", "channel", "y", "duration_seconds", "segment_index", "score"])
        for path in sorted(a.full_cache.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                for channel, y, values in zip(z["channel_names"].astype(str), z["pathological_labels"], z["segment_pathological_probs"]):
                    if int(y) not in (0, 1): continue
                    for j, score in enumerate(values):
                        w.writerow([str(z["patient"]), str(z["edf"]), channel, int(y), repr(float(z["duration_seconds"])), j, repr(float(score))]); rows += 1
    counts["train_full_segment_rows"] = rows
    metadata_path = a.output / "train_cache_metadata.csv"
    with metadata_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f); w.writerow(["patient", "edf", "historical_extracted_windows", "historical_windows_overlap"])
        for path in sorted(a.train_waveforms.glob("*.npz")):
            with np.load(path, allow_pickle=False) as z:
                starts = np.sort(np.asarray(z["starts"], dtype=np.int64))
                w.writerow([str(z["patient"]), str(z["edf"]), len(starts), int(np.any(np.diff(starts) < 60000)) if len(starts) > 1 else 0])
    payload = {**counts, "train_edfs": 296,
               "train5_segments_sha256": digest(train_path),
               "train_full_segments_sha256": digest(full_path),
               "train_cache_metadata_sha256": digest(metadata_path)}
    (a.output / "CONSOLIDATED_INPUT_AUDIT.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2), flush=True)


if __name__ == "__main__": main()
