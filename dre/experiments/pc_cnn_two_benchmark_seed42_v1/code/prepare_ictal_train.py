"""Private ictal 80-person RawCNN/PC-CNN records from phase-0-audited caches."""

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


NAMES = ("log_bp_delta", "log_bp_theta", "log_bp_beta", "log_bp_low_gamma",
         "log_bp_high_gamma", "rms", "variance", "line_length_per_sec",
         "spectral_entropy")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def views_exact_a1(features: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """Historical B0 same-channel pre-onset mean/std/absolute-ratio operator."""
    if features.ndim != 3 or features.shape[-1] != 9:
        raise ValueError("Expected [W,C,9] historical descriptor cache")
    pre = np.asarray(centers, dtype=np.float32) < 0
    if len(pre) != len(features) or not pre.any():
        raise RuntimeError("Historical same-channel pre-onset reference absent")
    baseline = features[pre]
    eps = 1e-5
    mean = baseline.mean(axis=0, keepdims=True)
    std = np.clip(baseline.std(axis=0, keepdims=True), eps, None)
    delta = features - mean
    ratio = np.log((np.abs(features) + eps) / (np.abs(mean) + eps))
    out = np.concatenate([features, delta, delta / std, ratio], axis=-1)
    if not np.isfinite(out).all():
        raise RuntimeError("Nonfinite historical A1 view")
    return out.astype(np.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--phase0", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    for kind, path in (("ictal_raw_cache_sha256", args.raw_cache),
                       ("ictal_feature_cache_sha256", args.feature_cache),
                       ("ictal_fold_manifest_sha256", args.manifest)):
        if sha(path) != lock[kind]:
            raise RuntimeError(f"Ictal source {kind} no longer matches lock")
    phase = json.loads(args.phase0.read_text(encoding="utf-8"))
    if phase["status"] not in ("PASS", "PASS_WITH_MASKED_PARTIAL_RECORDS") or \
            phase["matched_records"] != 256:
        raise RuntimeError("CRST phase-0 raw/feature alignment not passed")
    sys.path.insert(0, str(args.source_root))
    from neuroez_c.dual_view_data import _channel_names, _record_key  # noqa: E402
    with args.feature_cache.open("rb") as stream:
        feature = pickle.load(stream)
    with args.raw_cache.open("rb") as stream:
        raw = pickle.load(stream)
    with args.manifest.open(newline="", encoding="utf-8-sig") as stream:
        patients = sorted({str(row["subject_id"]) for row in csv.DictReader(stream)})
    if len(patients) != 80:
        raise RuntimeError("Historical cohort is not 80 patients")
    raw_records = defaultdict(dict)
    feature_records = defaultdict(dict)
    for record in raw["run_records"]:
        subject = str(record["subject_id"])
        if subject in patients:
            key = str(_record_key(record))
            if key in raw_records[subject]:
                raise RuntimeError("Duplicate raw record key")
            raw_records[subject][key] = record
    for record in feature["run_records"]:
        subject = str(record["subject_id"])
        if subject in patients:
            key = str(_record_key(record))
            if key in feature_records[subject]:
                raise RuntimeError("Duplicate feature record key")
            feature_records[subject][key] = record
    if sum(len(v) for v in raw_records.values()) != 256 or \
            any(set(raw_records[s]) != set(feature_records[s]) for s in patients):
        raise RuntimeError("Raw/feature record-key alignment changed")
    names9 = list(map(str, feature["window_feature_names"]))
    idx9 = [names9.index(name) for name in NAMES]
    args.output.mkdir(parents=True, exist_ok=True)
    protocol_sha = sha(args.protocol)
    totals = {"patients": 0, "records": 0, "partial_records": 0,
              "test_outcome_accessed": False}
    for ordinal, subject in enumerate(patients, 1):
        if args.limit and ordinal > args.limit:
            break
        key_hash = hashlib.sha256(subject.encode()).hexdigest()[:20]
        dest = args.output / f"patient_{key_hash}.npz"
        marker = dest.with_suffix(".json")
        if dest.exists() and marker.exists():
            old = json.loads(marker.read_text(encoding="utf-8"))
            if old.get("protocol_sha256") != protocol_sha or old.get("sha256") != sha(dest):
                raise RuntimeError("Private ictal cache resume mismatch")
            totals["patients"] += 1
            totals["records"] += old["records"]
            totals["partial_records"] += old["partial_records"]
            print(json.dumps({"ordinal": ordinal, "reused": True}), flush=True)
            continue
        if dest.exists() or marker.exists():
            raise RuntimeError("Partial private ictal cache needs inspection")
        canonical = list(map(str, feature["patient_index"][subject]["canonical_channels"]))
        labels = np.asarray(feature["patient_index"][subject]["labels"], dtype=np.int8)
        if len(canonical) != len(set(canonical)) or labels.shape != (len(canonical),):
            raise RuntimeError("Historical canonical channel/label mismatch")
        lookup = {name: i for i, name in enumerate(canonical)}
        keys = sorted(raw_records[subject])
        n, c = len(keys), len(canonical)
        waves = np.zeros((n, c, 15000), dtype=np.float32)
        descriptors = np.zeros((n, c, 59, 36), dtype=np.float32)
        window_mask = np.zeros((n, c, 59), dtype=bool)
        present = np.zeros((n, c), dtype=bool)
        valid_start = np.zeros(n, dtype=np.int32)
        valid_samples = np.zeros(n, dtype=np.int32)
        for ri, key in enumerate(keys):
            rr = raw_records[subject][key]
            fr = feature_records[subject][key]
            raw_names = list(map(str, _channel_names(rr)))
            feature_names = list(map(str, _channel_names(fr)))
            if set(raw_names) != set(feature_names):
                raise RuntimeError("Raw/descriptor local-channel membership differs")
            raw_sample, feature_sample = rr["sample"], fr["sample"]
            signal = np.asarray(raw_sample["raw_waveform"], dtype=np.float32)
            if signal.shape != (len(raw_names), 15000) or \
                    float(raw_sample["raw_temporal_sfreq"]) != 250.0:
                raise RuntimeError("Raw ictal 250-Hz/60-s shape changed")
            x = np.asarray(feature_sample["window_features"], dtype=np.float32)
            t = np.asarray(feature_sample["window_relative_centers_sec"], dtype=np.float32)
            if x.shape != (len(t), len(feature_names), len(names9)) or len(t) > 59:
                raise RuntimeError("Historical descriptor window shape changed")
            order = [feature_names.index(name) for name in raw_names]
            f = x[:, order][:, :, idx9]
            d = views_exact_a1(f, t)
            local = [lookup[name] for name in raw_names]
            waves[ri, local] = signal
            descriptors[ri, local, :len(t)] = np.transpose(d, (1, 0, 2))
            window_mask[ri, local, :len(t)] = True
            present[ri, local] = True
            valid_start[ri] = int(raw_sample["raw_valid_start_sample"])
            valid_samples[ri] = int(raw_sample["raw_valid_samples"])
            if valid_start[ri] < 0 or valid_samples[ri] <= 0 or \
                    valid_start[ri] + valid_samples[ri] > 15000:
                raise RuntimeError("Invalid observed ictal span")
        partial_records = int(np.sum(valid_samples < 15000))
        partial = dest.with_suffix(".npz.partial")
        with partial.open("wb") as stream:
            np.savez_compressed(stream, waveforms=waves, descriptors=descriptors,
                                window_mask=window_mask, channel_present=present,
                                valid_start=valid_start, valid_samples=valid_samples,
                                labels=labels, channel_names=np.asarray(canonical),
                                patient=subject, sampling_rate_hz=250.0)
        os.replace(partial, dest)
        status = {"protocol_sha256": protocol_sha, "sha256": sha(dest),
                  "bytes": dest.stat().st_size, "records": n,
                  "partial_records": partial_records, "channels": c}
        temp = marker.with_suffix(".json.tmp")
        temp.write_text(json.dumps(status, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temp, marker)
        totals["patients"] += 1
        totals["records"] += n
        totals["partial_records"] += partial_records
        print(json.dumps({"ordinal": ordinal, "records": n,
                          "partial_records": partial_records}), flush=True)
    print(json.dumps({"status": "ICTAL_TRAIN_CACHE_COMPLETE", **totals}), flush=True)


if __name__ == "__main__":
    main()
