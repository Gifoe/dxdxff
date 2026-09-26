"""Convert the frozen 80-person NeuroEZ cache into PaReSet's 9-descriptor schema.

This adapter copies neither raw EEG nor surgery outcomes. It selects nine
named descriptors from the server's 28-column window cache, preserving its
actual 2-second windows and onset-relative centers. Four-view expansion and
fit-only normalization are performed by the training runner, per fold.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


DESCRIPTORS = (
    "log_bp_delta", "log_bp_theta", "log_bp_beta", "log_bp_low_gamma",
    "log_bp_high_gamma", "rms", "variance", "line_length_per_sec",
    "spectral_entropy",
)
CENTERS = {"hup", "lzu", "multicenter", "pediatric"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sample_of(record: dict) -> dict:
    sample = record.get("sample")
    return sample if isinstance(sample, dict) else record


def read_cohort(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    ids = [str(row["subject_id"]) for row in rows]
    if len(ids) != 80 or len(set(ids)) != 80:
        raise ValueError("Formal cohort must contain exactly 80 unique subjects")
    return ids


def build(cache: dict, cohort_ids: list[str], feature_names: list[str]) -> tuple[list[dict], dict]:
    if "ez-positive" not in str(cache.get("label_semantics", "")).lower():
        raise ValueError("Cache does not explicitly declare EZ-positive labels")
    if list(map(str, cache.get("window_feature_names", []))) != feature_names:
        raise ValueError("Feature-name sidecar and cache payload disagree")
    indices = [feature_names.index(name) for name in DESCRIPTORS]
    index = cache["patient_index"]
    records = cache["run_records"]
    if not isinstance(index, dict) or not isinstance(records, list):
        raise TypeError("Cache requires patient_index dict and run_records list")
    if set(cohort_ids) - set(index):
        raise ValueError("Frozen cohort has patients absent from cache")
    grouped = defaultdict(list)
    allowed = set(cohort_ids)
    for record in records:
        patient_id = str(record.get("subject_id"))
        if patient_id in allowed:
            grouped[patient_id].append(record)
    if set(grouped) != allowed:
        raise ValueError("Frozen cohort has patients without run records")

    result = []
    run_count = 0
    centers = Counter()
    absent_canonical_channels = 0
    run_vs_patient_label_mismatches = 0
    patients_with_run_label_mismatch = set()
    maximum = {"seizures": 0, "windows": 0, "channels": 0}
    for patient_id in cohort_ids:
        meta = index[patient_id]
        canonical = tuple(map(str, meta["canonical_channels"]))
        if len(canonical) != len(set(canonical)):
            raise ValueError(f"Duplicate canonical channels: {patient_id}")
        channel_to_idx = {name: i for i, name in enumerate(canonical)}
        labels_ez = np.asarray(meta["labels"], dtype=np.float32)
        label_mask = np.asarray(meta.get("label_mask", np.ones(len(canonical))), dtype=bool)
        if labels_ez.shape != (len(canonical),) or not np.all(np.isin(labels_ez, [0, 1])):
            raise ValueError(f"Invalid EZ labels: {patient_id}")
        if label_mask.shape != labels_ez.shape or not label_mask.all():
            raise ValueError(f"Masked labels require a separate explicit protocol: {patient_id}")
        center = patient_id.split(":", 1)[0]
        if center not in CENTERS:
            raise ValueError(f"Unknown center prefix: {patient_id}")
        runs = sorted(grouped[patient_id], key=lambda r: (str(r.get("run_id")), str(sample_of(r).get("sample_id"))))
        run_keys = [(str(r.get("run_id")), str(sample_of(r).get("sample_id"))) for r in runs]
        if len(set(run_keys)) != len(run_keys):
            raise ValueError(f"Repeated run key: {patient_id}")
        wmax = max(len(np.asarray(sample_of(record)["window_relative_centers_sec"])) for record in runs)
        s, c = len(runs), len(canonical)
        x = np.zeros((s, wmax, c, len(DESCRIPTORS)), dtype=np.float32)
        times = np.zeros((s, wmax), dtype=np.float32)
        valid = np.zeros((s, wmax, c), dtype=bool)
        for seizure, record in enumerate(runs):
            sample = sample_of(record)
            if list(map(str, sample.get("window_feature_names", []))) != feature_names:
                raise ValueError(f"Run feature names differ: {patient_id}")
            f = np.asarray(sample["window_features"], dtype=np.float32)
            t = np.asarray(sample["window_relative_centers_sec"], dtype=np.float32)
            local = list(map(str, record["channel_names_norm"]))
            if f.ndim != 3 or f.shape != (len(t), len(local), len(feature_names)):
                raise ValueError(f"Feature/time/channel shape mismatch: {patient_id}")
            if len(local) != len(set(local)) or not all(name in channel_to_idx for name in local):
                raise ValueError(f"Local channels cannot be aligned: {patient_id}")
            if not np.isfinite(f).all() or not np.isfinite(t).all() or not np.all(np.diff(t) > 0):
                raise ValueError(f"Nonfinite feature or invalid times: {patient_id}")
            if not np.any(t < 0) or not np.any(t >= 0):
                raise ValueError(f"Missing pre/post onset phase: {patient_id}")
            local_labels = np.asarray(record["labels"], dtype=np.float32)
            target_indices = np.asarray([channel_to_idx[name] for name in local], dtype=np.int64)
            if local_labels.shape != (len(local),) or not np.all(np.isin(local_labels, [0, 1])):
                raise ValueError(f"Invalid run-local label vector: {patient_id}")
            # The formal server loader trains/evaluates against patient_index.labels.
            # Some run-local labels are stale; count them without changing the
            # canonical label source or discarding their feature windows.
            mismatch = int(np.count_nonzero(local_labels != labels_ez[target_indices]))
            run_vs_patient_label_mismatches += mismatch
            if mismatch:
                patients_with_run_label_mismatch.add(patient_id)
            w = len(t)
            x[seizure, :w, target_indices, :] = np.transpose(f[:, :, indices], (1, 0, 2))
            # Advanced NumPy indexing places the channel axis first above.
            times[seizure, :w] = t
            valid[seizure, :w, target_indices] = True
        if not valid.any():
            raise ValueError(f"Patient has no usable channels: {patient_id}")
        absent_canonical_channels += int((~valid.any(axis=(0, 1))).sum())
        result.append({
            "patient_id": patient_id, "center": center, "descriptors": x,
            "window_times": times, "valid": valid,
            "channel_names": canonical,
            "label_nez": (1 - labels_ez).astype(np.int64),
        })
        centers[center] += 1
        run_count += s
        maximum["seizures"] = max(maximum["seizures"], s)
        maximum["windows"] = max(maximum["windows"], wmax)
        maximum["channels"] = max(maximum["channels"], c)
    audit = {
        "status": "ADAPTED_80_PERSON_SCHEMA",
        "patients": len(result), "runs": run_count,
        "centers": dict(centers), "max_shape": maximum,
        "absent_canonical_channels_masked": absent_canonical_channels,
        "run_vs_patient_label_mismatches": run_vs_patient_label_mismatches,
        "patients_with_run_label_mismatch": sorted(patients_with_run_label_mismatch),
        "descriptor_names": list(DESCRIPTORS), "descriptor_indices": indices,
        "label_source": "patient_index.labels EZ-positive; converted once to NEZ-positive",
        "outcome_fields_copied": False,
        "raw_eeg_copied": False,
    }
    return result, audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--feature-names", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.audit.exists():
        raise FileExistsError("Adapter outputs already exist; use a new destination")
    with args.feature_names.open(newline="", encoding="utf-8-sig") as handle:
        names = [row["feature_name"] for row in csv.DictReader(handle)]
    cohort = read_cohort(args.cohort)
    with args.cache.open("rb") as handle:
        payload = pickle.load(handle)
    patients, audit = build(payload, cohort, names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".partial")
    with temporary.open("wb") as handle:
        pickle.dump(patients, handle, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(args.output)
    audit.update({
        "source_cache": str(args.cache), "source_cache_sha256": sha256(args.cache),
        "cohort_manifest": str(args.cohort), "cohort_sha256": sha256(args.cohort),
        "feature_names_sha256": sha256(args.feature_names),
        "output": str(args.output), "output_sha256": sha256(args.output),
        "output_bytes": args.output.stat().st_size,
    })
    args.audit.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
