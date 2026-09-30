"""Build a private raw-spectral Ictal bank with explicit real window centers."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import pickle
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from scm_core import BIN_EDGES_HZ, STATE_NAMES, scm_matrices, state_membership, visible_bins, window_spectra


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def sample(record: dict) -> dict:
    value = record.get("sample")
    return value if isinstance(value, dict) else record


def record_key(record: dict) -> tuple[str, str]:
    current = sample(record)
    return str(record.get("run_id")), str(current.get("sample_id"))


def channel_names(record: dict) -> list[str]:
    for owner in (record, sample(record)):
        for key in ("channel_names_norm", "channel_names"):
            if key in owner:
                return list(map(str, owner[key]))
    raise RuntimeError("record channel names absent")


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def load_sources(feature_path: Path, raw_path: Path, manifest_path: Path, protocol_path: Path):
    lock = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = ((feature_path, "ictal_feature_cache_sha256"),
                (raw_path, "ictal_raw_cache_sha256"),
                (manifest_path, "ictal_fold_manifest_sha256"))
    for path, key in expected:
        if digest(path) != lock[key]:
            raise RuntimeError(f"frozen source hash mismatch: {key}")
    with manifest_path.open(newline="", encoding="utf-8-sig") as stream:
        patients = sorted({str(row["subject_id"]) for row in csv.DictReader(stream)})
    if len(patients) != 80:
        raise RuntimeError("historical cohort is not 80 patients")
    with feature_path.open("rb") as stream:
        feature = pickle.load(stream)
    with raw_path.open("rb") as stream:
        raw = pickle.load(stream)
    return lock, patients, feature, raw


def grouped_records(payload: dict, patients: set[str]) -> dict[str, dict]:
    grouped = defaultdict(dict)
    for record in payload["run_records"]:
        patient = str(record["subject_id"])
        if patient not in patients:
            continue
        key = record_key(record)
        if key in grouped[patient]:
            raise RuntimeError("duplicate record key")
        grouped[patient][key] = record
    return grouped


def build_patient(patient: str, feature: dict, feature_records: dict, raw_records: dict) -> tuple[dict, dict]:
    if set(feature_records) != set(raw_records):
        raise RuntimeError("raw/feature record alignment differs")
    canonical = list(map(str, feature["patient_index"][patient]["canonical_channels"]))
    labels = np.asarray(feature["patient_index"][patient]["labels"], dtype=np.int8)
    if labels.shape != (len(canonical),) or not np.isin(labels, (0, 1)).all() or len(canonical) != len(set(canonical)):
        raise RuntimeError("invalid canonical channels/labels")
    lookup = {name: index for index, name in enumerate(canonical)}
    keys = sorted(raw_records)
    maximum_windows = max(len(np.asarray(sample(feature_records[key])["window_relative_centers_sec"])) for key in keys)
    r, c = len(keys), len(canonical)
    spectra = np.zeros((r, c, maximum_windows, 16), dtype=np.float32)
    window_mask = np.zeros((r, c, maximum_windows), dtype=bool)
    centers = np.zeros((r, maximum_windows), dtype=np.float32)
    center_mask = np.zeros((r, maximum_windows), dtype=bool)
    present = np.zeros((r, c), dtype=bool)
    state_patterns, fallback_records = Counter(), 0
    for record_index, key in enumerate(keys):
        feature_record, raw_record = feature_records[key], raw_records[key]
        feature_names, raw_names = channel_names(feature_record), channel_names(raw_record)
        if set(feature_names) != set(raw_names) or len(raw_names) != len(set(raw_names)):
            raise RuntimeError("raw/feature local channel membership differs")
        fsample, rsample = sample(feature_record), sample(raw_record)
        time_centers = np.asarray(fsample["window_relative_centers_sec"], dtype=np.float64)
        if not np.array_equal(time_centers, np.asarray(rsample["window_relative_centers_sec"], dtype=np.float64)):
            raise RuntimeError("raw/feature real window centers differ")
        waveform = np.asarray(rsample["raw_waveform"], dtype=np.float32)
        fs = float(rsample["raw_temporal_sfreq"])
        if fs != 250.0 or waveform.shape != (len(raw_names), 15000):
            raise RuntimeError("historical raw shape or sampling rate changed")
        valid_start = int(rsample["raw_valid_start_sample"])
        valid_samples = int(rsample["raw_valid_samples"])
        onset = valid_start + int(round((float(rsample["seizure_onset_sec"]) -
                                         float(rsample["start_sec"])) * fs))
        current, valid_windows = window_spectra(waveform, fs, time_centers, onset,
                                                valid_start, valid_samples)
        if not valid_windows.all():
            raise RuntimeError("source-declared center became invalid")
        reorder = [raw_names.index(name) for name in feature_names]
        current = current[reorder]
        local = np.asarray([lookup[name] for name in feature_names], dtype=np.int64)
        w = len(time_centers)
        spectra[record_index, local, :w] = current
        window_mask[record_index, local, :w] = True
        centers[record_index, :w] = time_centers
        center_mask[record_index, :w] = True
        present[record_index, local] = True
        membership = state_membership(time_centers)
        state_patterns[tuple(int(np.sum(membership == state)) for state in range(6))] += 1
        fallback_records += int(valid_samples < 15000)
    if not np.all(present.any(axis=0)):
        raise RuntimeError("canonical channel lacks raw signal across all seizures")
    value = {"spectra": spectra, "window_mask": window_mask, "centers": centers,
             "center_mask": center_mask, "channel_present": present,
             "labels": labels, "channel_names": np.asarray(canonical),
             "patient": np.asarray(patient), "sampling_rate_hz": np.asarray(250.0)}
    audit = {"patient_private": patient, "records": r, "channels": c,
             "windows": int(center_mask.sum()), "partial_records": fallback_records,
             "state_patterns": {str(key): count for key, count in state_patterns.items()}}
    return value, audit


def write_patient(output: Path, patient: str, value: dict, audit: dict, protocol_sha: str) -> None:
    token = hashlib.sha256(patient.encode()).hexdigest()[:20]
    destination = output / f"patient_{token}.npz"
    marker = destination.with_suffix(".json")
    if destination.exists() and marker.exists():
        old = json.loads(marker.read_text(encoding="utf-8"))
        if old.get("protocol_sha256") != protocol_sha or old.get("sha256") != digest(destination):
            raise RuntimeError("private cache resume mismatch")
        return
    if destination.exists() or marker.exists():
        raise RuntimeError("partial private patient cache")
    temporary = destination.with_suffix(".npz.partial")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **value)
    os.replace(temporary, destination)
    atomic_json(marker, {**audit, "protocol_sha256": protocol_sha,
                         "sha256": digest(destination), "bytes": destination.stat().st_size})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("speed-gate", "prepare-all"))
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-seconds", type=float, default=30.0)
    args = parser.parse_args()
    started = time.perf_counter()
    lock, patients, feature, raw = load_sources(args.feature_cache, args.raw_cache,
                                                args.manifest, args.protocol)
    source_load_seconds = time.perf_counter() - started
    feature_group = grouped_records(feature, set(patients))
    raw_group = grouped_records(raw, set(patients))
    if set(feature_group) != set(patients) or set(raw_group) != set(patients) or \
            sum(map(len, raw_group.values())) != 256:
        raise RuntimeError("historical 80-patient/256-seizure scope changed")
    protocol_sha = digest(args.protocol)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "speed-gate":
        patient = max(patients, key=lambda p: len(feature["patient_index"][p]["canonical_channels"]) * len(raw_group[p]))
        then = time.perf_counter()
        value, audit = build_patient(patient, feature, feature_group[patient], raw_group[patient])
        # Exercise matrix construction on the real record without fitting data-dependent values.
        record = 0
        centers = value["centers"][record, value["center_mask"][record]]
        membership = state_membership(centers)
        states = np.zeros((len(value["labels"]), 6, 16), dtype=np.float32)
        state_valid = np.zeros((len(value["labels"]), 6), dtype=bool)
        for state in range(6):
            use = membership == state
            for channel in np.flatnonzero(value["channel_present"][record]):
                selected = use & value["window_mask"][record, channel, :len(centers)]
                if selected.any():
                    states[channel, state] = np.median(value["spectra"][record, channel, :len(centers)][selected], axis=0)
                    state_valid[channel, state] = True
        scm_matrices(states[value["channel_present"][record]], state_valid[value["channel_present"][record]])
        extraction_seconds = time.perf_counter() - then
        report = {"status": "PASS" if extraction_seconds <= args.max_seconds else "FAIL",
                  "patient_private_hash": hashlib.sha256(patient.encode()).hexdigest(),
                  "records": audit["records"], "channels": audit["channels"],
                  "source_load_seconds": source_load_seconds,
                  "real_patient_extract_and_matrix_seconds": extraction_seconds,
                  "maximum_seconds": args.max_seconds, "test_accessed": False}
        atomic_json(args.output / "SPEED_GATE.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if report["status"] != "PASS":
            raise SystemExit(2)
        return
    summaries = []
    for ordinal, patient in enumerate(patients, 1):
        token = hashlib.sha256(patient.encode()).hexdigest()[:20]
        destination = args.output / f"patient_{token}.npz"
        marker = destination.with_suffix(".json")
        if destination.exists() and marker.exists():
            old = json.loads(marker.read_text(encoding="utf-8"))
            if old.get("protocol_sha256") != protocol_sha or old.get("sha256") != digest(destination):
                raise RuntimeError("private cache resume mismatch")
            summaries.append(old)
            print(f"SCM cache {ordinal}/80 reused", flush=True)
            continue
        value, audit = build_patient(patient, feature, feature_group[patient], raw_group[patient])
        write_patient(args.output, patient, value, audit, protocol_sha)
        summaries.append(audit)
        print(f"SCM cache {ordinal}/80 records={audit['records']} channels={audit['channels']}", flush=True)
    state_counts = Counter()
    for item in summaries:
        for key, count in item["state_patterns"].items():
            state_counts[key] += int(count)
    report = {"status": "SCM_PRIVATE_CACHE_COMPLETE", "patients": 80, "records": 256,
              "patient_channels": int(sum(item["channels"] for item in summaries)),
              "partial_records": int(sum(item["partial_records"] for item in summaries)),
              "state_patterns": dict(state_counts), "source_load_seconds": source_load_seconds,
              "elapsed_seconds": time.perf_counter() - started,
              "frequency_visible": visible_bins(250.0).astype(int).tolist(),
              "bin_edges_hz": BIN_EDGES_HZ.tolist(), "state_names": list(STATE_NAMES),
              "outcome_fields_copied": False, "outer_test_accessed": False}
    atomic_json(args.output / "CACHE_STATUS.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
