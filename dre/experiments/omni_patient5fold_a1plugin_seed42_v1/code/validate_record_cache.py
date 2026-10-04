#!/usr/bin/env python3
"""Validate the private full-record cache before any fold may train."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from modeling import sha256


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--private-manifest", type=Path, required=True)
    parser.add_argument("--cohort-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.cohort_protocol.read_text(encoding="utf-8"))
    if sha256(args.private_manifest) != protocol["patient_manifest_private_sha256"]:
        raise RuntimeError("Private manifest hash mismatch")
    manifest = pd.read_csv(args.private_manifest)
    expected = int(manifest.n_eligible_edfs.sum())
    markers = sorted(args.records.glob("SHARD_*_OF_04.json"))
    if len(markers) != 4 or any(json.loads(item.read_text(encoding="utf-8"))["status"] != "COMPLETE" for item in markers):
        raise RuntimeError("Incomplete record extraction shards")
    files = sorted(args.records.glob("edf_*.npz"))
    if len(files) != expected:
        raise RuntimeError(f"Expected {expected} EDF records, found {len(files)}")
    allowed = set(manifest.patient_name.astype(str))
    per_patient_edfs: dict[str, int] = defaultdict(int)
    labels: dict[tuple[str, str], int] = {}
    unknown = 0
    seen_edfs = set()
    for path in files:
        marker = path.with_suffix(".json")
        if not marker.is_file():
            raise RuntimeError("Private record is missing integrity marker")
        record_marker = json.loads(marker.read_text(encoding="utf-8"))
        if record_marker["sha256"] != sha256(path):
            raise RuntimeError("Private record content hash mismatch")
        with np.load(path, allow_pickle=False) as data:
            patient = str(data["patient_name"].item()); edf = str(data["edf_name"].item())
            names = [str(value) for value in data["channel_names"].tolist()]
            y = data["labels"].astype(np.int8)
            waves = data["waveforms"]
            if waves.shape != (len(names), 60000) or y.shape != (len(names),) or len(set(names)) != len(names):
                raise RuntimeError("Invalid record shape or channel identity")
            if patient not in allowed or edf in seen_edfs:
                raise RuntimeError("Cache patient/EDF membership mismatch")
            seen_edfs.add(edf); per_patient_edfs[patient] += 1
            for channel, target in zip(names, y.tolist()):
                if target < 0:
                    unknown += 1; continue
                key = (patient, channel)
                prior = labels.setdefault(key, int(target))
                if prior != int(target):
                    raise RuntimeError("STOP_LABEL_CONFLICT in private cache")
    expected_edfs = manifest.set_index("patient_name").n_eligible_edfs.astype(int).to_dict()
    if {str(key): int(value) for key, value in per_patient_edfs.items()} != {str(key): int(value) for key, value in expected_edfs.items()}:
        raise RuntimeError("Patient EDF count differs from manifest")
    path_count = sum(value == 1 for value in labels.values())
    normal_count = sum(value == 0 for value in labels.values())
    if path_count != int(manifest.n_pathological_channels.sum()) or normal_count != int(manifest.n_normal_channels.sum()):
        raise RuntimeError("Unique label support differs from frozen manifest")
    atomic_json(args.output, {"status": "PASS", "records": len(files), "patients": len(per_patient_edfs),
                              "unique_pathological_patient_channels": path_count,
                              "unique_normal_patient_channels": normal_count,
                              "unknown_good_channel_occurrences": unknown,
                              "partial_files": len(list(args.records.glob("*.partial"))),
                              "test_or_model_predictions_generated": False,
                              "cohort_protocol_sha256": sha256(args.cohort_protocol)})


if __name__ == "__main__":
    main()
