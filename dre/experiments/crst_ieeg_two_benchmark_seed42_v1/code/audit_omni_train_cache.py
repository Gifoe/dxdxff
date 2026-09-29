"""Aggregate-only audit of frozen official Omni TRAIN spectral extraction."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    paths = sorted(args.cache.glob("edf_*.npz"))
    patients = set()
    clips = 0
    labeled = 0
    pathological = 0
    center = Counter()
    conflicts = Counter()
    seen = {}
    for path in paths:
        if not path.with_suffix(".json").is_file():
            raise RuntimeError("Private EDF marker absent")
        with np.load(path) as z:
            patient = str(z["patient"])
            patients.add(patient)
            if str(z["validation_role"]) not in {"inner_train", "inner_val"}:
                raise RuntimeError("Official test or unknown role in TRAIN cache")
            y = np.asarray(z["labels"], dtype=np.int8)
            names = [str(n) for n in z["channel_names"]]
            if len(names) != len(y) or len(names) != len(set(names)):
                raise RuntimeError("Official channel schema mismatch")
            for name, label in zip(names, y):
                if label >= 0:
                    key = patient, name
                    if key in seen and seen[key] != label:
                        conflicts[key] += 1
                    seen[key] = int(label)
            labeled += int((y >= 0).sum())
            pathological += int((y == 1).sum())
            clips += int(z["patches"].shape[0])
            center[str(z["dataset"])] += 1
            if not (z["patches"].shape[1:] == (len(y), 59, 64, 8) and
                    z["edges"].shape[1:] == (len(y), len(y), 15)):
                raise RuntimeError("Private spectral/connectivity tensor shape mismatch")
    result = {"pass": len(patients) == 141 and len(paths) == 296 and labeled == 13350,
              "official_train_patients": len(patients), "official_train_edfs": len(paths),
              "official_train_labeled_edf_channels": labeled,
              "official_train_pathological_edf_channels": pathological,
              "spectral_clips": clips, "edfs_by_center": dict(center),
              "cross_edf_patient_channel_label_conflicts": len(conflicts),
              "test_split_accessed": False,
              "no_patient_edf_channel_rows_exported": True}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result), flush=True)
    if not result["pass"]:
        raise RuntimeError("Omni frozen TRAIN cohort/label count not reproduced")


if __name__ == "__main__":
    main()
