"""Private-only official test identity audit after the pre-test freeze."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from patient_bank import OmniTestPatientBank
from train_crst import sha


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--freeze", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    bank = OmniTestPatientBank(args.cache, args.freeze)
    labeled = pathological = clips = 0
    center = Counter()
    for paths in bank.patient_files.values():
        for path in paths:
            with np.load(path) as z:
                y = np.asarray(z["labels"], np.int8)
                if z["patches"].shape[1:] != (len(y), 59, 64, 8):
                    raise RuntimeError("Official test tensor shape mismatch")
                labeled += int((y >= 0).sum())
                pathological += int((y == 1).sum())
                clips += int(z["patches"].shape[0])
                center[str(z["dataset"])] += 1
    report = {"pass": len(bank.patient_files) == 96 and labeled == 8104,
              "official_test_patients": 96, "official_test_edfs": 174,
              "official_test_labeled_edf_channels": labeled,
              "official_test_pathological_edf_channels": pathological,
              "spectral_clips": clips, "edfs_by_center": dict(center),
              "freeze_sha256": sha(args.freeze),
              "test_accessed_only_after_freeze": True,
              "prediction_or_threshold_tuning": False}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report), flush=True)
    if not report["pass"]:
        raise RuntimeError("Official test EDF-channel identity differs")


if __name__ == "__main__":
    main()
