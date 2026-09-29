"""Aggregate training-weight audit without patient/channel-level publication."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from patient_bank import IctalBank, OmniTrainBank


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=("ictal", "omni"), required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--ictal-manifest", type=Path)
    p.add_argument("--omni-inner-split", type=Path)
    p.add_argument("--omni-official-split", type=Path)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.benchmark == "ictal":
        bank = IctalBank(args.cache, args.ictal_manifest)
        groups = {f"fold{fold}": bank.folds[fold]["fit"] for fold in range(1, 6)}
    else:
        bank = OmniTrainBank(args.cache, args.omni_inner_split,
                             args.omni_official_split)
        groups = {"inner_train": bank.patients("inner_train")}
    folds = {}
    for key, patients in groups.items():
        counts = [bank.labeled_observations(patient) for patient in patients]
        if not counts or min(counts) <= 0:
            raise RuntimeError("Supervised patient with zero observations")
        folds[key] = {"fit_patients": len(patients),
                      "min_labeled_observations_per_patient": min(counts),
                      "max_labeled_observations_per_patient": max(counts),
                      "patient_optimizer_steps_per_epoch": len(patients)}
    audit = {"status": "PASS", "benchmark": args.benchmark,
             "folds": folds, "loss_rule": "weighted_BCE_sum_within_patient / labeled_channel_record_count",
             "positive_weight": 2.0, "negative_weight": 1.0,
             "optimizer_steps_per_patient_per_epoch": 1,
             "patients_given_equal_epoch_step_weight": True,
             "test_accessed": False}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
