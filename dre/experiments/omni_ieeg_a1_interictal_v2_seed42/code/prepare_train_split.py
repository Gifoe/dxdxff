"""Freeze official-train-only patient split under the v2 label definition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.cohort)
    eligible = frame.loc[(frame["official_split"] == "train") &
                         (frame["official_labeled_channels"] > 0)]
    patient = eligible.groupby("patient", sort=True).agg(
        dataset=("dataset", "first"), edfs=("edf", "size"),
        pathological_channels=("pathological_channels", "sum"),
        normal_channels=("normal_channels", "sum"),
    ).reset_index()
    if len(patient) != 141:
        raise RuntimeError("Expected 141 v2 eligible official-train patients")
    patient["has_pathology"] = patient["pathological_channels"].gt(0)
    stratum = patient["dataset"].astype(str) + "|" + patient["has_pathology"].astype(str)
    if stratum.value_counts().min() < 2:
        raise RuntimeError("Dataset × pathology-presence stratification no longer feasible")
    fit, val = next(StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
                    .split(patient, stratum))
    patient["role"] = "inner_train"
    patient.loc[val, "role"] = "inner_val"
    if len(fit) != 112 or len(val) != 29:
        raise RuntimeError("Unexpected v2 inner patient counts")
    if set(patient["patient"]) & set(frame.loc[frame["official_split"] == "test", "patient"]):
        raise RuntimeError("Official test patient entered inner train/validation")
    if not (patient.loc[val, "has_pathology"]).any():
        raise RuntimeError("No estimable pathology patient in validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    patient.to_csv(args.output, index=False)
    counts = patient.groupby(["dataset", "has_pathology", "role"]).size()
    audit = {"seed": 42, "stratification": "dataset_x_pathology_presence",
             "inner_train_patients": len(fit), "inner_val_patients": len(val),
             "inner_val_estimable_patients": int(patient.loc[val, "has_pathology"].sum()),
             "stratum_role_counts": {"|".join(map(str, k)): int(v) for k, v in counts.items()}}
    (args.output.parent / "TRAIN_VAL_SPLIT_AUDIT.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(audit))


if __name__ == "__main__":
    main()
