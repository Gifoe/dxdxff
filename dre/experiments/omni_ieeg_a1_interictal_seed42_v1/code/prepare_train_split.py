"""Freeze an official-train-only deterministic patient validation split."""

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
    cohort = pd.read_csv(args.cohort)
    train = cohort.loc[(cohort["official_split"] == "train") & (cohort["valid_channels"] > 0)]
    patients = train.groupby("patient", sort=True).agg(
        dataset=("dataset", "first"), edfs=("edf", "size"),
        valid_channels=("valid_channels", "max"), soz_channels=("soz_channels", "max"),
    ).reset_index()
    patients["has_soz"] = (patients["soz_channels"] > 0).astype(int)
    stratum = patients["dataset"].astype(str) + "|" + patients["has_soz"].astype(str)
    if stratum.value_counts().min() < 2:
        stratum = patients["dataset"].astype(str)
        stratification = "dataset_only_due_to_rare_dataset_x_soz_stratum"
    else:
        stratification = "dataset_x_soz_presence"
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    fit, val = next(splitter.split(patients, stratum))
    patients["role"] = "inner_train"
    patients.loc[val, "role"] = "inner_val"
    if set(patients["patient"]) & set(cohort.loc[cohort["official_split"] == "test", "patient"]):
        raise RuntimeError("Official test patient leaked into train-side validation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    patients.to_csv(args.output, index=False)
    summary = {"seed": 42, "stratification": stratification,
               "inner_train_patients": len(fit), "inner_val_patients": len(val),
               "dataset_role_counts": patients.groupby(["dataset", "role"]).size().to_dict()}
    (args.output.parent / "TRAIN_VAL_SPLIT_AUDIT.json").write_text(
        json.dumps({**summary, "dataset_role_counts": {f"{k[0]}|{k[1]}": v for k, v in summary["dataset_role_counts"].items()}},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"stratification": stratification, "inner_train": len(fit), "inner_val": len(val)}))


if __name__ == "__main__":
    main()
