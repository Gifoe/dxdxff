"""Private engineering smoke: one epoch on three completed official-train patients."""

import argparse
from pathlib import Path

import pandas as pd

from train_a1_omni import Bank, check_source, fit_stage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    args = parser.parse_args()
    frame = pd.read_csv(args.cohort)
    eligible = frame.loc[(frame["official_split"] == "train") & (frame["valid_channels"] > 0)]
    complete = []
    for patient, rows in eligible.groupby("patient", sort=True):
        if all((args.features / Path(edf).with_suffix(".npz")).is_file() for edf in rows["edf"]):
            complete.append(patient)
    if len(complete) < 3:
        raise RuntimeError("Wait until three train patients have all feature files")
    subset = eligible.loc[eligible["patient"].isin(complete[:3])]
    args.runtime.mkdir(parents=True, exist_ok=True)
    subset_path = args.runtime / "smoke_train_cohort.csv"
    subset.to_csv(subset_path, index=False)
    collate, Model = check_source()
    bank = Bank(subset_path, args.features, None)
    fit_stage("smoke", Model, collate, bank, bank.patients[:2], bank.patients[2:], 1, args.runtime)
    print({"pass": True, "train_patients": 2, "validation_patients": 1})


if __name__ == "__main__":
    main()
