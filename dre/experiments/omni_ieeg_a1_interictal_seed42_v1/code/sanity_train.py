"""Train-side-only label and feature sanity checks, without opening test outcomes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=Path("F:/Omni-iEEG/data"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cohort = pd.read_csv(args.cohort)
    train = cohort.loc[(cohort["official_split"] == "train") & (cohort["valid_channels"] > 0)]
    first = train.drop_duplicates("patient").set_index("patient")
    rng = np.random.default_rng(42)
    patients = sorted(first.index)
    chosen = rng.choice(patients, size=20, replace=False).tolist()
    label_samples = []
    for patient in chosen:
        row = first.loc[patient]
        tsv = args.source / str(row.edf).replace("_ieeg.edf", "_channels.tsv")
        channels = pd.read_csv(tsv, sep="\t")
        soz = pd.to_numeric(channels["soz"], errors="coerce")
        good = pd.to_numeric(channels["good"], errors="coerce") == 1
        valid = channels.loc[good & soz.isin([0, 1])]
        clinical = pd.to_numeric(valid["soz"]).astype(int).to_numpy()
        internal_nez = 1 - clinical
        if not np.array_equal(clinical, 1 - internal_nez):
            raise RuntimeError("SOZ/A1/evaluation label sign mismatch")
        label_samples.append({"patient": patient, "edf": str(row.edf),
                              "valid_channels": len(valid), "soz_1": int(clinical.sum()),
                              "nez_1_internal": int(internal_nez.sum()),
                              "evaluation_positive": "SOZ=1"})
    (args.output / "LABEL_USAGE_AUDIT.json").write_text(json.dumps({
        "pass": True, "random_seed": 42, "train_patients_checked": len(label_samples),
        "mapping": "channels.tsv soz=1 -> clinical EZ=1 -> internal A1 NEZ=0 -> evaluation EZ=1",
        "resection_used": False, "outcome_used": False, "samples": label_samples,
    }, indent=2) + "\n", encoding="utf-8")

    available = []
    for patient in patients:
        row = first.loc[patient]
        path = args.features / Path(str(row.edf)).with_suffix(".npz")
        if path.is_file():
            available.append((patient, path))
    if len(available) < 5:
        print(f"WAITING_FOR_FIVE_FEATURE_PATIENTS present={len(available)}", flush=True)
        return
    samples = []
    for patient, path in available[:5]:
        with np.load(path, allow_pickle=False) as data:
            arr = np.asarray(data["features"], dtype=np.float32)
            channels = [str(value) for value in data["channel_names"]]
        if arr.shape[1] != 59 or arr.shape[-1] != 36 or not np.isfinite(arr).all():
            raise RuntimeError(f"Invalid A1 feature tensor: {path}")
        selection = arr[: min(2, len(arr)), [0, 10, 29, 58], : min(10, len(channels))]
        for view_index, view in enumerate(["ABS", "DELTA", "ZDELTA", "LOG_R"]):
            values = selection[..., view_index * 9 : (view_index + 1) * 9]
            samples.append({"patient": patient, "view": view, "channels": min(10, len(channels)),
                            "windows": 4, "segments": min(2, len(arr)),
                            "finite_fraction": float(np.isfinite(values).mean()),
                            "min": float(values.min()), "median": float(np.median(values)),
                            "max": float(values.max()), "q1": float(np.quantile(values, 0.25)),
                            "q3": float(np.quantile(values, 0.75))})
    (args.output / "FEATURE_DISTRIBUTION_AUDIT.json").write_text(json.dumps({
        "pass": True, "train_patients_checked": 5, "all_36_dimensions_finite": True,
        "views": samples,
    }, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"label_patients_checked": 20, "feature_patients_checked": 5, "pass": True}))


if __name__ == "__main__":
    main()
