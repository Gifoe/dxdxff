#!/usr/bin/env python3
"""Build the leakage-safe full Omni Task-2 patient cohort and fold manifest.

This stage reads only frozen dataset metadata and ``*_channels.tsv`` sidecars.
It writes the actual patient/path manifest only to a private runtime directory;
the public manifest uses irreversible run-local tokens.  No waveform, model,
prediction, or test-fold optimisation is performed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


SEED = 42
REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"


def flag(value: object) -> bool:
    if isinstance(value, str):
        return value.strip().casefold() in {"1", "1.0", "true"}
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False, float_format="%.12g")
    os.replace(temporary, path)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def center_name(value: object) -> str:
    table = {"hup": "HUP", "openieeg": "Open-iEEG", "sourcesink": "SourceSink", "zurich": "Zurich"}
    key = str(value).casefold().replace("-", "")
    if key not in table:
        raise RuntimeError(f"Unexpected eligible center {value!r}")
    return table[key]


def channel_label(outcome: object, channel: pd.Series) -> int:
    """Return unified pathology semantics: 1 SOZ/pathology, 0 normal, -1 unknown."""
    if not flag(channel.good):
        return -1
    # This order exactly preserves the official normal-over-SOZ construction.
    if flag(outcome) and not flag(channel.resection):
        return 0
    if flag(channel.soz):
        return 1
    return -1


def fold_assignment(frame: pd.DataFrame) -> pd.Series:
    """Deterministic greedy five-fold balance of center, labels, and prevalence.

    Small centers make exact multiway stratification impossible.  This objective
    is fixed before any fold model runs and its resulting balance is published.
    """
    work = frame.copy()
    work["tie"] = [int.from_bytes(hashlib.sha256(f"{SEED}|{p}".encode()).digest()[:8], "big")
                   for p in work.patient_name]
    total = {"patients": len(work), "path": int(work.n_pathological_channels.sum()),
             "normal": int(work.n_normal_channels.sum()), "labels": int(work.n_labeled_channels.sum())}
    state = [{"patients": 0, "path": 0, "normal": 0, "labels": 0,
              "centers": defaultdict(int)} for _ in range(5)]
    assigned: dict[str, int] = {}
    # Allocate independently within each center, always preferring the fold
    # with the fewest patients from that center.  This strict first key avoids
    # the pathological prior implementation that placed entire large-center
    # strata in only a subset of folds.  Remaining keys balance overall count
    # and support deterministically.
    for center in sorted(work.center.unique()):
        group = work.loc[work.center == center].sort_values(
            ["n_labeled_channels", "pathological_fraction", "tie"],
            ascending=[False, False, True])
        for row in group.itertuples(index=False):
            options = []
            for fold, current in enumerate(state):
                candidate = {"patients": current["patients"] + 1,
                             "path": current["path"] + int(row.n_pathological_channels),
                             "normal": current["normal"] + int(row.n_normal_channels),
                             "labels": current["labels"] + int(row.n_labeled_channels)}
                score = (current["centers"][center],
                         abs(candidate["patients"] - total["patients"] / 5),
                         abs(candidate["path"] - total["path"] / 5),
                         abs(candidate["normal"] - total["normal"] / 5),
                         abs(candidate["labels"] - total["labels"] / 5), fold)
                options.append((score, fold))
            _, chosen = min(options)
            current = state[chosen]
            current["patients"] += 1
            current["path"] += int(row.n_pathological_channels)
            current["normal"] += int(row.n_normal_channels)
            current["labels"] += int(row.n_labeled_channels)
            current["centers"][center] += 1
            assigned[row.patient_name] = chosen + 1
    return frame.patient_name.map(assigned)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--official-split", type=Path, required=True)
    parser.add_argument("--sidecar-root", type=Path, required=True)
    parser.add_argument("--signal-cache", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.runtime.exists() or args.output.exists():
        raise RuntimeError("Refusing to overwrite an existing cohort run")
    rows = pd.read_csv(args.official_split)
    eligible = rows.loc[(rows.dataset.astype(str) != "Multicenter") & rows.interictal.map(flag) &
                        (pd.to_numeric(rows.frequency, errors="coerce") > 900) &
                        (pd.to_numeric(rows.length, errors="coerce") >= 62)].copy()
    if eligible.empty or set(eligible.split.astype(str)) - {"train", "test"}:
        raise RuntimeError("Official full Task-2 eligibility filter failed")
    observed: dict[str, dict] = {}
    audit_rows: list[dict] = []
    for record in eligible.itertuples(index=False):
        rel = Path(str(record.edf_name))
        if rel.is_absolute() or ".." in rel.parts or rel.suffix.casefold() != ".edf":
            raise RuntimeError(f"Unsafe EDF path {rel}")
        sidecar = args.sidecar_root / Path(str(rel).replace("_ieeg.edf", "_channels.tsv"))
        h5 = args.signal_cache / rel.with_suffix(".edf.h5")
        if not sidecar.is_file() or not h5.is_file():
            raise FileNotFoundError(f"Missing frozen sidecar/cache for {rel}")
        channels = pd.read_csv(sidecar, sep="\t")
        needed = {"name", "good", "resection", "soz"}
        if not needed.issubset(channels.columns) or channels.name.astype(str).duplicated().any():
            raise RuntimeError(f"Invalid channel sidecar {rel}")
        patient, center = str(record.patient_name), center_name(record.dataset)
        info = observed.setdefault(patient, {"center": center, "edfs": [], "labels": {}, "unknown": set(),
                                             "good_occurrences": 0, "unknown_occurrences": 0})
        if info["center"] != center:
            raise RuntimeError(f"Patient appears at multiple centers: {patient}")
        info["edfs"].append(rel.as_posix())
        good_count = labeled_count = unknown_count = 0
        for channel in channels.itertuples(index=False):
            label = channel_label(record.outcome, channel)
            if label < 0 and not flag(channel.good):
                continue
            name = str(channel.name)
            good_count += 1
            info["good_occurrences"] += 1
            if label < 0:
                unknown_count += 1; info["unknown_occurrences"] += 1; info["unknown"].add(name)
                continue
            labeled_count += 1
            previous = info["labels"].setdefault(name, label)
            if previous != label:
                raise RuntimeError(f"STOP_LABEL_CONFLICT: patient={patient}, channel={name}")
        audit_rows.append({"center": center, "official_partition": str(record.split), "eligible_edfs": 1,
                           "good_channel_occurrences": good_count, "labeled_channel_occurrences": labeled_count,
                           "unknown_good_channel_occurrences": unknown_count})
    patient_rows = []
    for patient, value in observed.items():
        labels = value["labels"]
        pathology = sum(label == 1 for label in labels.values())
        normal = sum(label == 0 for label in labels.values())
        category = "both_class" if pathology and normal else "pathological_only" if pathology else "normal_only" if normal else "no_valid_label"
        patient_rows.append({"patient_name": patient, "center": value["center"], "n_eligible_edfs": len(value["edfs"]),
                             "n_pathological_channels": pathology, "n_normal_channels": normal,
                             "n_labeled_channels": pathology + normal,
                             "pathological_fraction": pathology / (pathology + normal) if pathology + normal else np.nan,
                             "n_unique_good_context_channels": len(set(labels) | value["unknown"]),
                             "n_unique_unknown_context_channels": len(value["unknown"] - set(labels)), "category": category})
    patients = pd.DataFrame(patient_rows).sort_values("patient_name").reset_index(drop=True)
    both = patients.loc[patients.category == "both_class"].copy()
    if len(both) < 5:
        raise RuntimeError("STOP_PROTOCOL_INVALID: fewer than five both-class patients")
    both["fold"] = fold_assignment(both)
    if both.patient_name.duplicated().any() or set(both.fold) != {1, 2, 3, 4, 5}:
        raise RuntimeError("STOP_PROTOCOL_INVALID: invalid patient fold assignment")
    args.runtime.mkdir(parents=True); args.output.mkdir(parents=True)
    atomic_csv(args.runtime / "PATIENT_5FOLD_MANIFEST_PRIVATE.csv", both)
    # Public token only: patient/channel observations and source paths remain private.
    public = both.copy(); public["patient_id"] = [f"P{i:03d}" for i in range(1, len(public) + 1)]
    public = public[["patient_id", "center", "fold", "n_eligible_edfs", "n_pathological_channels", "n_normal_channels", "pathological_fraction"]]
    atomic_csv(args.output / "PATIENT_5FOLD_MANIFEST.csv", public)
    categories = ["both_class", "pathological_only", "normal_only", "no_valid_label"]
    support = []
    for center in ["ALL", *sorted(patients.center.unique())]:
        group = patients if center == "ALL" else patients.loc[patients.center == center]
        values = group.category.value_counts().to_dict()
        support.append({"center": center, "n_total_patients": len(group), "n_both_class": int(values.get("both_class", 0)),
                        "n_pathological_only": int(values.get("pathological_only", 0)), "n_normal_only": int(values.get("normal_only", 0)),
                        "n_no_valid_label": int(values.get("no_valid_label", 0)), "n_unknown_context_channels": int(group.n_unique_unknown_context_channels.sum())})
    atomic_csv(args.output / "PATIENT_CLASS_SUPPORT_ALL.csv", pd.DataFrame(support))
    full_audit = pd.DataFrame(audit_rows).groupby(["center", "official_partition"], as_index=False).sum(numeric_only=True)
    atomic_csv(args.output / "FULL_COHORT_LABEL_AUDIT.csv", full_audit)
    balance = both.groupby(["fold", "center"], as_index=False).agg(n_patients=("patient_name", "size"),
        eligible_edfs=("n_eligible_edfs", "sum"), pathological_channels=("n_pathological_channels", "sum"),
        normal_channels=("n_normal_channels", "sum"), labeled_channels=("n_labeled_channels", "sum"))
    balance["pathological_fraction"] = balance.pathological_channels / balance.labeled_channels
    atomic_csv(args.output / "FOLD_BALANCE_AUDIT.csv", balance)
    cohort_digest = sha256(args.runtime / "PATIENT_5FOLD_MANIFEST_PRIVATE.csv")
    status = {"status": "PASS", "dataset_revision": REVISION, "seed": SEED,
              "official_split_sha256": sha256(args.official_split), "n_eligible_edfs": len(eligible),
              "n_total_patients": len(patients), "n_both_class": len(both),
              "private_manifest_sha256": cohort_digest, "all_retained_patients_both_class": bool((both.n_pathological_channels.gt(0) & both.n_normal_channels.gt(0)).all()),
              "label_conflicts": 0, "historical_test_only_subset_used": False,
              "raw_waveforms_or_predictions_generated": False}
    atomic_json(args.output / "BOTH_CLASS_COHORT_AUDIT.json", status)
    atomic_json(args.output / "PROTOCOL_LOCK.json", {"experiment": "omni_patient5fold_a1plugin_seed42_v1", "seed": SEED,
        "task": "full Omni Task2 patient-level pathological-channel localization", "eligibility": "interictal; frequency>900; length>=62; dataset!=Multicenter; good==1",
        "label_rule": "normal iff outcome==1 and resection==0; else pathological iff soz==1; otherwise unknown=-1",
        "patient_manifest_private_sha256": cohort_digest, "folds": 5, "epochs": 30,
        "optimizer": {"name": "AdamW", "learning_rate": 1e-4, "weight_decay": 1e-3, "gradient_clip": 1.0},
        "test_fold_labels_used_for_optimization": False, "pretrained_omni_labelled_checkpoint_allowed": False})
    print(json.dumps(status, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
