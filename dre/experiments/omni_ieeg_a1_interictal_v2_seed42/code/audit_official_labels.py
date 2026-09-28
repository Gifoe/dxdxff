"""Metadata-only audit of exact Omni evaluation-channel ground truth.

This reproduces the *order* of evaluation_channel.py lines 51-60. In
particular, successful-outcome unresected channels are normal even when SOZ=1.
No model scores or EEG samples are read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import pandas as pd


REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"
SPLIT_SHA256 = "e329bab57037d649a1995074a7e5ee31c7c002f00ea6f0ee54bd48d67be1dc42"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def one(value) -> bool:
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return False


def zero(value) -> bool:
    try:
        return float(value) == 0.0
    except (TypeError, ValueError):
        return False


def official_pathology_label(outcome, resection, soz, good) -> int | None:
    """Clinical 1=pathological, inverse of official ground_truth 0.

    Exact branch priority from Omni evaluation_channel.py:57-60.
    """
    if not one(good):
        return None
    if one(outcome) and zero(resection):
        return 0
    if one(soz):
        return 1
    return None


def run(source: Path, cache: Path, output: Path):
    split_path = source / "derivatives/datasplit/final_split.csv"
    if sha256(split_path) != SPLIT_SHA256:
        raise RuntimeError("Frozen official split changed")
    validation = json.loads((cache / "CACHE_VALIDATION.json").read_text(encoding="utf-8"))
    migration = json.loads((cache / "CACHE_MIGRATION.json").read_text(encoding="utf-8"))
    if not validation.get("pass") or not migration.get("pass"):
        raise RuntimeError("Validated native signal cache absent")
    if validation.get("revision") != REVISION or migration.get("revision") != REVISION:
        raise RuntimeError("Dataset revision mismatch")
    split = pd.read_csv(split_path)
    cohort = split.loc[
        split["split"].isin(["train", "test"])
        & (split["dataset"] != "Multicenter")
        & (pd.to_numeric(split["frequency"], errors="coerce") > 900)
        & (split["interictal"] == True)  # noqa: E712, reproduces official filter
        & (pd.to_numeric(split["length"], errors="coerce") >= 62)
    ].copy()
    if cohort["edf_name"].duplicated().any():
        raise RuntimeError("Duplicate official EDF")
    if (cohort.groupby("patient_name")["split"].nunique() != 1).any():
        raise RuntimeError("Official train/test patient overlap")
    records = []
    label_sets = defaultdict(set)
    excluded_counts = Counter()
    conflict_counts = Counter()
    for row in cohort.itertuples(index=False):
        relative = Path(str(row.edf_name))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
            raise RuntimeError("Unsafe official EDF path")
        h5_path = cache / relative.with_suffix(".edf.h5")
        sidecar = source / Path(str(relative).replace("_ieeg.edf", "_channels.tsv"))
        if not h5_path.is_file() or not sidecar.is_file():
            raise RuntimeError(f"Missing frozen EDF equivalent or channels sidecar: {relative}")
        with h5py.File(h5_path, "r") as h5:
            if str(h5.attrs["dataset_revision"]) != REVISION:
                raise RuntimeError("EDF-equivalent cache revision changed")
            meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
            signal_names = {str(h["label"]) for h in meta["signal_headers"]}
            duration = float(meta["file_duration_seconds"])
        channels = pd.read_csv(sidecar, sep="\t")
        if not {"name", "good", "soz", "resection"} <= set(channels):
            raise RuntimeError(f"Missing official label fields: {sidecar}")
        channels = channels.loc[channels["name"].astype(str).isin(signal_names)].copy()
        if channels["name"].duplicated().any():
            raise RuntimeError(f"Duplicate channel in official sidecar: {sidecar}")
        good = channels.loc[channels["good"].map(one)]
        labels = []
        conflicts = 0
        for channel in good.itertuples(index=False):
            label = official_pathology_label(row.outcome, channel.resection, channel.soz, channel.good)
            if one(row.outcome) and zero(channel.resection) and one(channel.soz):
                conflicts += 1
                conflict_counts[str(row.split)] += 1
            if label is None:
                excluded_counts[str(row.split)] += 1
            else:
                labels.append(label)
                label_sets[(str(row.patient_name), str(channel.name))].add(label)
        records.append({
            "dataset": str(row.dataset), "patient": str(row.patient_name),
            "official_split": str(row.split), "edf": relative.as_posix(),
            "outcome": row.outcome, "interictal": bool(row.interictal),
            "original_frequency": float(row.frequency), "duration_sec": duration,
            "good_signal_channels": len(good), "official_labeled_channels": len(labels),
            "pathological_channels": int(sum(labels)),
            "normal_channels": int(len(labels) - sum(labels)),
            "unknown_good_channels": int(len(good) - len(labels)),
            "normal_precedes_soz_conflicts": conflicts,
            "uniform_60s_segments": max(0, int((duration - 2) // 60)),
        })
    frame = pd.DataFrame(records)
    inconsistency = sorted(f"{p}|{c}" for (p, c), values in label_sets.items() if len(values) > 1)
    if inconsistency:
        raise RuntimeError(f"Patient-channel official label inconsistency: {len(inconsistency)}")
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "OFFICIAL_COHORT_AUDIT.csv", index=False)
    eligible = frame.loc[frame["official_labeled_channels"] > 0]
    unique_by_split = {part: {k: next(iter(v)) for k, v in label_sets.items()
                              if k[0] in set(eligible.loc[eligible["official_split"] == part, "patient"])}
                       for part in ("train", "test")}
    summary = {
        "official_rule": "good=1; if outcome=1 and resection=0 then normal; elif soz=1 then pathological; else excluded",
        "official_ground_truth_mapping": "normal=1, pathological=0; exported clinical pathological=1",
        "omni_code_revision": "57c22a75a59b5c3a98006806ad42000f6a3fa5b6",
        "dataset_revision": REVISION, "official_split_sha256": SPLIT_SHA256,
        "filtered_edfs": frame.groupby("official_split").size().to_dict(),
        "filtered_patients": frame.groupby("official_split")["patient"].nunique().to_dict(),
        "eligible_edfs": eligible.groupby("official_split").size().to_dict(),
        "eligible_patients": eligible.groupby("official_split")["patient"].nunique().to_dict(),
        "good_channel_records": frame.groupby("official_split")["good_signal_channels"].sum().to_dict(),
        "labeled_channel_records": frame.groupby("official_split")["official_labeled_channels"].sum().to_dict(),
        "pathological_channel_records": frame.groupby("official_split")["pathological_channels"].sum().to_dict(),
        "normal_channel_records": frame.groupby("official_split")["normal_channels"].sum().to_dict(),
        "unknown_good_channel_records": dict(excluded_counts),
        "normal_precedes_soz_conflict_records": dict(conflict_counts),
        "unique_patient_channels": {part: len(unique_by_split[part]) for part in unique_by_split},
        "unique_pathological_patient_channels": {part: sum(unique_by_split[part].values()) for part in unique_by_split},
        "datasets_with_labeled_channels": eligible.groupby(["official_split", "dataset"])["patient"].nunique().to_dict(),
        "patient_channel_label_conflicts": inconsistency,
    }
    summary["datasets_with_labeled_channels"] = {
        f"{k[0]}|{k[1]}": int(v) for k, v in summary["datasets_with_labeled_channels"].items()}
    encode = lambda value: value.item() if hasattr(value, "item") else str(value)
    (output / "OFFICIAL_LABEL_COUNTS.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=encode) + "\n", encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("eligible_patients", "eligible_edfs",
                                               "unique_patient_channels", "normal_precedes_soz_conflict_records")},
                     default=encode))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("F:/Omni-iEEG/data"))
    parser.add_argument("--cache", type=Path, default=Path("F:/Omni-iEEG/signal_cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.cache, args.output)
