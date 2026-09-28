"""Metadata-only preflight for the frozen Omni-iEEG Task-2 SOZ cohort.

No waveform samples or model outcomes are read here.  The official split is
authoritative; labels come only from the corresponding channels.tsv ``soz``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import pandas as pd


REVISION = "73b9c5180a57828ab2a83c040e7e9d112e77b2cc"
OMNI_CODE = "57c22a75a59b5c3a98006806ad42000f6a3fa5b6"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(path)


def strict_label(value: object) -> int | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if number == 0:
        return 0
    if number == 1:
        return 1
    return None


def run(source: Path, cache: Path, output: Path) -> None:
    split_path = source / "derivatives/datasplit/final_split.csv"
    participants = source / "participants.tsv"
    validation_path = cache / "CACHE_VALIDATION.json"
    migration_path = cache / "CACHE_MIGRATION.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    migration = json.loads(migration_path.read_text(encoding="utf-8"))
    if not validation.get("pass") or not migration.get("pass"):
        raise RuntimeError("Native signal cache has not passed final verification")
    if validation.get("revision") != REVISION or migration.get("revision") != REVISION:
        raise RuntimeError("Unexpected Omni dataset revision")
    split = pd.read_csv(split_path)
    needed = {"patient_name", "edf_name", "dataset", "split", "frequency", "interictal", "length"}
    if not needed <= set(split.columns):
        raise RuntimeError(f"Official split columns missing: {needed - set(split.columns)}")
    cohort = split.loc[
        split["split"].isin(["train", "test"])
        & (split["dataset"] != "Multicenter")
        & (pd.to_numeric(split["frequency"], errors="coerce") > 900)
        & (split["interictal"] == True)  # noqa: E712 - exact boolean condition
        & (pd.to_numeric(split["length"], errors="coerce") >= 62)
    ].copy()
    if cohort["edf_name"].duplicated().any():
        raise RuntimeError("Duplicate EDF in official filtered split")
    membership = cohort.groupby("patient_name")["split"].nunique()
    if (membership != 1).any():
        raise RuntimeError("Patient appears in both train and test")
    if not participants.exists():
        raise RuntimeError("participants.tsv missing")
    patient_metadata = pd.read_csv(participants, sep="\t")
    known_patients = set(patient_metadata["participant_id"].astype(str))
    missing_patients = sorted(set(cohort["patient_name"].astype(str)) - known_patients)
    if missing_patients:
        raise RuntimeError(f"Filtered patients missing from participants.tsv: {missing_patients[:5]}")

    output.mkdir(parents=True, exist_ok=True)
    edf_rows: list[dict] = []
    missing: list[dict] = []
    duplicate_labels: dict[tuple[str, str], set[int]] = defaultdict(set)
    sample_rates: Counter[str] = Counter()
    for row in cohort.itertuples(index=False):
        relative = Path(str(row.edf_name))
        if relative.is_absolute() or ".." in relative.parts or relative.suffix.lower() != ".edf":
            raise RuntimeError(f"Invalid official EDF path: {relative}")
        h5_path = cache / relative.with_suffix(".edf.h5")
        sidecar = source / Path(str(relative).replace("_ieeg.edf", "_channels.tsv"))
        if not h5_path.is_file() or not sidecar.is_file():
            missing.append({"edf": relative.as_posix(), "cache": h5_path.is_file(), "channels_tsv": sidecar.is_file()})
            continue
        channels = pd.read_csv(sidecar, sep="\t")
        if not {"name", "good", "soz"} <= set(channels.columns):
            raise RuntimeError(f"Required channel fields missing: {relative}")
        with h5py.File(h5_path, "r") as h5:
            if h5.attrs["schema"] != "omni-ieeg-native-digital-v1":
                raise RuntimeError(f"Unexpected HDF5 schema: {relative}")
            if str(h5.attrs["dataset_revision"]) != REVISION:
                raise RuntimeError(f"HDF5 revision mismatch: {relative}")
            meta = json.loads(h5["metadata_json"][()].decode("utf-8"))
            names = [str(h["label"]) for h in meta["signal_headers"]]
            if len(names) != len(set(names)):
                raise RuntimeError(f"Duplicate EDF channel names: {relative}")
            name_set = set(names)
            duration = float(meta["file_duration_seconds"])
            for header in meta["signal_headers"]:
                sample_rates[str(header.get("sample_frequency", header.get("sample_rate", "missing")))] += 1
        good = channels.loc[pd.to_numeric(channels["good"], errors="coerce") == 1].copy()
        good["soz_clean"] = good["soz"].map(strict_label)
        good = good.loc[good["soz_clean"].notna() & good["name"].astype(str).isin(name_set)]
        if good["name"].duplicated().any():
            raise RuntimeError(f"Duplicate supervised channel in TSV: {relative}")
        n_soz = int((good["soz_clean"] == 1).sum())
        n_non_soz = int((good["soz_clean"] == 0).sum())
        for record in good.itertuples(index=False):
            duplicate_labels[(str(row.patient_name), str(record.name))].add(int(record.soz_clean))
        edf_rows.append({
            "dataset": str(row.dataset), "patient": str(row.patient_name),
            "official_split": str(row.split), "edf": relative.as_posix(),
            "interictal": bool(row.interictal), "original_frequency": float(row.frequency),
            "split_duration_sec": float(row.length), "cache_duration_sec": duration,
            "valid_channels": int(len(good)), "soz_channels": n_soz,
            "non_soz_channels": n_non_soz,
            "uniform_60s_segments_per_channel": max(0, int((duration - 2) // 60)),
            "cache_h5_bytes": h5_path.stat().st_size,
        })
    conflicts = sorted(f"{p}:{c}" for (p, c), labels in duplicate_labels.items() if len(labels) > 1)
    if missing or conflicts:
        atomic_json(output / "OMNI_CACHE_AUDIT.json", {"pass": False, "missing": missing, "label_conflicts": conflicts})
        raise RuntimeError(f"Cohort preflight failed: missing={len(missing)}, label_conflicts={len(conflicts)}")
    with (output / "OMNI_COHORT_AUDIT.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(edf_rows[0]))
        writer.writeheader()
        writer.writerows(edf_rows)
    unique_labels = {key: next(iter(labels)) for key, labels in duplicate_labels.items()}
    patient_splits = {str(k): str(v) for k, v in cohort.groupby("patient_name")["split"].first().items()}
    patient_counts = Counter(patient_splits.values())
    split_counts = Counter(row["official_split"] for row in edf_rows)
    supervised_rows = [row for row in edf_rows if row["valid_channels"] > 0]
    supervised_patients = {
        part: len({row["patient"] for row in supervised_rows if row["official_split"] == part})
        for part in ("train", "test")
    }
    supervised_edfs = dict(Counter(row["official_split"] for row in supervised_rows))
    no_label_edfs_by_dataset = dict(Counter(row["dataset"] for row in edf_rows if row["valid_channels"] == 0))
    soz_count = sum(unique_labels.values())
    label_audit = {
        "positive_clinical_class": "EZ=SOZ=1", "internal_A1_label": "EZ=0, NEZ=1",
        "unknown_soz_excluded": True, "resection_used": False, "outcome_used": False,
        "source": "channels.tsv:soz only", "unique_patient_channels": len(unique_labels),
        "soz_channels": soz_count, "non_soz_channels": len(unique_labels) - soz_count,
        "patient_channel_label_conflicts": conflicts,
    }
    atomic_json(output / "LABEL_SEMANTICS_AUDIT.json", label_audit)
    atomic_json(output / "COHORT_SUPERVISION_AUDIT.json", {
        "official_filtered_patients": dict(patient_counts),
        "official_filtered_edfs": dict(split_counts),
        "patients_with_at_least_one_good_known_soz_channel": supervised_patients,
        "edfs_with_at_least_one_good_known_soz_channel": supervised_edfs,
        "edfs_without_supervised_channels_by_dataset": no_label_edfs_by_dataset,
        "rule": "Keep official split and all metadata; exclude only channels with soz unknown/-1, and therefore records/patients with no known supervised label cannot enter supervised training/evaluation.",
        "resection_or_outcome_fallback": False,
    })
    atomic_json(output / "OMNI_CACHE_AUDIT.json", {
        "pass": True, "source_root": str(source), "cache_root": str(cache),
        "dataset_revision": REVISION, "official_code_revision": OMNI_CODE,
        "cache_schema": validation.get("schema"), "all_cache_edfs": validation.get("edf_files"),
        "all_cache_sidecars": migration.get("sidecar_cache_files"),
        "all_cache_bytes": validation.get("cache_bytes"),
        "participants_tsv_sha256": sha256(participants),
        "official_split_sha256": sha256(split_path),
        "cohort_patients": dict(patient_counts), "cohort_edfs": dict(split_counts),
        "supervised_patients": supervised_patients, "supervised_edfs": supervised_edfs,
        "edfs_without_supervised_labels_by_dataset": no_label_edfs_by_dataset,
        "cohort_edfs_total": len(edf_rows), "cohort_valid_channel_records": sum(r["valid_channels"] for r in edf_rows),
        "unique_supervised_patient_channels": len(unique_labels),
        "unique_soz_channels": soz_count, "unique_non_soz_channels": len(unique_labels) - soz_count,
        "sampling_rates_in_headers": dict(sample_rates),
        "duration_sec_min": min(r["cache_duration_sec"] for r in edf_rows),
        "duration_sec_max": max(r["cache_duration_sec"] for r in edf_rows),
        "expected_uniform_60s_channel_segments": sum(r["uniform_60s_segments_per_channel"] * r["valid_channels"] for r in edf_rows),
        "missing_metadata": missing, "label_conflicts": conflicts,
    })
    print(json.dumps({"patients": dict(patient_counts), "edfs": dict(split_counts),
                      "valid_channel_records": sum(r["valid_channels"] for r in edf_rows),
                      "unique_patient_channels": len(unique_labels), "missing": len(missing)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, default=Path("F:/Omni-iEEG/data"))
    parser.add_argument("--cache", type=Path, default=Path("F:/Omni-iEEG/signal_cache"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    run(args.source, args.cache, args.output)
