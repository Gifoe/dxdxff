from __future__ import annotations

import argparse
import csv
import pickle
from pathlib import Path
from typing import Any


def center_of(subject_id: str, meta: dict[str, Any]) -> str:
    values = (
        meta.get("source_center"),
        meta.get("center"),
        meta.get("source_dataset"),
        str(subject_id).split(":", 1)[0] if ":" in str(subject_id) else None,
    )
    for value in values:
        value = str(value or "").strip().lower()
        if value.startswith("ped") or value == "pediatric":
            return "pediatric"
        if value.startswith("hup"):
            return "hup"
        if value.startswith("lzu"):
            return "lzu"
        if value.startswith("multi"):
            return "multicenter"
    return "unknown"


def read_subjects(path: str) -> set[str] | None:
    if not path:
        return None
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = csv.DictReader(handle)
        fieldnames = list(rows.fieldnames or [])
        lower_names = {str(name).strip().lower() for name in fieldnames}
        column_name = next(
            (name for name in ("subject_id", "patient_id", "subject") if name in lower_names),
            None,
        )
        if column_name is None:
            raise ValueError("Subjects file must contain subject_id, patient_id, or subject")
        actual_column = next(name for name in fieldnames if str(name).strip().lower() == column_name)
        return {
            str(row.get(actual_column, "")).strip()
            for row in rows
            if str(row.get(actual_column, "")).strip()
        }


def audit(cache_path: str, subjects_path: str, threshold: float) -> list[dict[str, Any]]:
    with Path(cache_path).open("rb") as handle:
        payload = pickle.load(handle)
    patient_index = payload.get("patient_index")
    if not isinstance(patient_index, dict):
        raise ValueError("Cache does not contain a dictionary patient_index")

    allowed = read_subjects(subjects_path)
    rows: list[dict[str, Any]] = []
    for raw_subject_id, raw_meta in patient_index.items():
        subject_id = str(raw_subject_id)
        if allowed is not None and subject_id not in allowed:
            continue
        meta = raw_meta or {}
        if center_of(subject_id, meta) != "pediatric":
            continue
        labels = meta.get("labels_ez", meta.get("labels"))
        if labels is None:
            raise ValueError(f"Missing EZ-positive labels for {subject_id}")
        labels = list(labels)
        label_mask = meta.get("label_mask", [True] * len(labels))
        valid = [float(label) >= 0.0 and bool(flag) for label, flag in zip(labels, label_mask)]
        valid_channels = int(sum(valid))
        ez_channels = int(sum(valid[index] and float(labels[index]) > 0.5 for index in range(len(labels))))
        ez_fraction = float(ez_channels / valid_channels) if valid_channels else 0.0
        if ez_fraction > threshold:
            rows.append({
                "subject_id": subject_id,
                "center": "pediatric",
                "ez_channels": ez_channels,
                "valid_channels": valid_channels,
                "ez_fraction": ez_fraction,
                "above_threshold": True,
            })
    return sorted(rows, key=lambda row: (-row["ez_fraction"], row["subject_id"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="List pediatric patients with high EZ-channel prevalence.")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--subjects", default="")
    parser.add_argument("--threshold", type=float, required=True)
    parser.add_argument("--output-csv", default="")
    args = parser.parse_args()

    rows = audit(args.cache, args.subjects, args.threshold)
    if args.output_csv:
        output = Path(args.output_csv)
        output.parent.mkdir(parents=True, exist_ok=True)
        fields = ["subject_id", "center", "ez_channels", "valid_channels", "ez_fraction", "above_threshold"]
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    print(f"cache: {args.cache}")
    print(f"cohort_filter: {args.subjects or '<all cache patients>'}")
    print(f"threshold: EZ fraction > {args.threshold:.6f}")
    print(f"pediatric patients above threshold: {len(rows)}")
    for row in rows:
        print(f"{row['subject_id']}\tez={row['ez_channels']}/{row['valid_channels']}\tez_fraction={row['ez_fraction']:.6f}")
    if args.output_csv:
        print(f"csv: {args.output_csv}")


if __name__ == "__main__":
    main()
