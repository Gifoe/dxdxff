"""Build the fixed non-LZU cohort for the P2-Q10 ablation.

This is an allow-list operation, not a random patient selection.  The source
ledger remains the protocol authority; only rows whose center is LZU are
removed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


EXPECTED_RETAINED = {"hup": 36, "multicenter": 15, "pediatric": 11}


def build_ledger(source: Path, output: Path, audit: Path, excluded_center: str = "lzu") -> dict:
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows or "subject_id" not in rows[0]:
        raise ValueError("source ledger must contain a subject_id column")
    # The frozen All90 ledger historically contains only subject_id.  Its
    # center is encoded in the prefix (for example, lzu:patient_name).
    for row in rows:
        if not row.get("center"):
            subject_id = str(row["subject_id"])
            row["center"] = subject_id.split(":", 1)[0].strip().lower()
        row["center"] = str(row["center"]).strip().lower()
    if len({row["subject_id"] for row in rows}) != len(rows):
        raise ValueError("source ledger contains duplicate subject_id values")

    excluded_center = excluded_center.strip().lower()
    kept = [row for row in rows if row["center"].strip().lower() != excluded_center]
    excluded = [row for row in rows if row["center"].strip().lower() == excluded_center]
    counts: dict[str, int] = {}
    for row in kept:
        center = row["center"].strip().lower()
        counts[center] = counts.get(center, 0) + 1
    if counts != EXPECTED_RETAINED:
        raise ValueError(f"unexpected retained center counts: {counts}; expected {EXPECTED_RETAINED}")
    if len(kept) != 62 or len(excluded) != 28:
        raise ValueError(f"expected 62 retained and 28 excluded patients, got {len(kept)} and {len(excluded)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["subject_id", "center"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in sorted(kept, key=lambda item: item["subject_id"]):
            writer.writerow({key: row[key] for key in fieldnames})

    report = {
        "status": "passed",
        "source_ledger": str(source),
        "output_ledger": str(output),
        "excluded_center": excluded_center,
        "n_source_patients": len(rows),
        "n_excluded_patients": len(excluded),
        "excluded_subjects": sorted(row["subject_id"] for row in excluded),
        "n_retained_patients": len(kept),
        "retained_center_counts": counts,
        "protocol": "P2_TEMPORAL_Q10_non_lzu_all62_direct_outer_seed42",
        "positive_label": "nez",
        "score_semantics": "nez_probability",
        "training_centers": sorted(counts),
        "lzu_in_training_or_test": False,
    }
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-ledger", required=True, type=Path)
    parser.add_argument("--output-ledger", required=True, type=Path)
    parser.add_argument("--audit-output", required=True, type=Path)
    parser.add_argument("--excluded-center", default="lzu")
    args = parser.parse_args()
    build_ledger(args.source_ledger, args.output_ledger, args.audit_output, args.excluded_center)


if __name__ == "__main__":
    main()
