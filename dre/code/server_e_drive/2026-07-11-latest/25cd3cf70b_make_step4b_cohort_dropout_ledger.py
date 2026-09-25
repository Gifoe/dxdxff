from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def read_subject_ids(path: Path) -> list[str]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path}")
        column = next((name for name in reader.fieldnames if str(name).strip().lower() == "subject_id"), None)
        if column is None:
            raise ValueError(f"CSV must contain subject_id: {path}")
        values = [str(row.get(column, "")).strip() for row in reader]
    values = [value for value in values if value]
    if len(values) != len(set(values)):
        raise ValueError(f"CSV contains duplicate subject IDs: {path}")
    return sorted(values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one globally dropped Step4B cohort ledger.")
    parser.add_argument("--all90-subjects", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--drop-count", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dropped-subjects", default="")
    parser.add_argument("--output-ledger", required=True)
    parser.add_argument("--output-manifest", required=True)
    args = parser.parse_args()

    all90 = read_subject_ids(Path(args.all90_subjects))
    candidates = read_subject_ids(Path(args.candidates))
    if len(all90) != 90:
        raise ValueError(f"Expected exactly 90 subjects in the base ledger, got {len(all90)}")
    missing = sorted(set(candidates) - set(all90))
    if missing:
        raise ValueError(f"Candidates outside the All90 ledger: {missing}")
    explicit_dropped = [value.strip() for value in args.dropped_subjects.split(",") if value.strip()]
    if explicit_dropped:
        if len(explicit_dropped) != len(set(explicit_dropped)):
            raise ValueError("dropped-subjects contains duplicate patient IDs")
        missing_from_candidates = sorted(set(explicit_dropped) - set(candidates))
        if missing_from_candidates:
            raise ValueError(f"dropped-subjects outside the candidate list: {missing_from_candidates}")
        dropped = sorted(explicit_dropped)
    else:
        if args.drop_count is None or args.drop_count < 1 or args.drop_count > len(candidates):
            raise ValueError(f"drop-count must be in [1, {len(candidates)}] when dropped-subjects is omitted")
        rng = np.random.default_rng(args.seed)
        dropped = sorted(rng.choice(np.asarray(candidates, dtype=object), size=args.drop_count, replace=False).tolist())
    dropped_set = set(dropped)
    retained = [subject_id for subject_id in all90 if subject_id not in dropped_set]

    ledger_path = Path(args.output_ledger)
    manifest_path = Path(args.output_manifest)
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["subject_id"])
        writer.writeheader()
        writer.writerows({"subject_id": subject_id} for subject_id in retained)
    manifest_path.write_text(json.dumps({
        "base_cohort_size": len(all90),
        "candidate_count": len(candidates),
        "drop_count": len(dropped),
        "random_seed": args.seed,
        "selection_mode": "explicit" if explicit_dropped else "random",
        "dropped_subjects": dropped,
        "retained_subject_count": len(retained),
        "retained_subjects_file": str(ledger_path),
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"dropped_subjects": dropped, "retained_subject_count": len(retained)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
