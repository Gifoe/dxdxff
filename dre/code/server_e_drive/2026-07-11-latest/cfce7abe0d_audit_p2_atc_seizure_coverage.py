"""Read-only seizure-count coverage audit for P2_ATC cohorts."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


def _subjects(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [str(row["subject_id"]).strip() for row in csv.DictReader(handle) if str(row.get("subject_id", "")).strip()]


def _excluded(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    return set(_subjects(path))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--all90-ledger", default="reference/all90_subjects.csv")
    parser.add_argument("--sensitivity80-exclusions", default="configs/task1_sensitivity80_exclude_suspected_10.csv")
    args = parser.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    all90 = _subjects(Path(args.all90_ledger)); excluded = _excluded(Path(args.sensitivity80_exclusions))
    with Path(args.feature_cache_path).open("rb") as handle:
        payload = pickle.load(handle)
    runs = payload.get("run_records", [])
    counts: dict[tuple[str, str], int] = defaultdict(int)
    labels: dict[tuple[str, str], int] = {}
    centers: dict[str, str] = {}
    for run in runs:
        subject = str(run.get("subject_id", ""))
        if subject not in all90:
            continue
        center = subject.split(":", 1)[0].lower(); centers[subject] = center
        names = list(run.get("channel_names_norm", [])); values = np.asarray(run.get("labels", []), dtype=float)
        for index, channel in enumerate(names):
            if index >= values.size:
                continue
            key = (subject, str(channel))
            counts[key] += 1
            # Cache labels are EZ=1; ATC trains NEZ=1 exactly once at runtime.
            labels[key] = int(values[index] < 0.5)
    rows = []
    for (subject, channel), count in sorted(counts.items()):
        bucket = "1" if count == 1 else "2" if count == 2 else "3+"
        rows.append({"subject_id": subject, "center": centers[subject], "channel": channel, "valid_seizure_count": count, "count_bucket": bucket, "label_nez": labels[(subject, channel)], "label_role": "clean_nez" if labels[(subject, channel)] else "observed_ez"})
    channel = pd.DataFrame(rows)
    channel.to_csv(out / "p2_atc_seizure_coverage_channel.csv", index=False)
    patient = channel.groupby(["subject_id", "center"], as_index=False).agg(n_channels=("channel", "size"), mean_valid_seizure_count=("valid_seizure_count", "mean"))
    patient.to_csv(out / "p2_atc_seizure_coverage_patient.csv", index=False)
    by_center = channel.groupby(["center", "label_role", "count_bucket"], as_index=False).size().rename(columns={"size": "n_channels"})
    by_center.to_csv(out / "p2_atc_seizure_coverage_by_center.csv", index=False)
    pair_rows = []
    for cohort, allowed in {"primary90": set(all90), "sensitivity80": set(all90) - excluded}.items():
        subset = channel[channel.subject_id.isin(allowed)]
        for subject, frame in subset.groupby("subject_id"):
            clean = frame[(frame.label_nez == 1) & (frame.valid_seizure_count >= 2)]
            observed = frame[(frame.label_nez == 0) & (frame.valid_seizure_count >= 2)]
            trusted = max(1, int(np.ceil(0.20 * len(observed)))) if len(observed) else 0
            pair_rows.append({"cohort": cohort, "subject_id": subject, "center": frame.center.iloc[0], "eligible_clean_nez": len(clean), "eligible_observed_ez": len(observed), "trusted_ez_selected": trusted, "potential_pairs": len(clean) * trusted, "a2_eligible": bool(len(clean)), "a3_eligible": bool(len(clean) and trusted)})
    pairs = pd.DataFrame(pair_rows); pairs.to_csv(out / "p2_atc_pair_coverage.csv", index=False)
    cohort_coverage = {
        cohort: {
            "n_patients_with_channels": int(frame.subject_id.nunique()),
            "a2_eligible_patients": int(frame.a2_eligible.sum()),
            "a3_eligible_patients": int(frame.a3_eligible.sum()),
            "n1_channel_fraction": float((channel[channel.subject_id.isin(set(all90) if cohort == "primary90" else set(all90) - excluded)].valid_seizure_count == 1).mean()),
        }
        for cohort, frame in pairs.groupby("cohort")
    }
    summary = {"status": "passed", "all90_patients": len(all90), "sensitivity80_patients": len(set(all90) - excluded), "n_channels": len(channel), "n_runs": len(runs), "cohort_coverage": cohort_coverage}
    (out / "p2_atc_seizure_coverage_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "P2_ATC_COVERAGE_REPORT.md").write_text("# P2_ATC Seizure Coverage\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in summary.items()) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
