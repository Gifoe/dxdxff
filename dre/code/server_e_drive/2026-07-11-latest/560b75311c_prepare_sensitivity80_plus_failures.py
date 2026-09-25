from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from neuroez_c.task2.outcomes import load_outcome_table


def _patient_column(frame: pd.DataFrame) -> str:
    column = next((value for value in ("patient_key", "patient_id", "subject_id") if value in frame), None)
    if column is None:
        raise ValueError("Manifest requires patient_key/patient_id/subject_id")
    return column


def _fold_mapping(path: str | Path) -> dict[str, int]:
    frame = pd.read_csv(Path(path).expanduser())
    patient = _patient_column(frame)
    fold = next((value for value in ("outer_fold", "fold_idx") if value in frame), None)
    if fold is None:
        raise ValueError(f"Fold column missing from {path}")
    if "partition" in frame:
        frame = frame[frame["partition"].astype(str).str.lower() == "test"]
    if frame[patient].astype(str).duplicated().any():
        raise ValueError(f"Duplicate outer-test patients in {path}")
    return dict(zip(frame[patient].astype(str), frame[fold].astype(int)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Build fixed sensitivity80-success plus failure Task-2 protocol")
    parser.add_argument("--cache", required=True)
    parser.add_argument("--success_subjects", required=True)
    parser.add_argument("--p2_fold_ledger", required=True)
    parser.add_argument("--fixed_fold_source", required=True)
    parser.add_argument("--fallback_failure_folds")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    with Path(args.cache).expanduser().open("rb") as handle:
        cache = pickle.load(handle)
    outcomes, _ = load_outcome_table(None, cache=cache)
    outcomes = outcomes[outcomes["outcome_group"].isin(("success", "failure"))].copy()
    outcome_lookup = outcomes.set_index("patient_key")

    success_frame = pd.read_csv(Path(args.success_subjects).expanduser())
    success_column = _patient_column(success_frame)
    success_subjects = set(success_frame[success_column].astype(str))
    missing_success = sorted(success_subjects - set(outcome_lookup.index))
    wrong_success = sorted(subject for subject in success_subjects if subject in outcome_lookup.index and outcome_lookup.loc[subject, "outcome_group"] != "success")
    if missing_success or wrong_success:
        raise ValueError(f"Invalid sensitivity80 subjects; missing={missing_success[:10]}, non_success={wrong_success[:10]}")

    failure_subjects = set(outcomes.loc[outcomes["outcome_group"] == "failure", "patient_key"].astype(str))
    if not failure_subjects:
        raise ValueError("No failure patients were resolved from the complete cache")
    target = success_subjects | failure_subjects

    p2_folds = _fold_mapping(args.p2_fold_ledger)
    missing_p2 = sorted(success_subjects - set(p2_folds))
    if missing_p2:
        raise ValueError(f"Sensitivity80 patients missing P2 outer-test fold: {missing_p2[:20]}")
    assignments = {subject: p2_folds[subject] for subject in success_subjects}

    fixed = _fold_mapping(args.fixed_fold_source)
    fallback = _fold_mapping(args.fallback_failure_folds) if args.fallback_failure_folds else {}
    missing_failure = []
    for subject in sorted(failure_subjects):
        if subject in fixed:
            assignments[subject] = fixed[subject]
        elif subject in fallback:
            assignments[subject] = fallback[subject]
        else:
            missing_failure.append(subject)
    if missing_failure:
        raise ValueError(f"Failure patients missing fixed fold assignment: {missing_failure[:20]}")

    fold_values = sorted(set(assignments.values()))
    if len(fold_values) != 5:
        raise ValueError(f"Combined protocol requires exactly five folds; found {fold_values}")
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    cohort = outcomes[outcomes["patient_key"].isin(target)][["patient_key", "center", "outcome_group", "outcome_label"]].sort_values("patient_key")
    folds = cohort.copy()
    folds["outer_fold"] = folds["patient_key"].map(assignments).astype(int)
    folds["success_label"] = 1
    folds["failure_label"] = 0
    cohort_path = output / "sensitivity80_plus_failures_cohort.csv"
    folds_path = output / "fixed_outer_folds_sensitivity80_plus_failures.csv"
    cohort.to_csv(cohort_path, index=False)
    folds.to_csv(folds_path, index=False)
    audit = {
        "n_patients": int(len(cohort)),
        "n_success": int((cohort["outcome_label"] == 1).sum()),
        "n_failure": int((cohort["outcome_label"] == 0).sum()),
        "fold_sizes": folds.groupby("outer_fold").size().astype(int).to_dict(),
        "fold_success": folds.groupby("outer_fold")["outcome_label"].sum().astype(int).to_dict(),
        "fold_failure": folds.groupby("outer_fold")["outcome_label"].apply(lambda values: int((values == 0).sum())).to_dict(),
        "success_fold_source": str(Path(args.p2_fold_ledger).expanduser().resolve()),
        "failure_fold_source": str(Path(args.fixed_fold_source).expanduser().resolve()),
        "fallback_failure_fold_source": str(Path(args.fallback_failure_folds).expanduser().resolve()) if args.fallback_failure_folds else None,
        "cohort_path": str(cohort_path.resolve()),
        "folds_path": str(folds_path.resolve()),
    }
    (output / "sensitivity80_plus_failures_protocol_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
