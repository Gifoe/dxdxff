"""Fail-closed A0 reproduction gate for the frozen Sensitivity80 P2 reference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


REFERENCE = {
    "patient_macro_f1": 0.628464,
    "patient_oracle_macro_f1": 0.703912,
    "patient_macro_auprc_ez": 0.521169,
    "patient_macro_ez_mrr": 0.688629,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--tolerance", type=float, default=0.005)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary_path = run_dir / "overall_summary.csv"
    audit_path = run_dir / "cohort_audit.json"
    if not summary_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("A0 baseline gate requires overall_summary.csv and cohort_audit.json")

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("cohort_name") != "sensitivity80" or int(audit.get("n_patients", -1)) != 80:
        raise RuntimeError("A0 baseline gate only accepts the exact Sensitivity80 cohort contract")
    if not audit.get("outer_test_each_subject_once") or not audit.get("outer_train_test_disjoint"):
        raise RuntimeError("A0 baseline cohort/fold audit failed")

    summary = pd.read_csv(summary_path)
    if len(summary) != 1:
        raise RuntimeError("A0 baseline overall summary must contain exactly one row")
    row = summary.iloc[0]
    observed = {name: float(row[name]) for name in REFERENCE}
    deltas = {name: observed[name] - target for name, target in REFERENCE.items()}
    failures = {
        name: value for name, value in deltas.items()
        if abs(value) > float(args.tolerance)
    }
    report = {
        "status": "passed" if not failures else "failed",
        "reference": REFERENCE,
        "observed": observed,
        "delta": deltas,
        "tolerance": float(args.tolerance),
        "cohort_subject_set_hash": audit.get("subject_set_hash"),
        "outer_fold_ledger_hash": audit.get("outer_fold_ledger_hash"),
    }
    report_path = run_dir / "baseline_reproduction_audit.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if failures:
        failure_md = run_dir / "baseline_reproduction_failure.md"
        failure_md.write_text(
            "# A0 Baseline Reproduction Failed\n\n"
            "The requested P2_RTC_SHIFT ablations must not continue because the "
            "Sensitivity80 A0 run differs from the frozen P2 reference beyond the "
            f"predeclared tolerance of {args.tolerance:.3f}.\n\n"
            + "| Metric | Reference | Observed | Delta |\n|---|---:|---:|---:|\n"
            + "\n".join(
                f"| {name} | {REFERENCE[name]:.6f} | {observed[name]:.6f} | {deltas[name]:+.6f} |"
                for name in REFERENCE
            )
            + "\n\nInvestigate cohort ledger, fold ledger, checkpoint selection, threshold protocol, "
            "initialization, and data/cache versions before any A1-A5 run.\n",
            encoding="utf-8",
        )
        raise RuntimeError(f"A0 baseline reproduction failed: {failures}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
