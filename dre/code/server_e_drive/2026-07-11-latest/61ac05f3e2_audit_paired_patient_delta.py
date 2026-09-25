from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from scripts.audit_prevalence_lift import build_prevalence_table


def run(baseline_pred_dir: Path, candidate_pred_dir: Path, output_dir: Path) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    baseline = build_prevalence_table(baseline_pred_dir)
    candidate = build_prevalence_table(candidate_pred_dir)
    baseline_subjects = set(baseline["subject_id"])
    candidate_subjects = set(candidate["subject_id"])
    missing_in_candidate = sorted(baseline_subjects - candidate_subjects)
    missing_in_baseline = sorted(candidate_subjects - baseline_subjects)
    merged = candidate.merge(
        baseline[
            [
                "subject_id",
                "patient_macro_f1",
                "patient_ez_f1",
                "patient_auprc_ez",
                "ez_mrr",
                "recall_lift",
            ]
        ].rename(
            columns={
                "patient_macro_f1": "baseline_patient_macro_f1",
                "patient_ez_f1": "baseline_patient_ez_f1",
                "patient_auprc_ez": "baseline_patient_auprc_ez",
                "ez_mrr": "baseline_ez_mrr",
                "recall_lift": "baseline_recall_lift",
            }
        ),
        on="subject_id",
        how="inner",
    )
    merged["delta_patient_macro_f1"] = merged["patient_macro_f1"] - merged["baseline_patient_macro_f1"]
    merged["delta_patient_ez_f1"] = merged["patient_ez_f1"] - merged["baseline_patient_ez_f1"]
    merged["delta_auprc"] = merged["patient_auprc_ez"] - merged["baseline_patient_auprc_ez"]
    merged["delta_mrr"] = merged["ez_mrr"] - merged["baseline_ez_mrr"]
    merged["delta_recall_lift"] = merged["recall_lift"] - merged["baseline_recall_lift"]
    merged.to_csv(output_dir / "paired_patient_delta.csv", index=False)
    merged.groupby("center", dropna=False).mean(numeric_only=True).reset_index().to_csv(
        output_dir / "paired_by_center_summary.csv",
        index=False,
    )
    merged.sort_values("delta_patient_macro_f1", ascending=False).head(20).to_csv(
        output_dir / "paired_improved_top20.csv",
        index=False,
    )
    merged.sort_values("delta_patient_macro_f1", ascending=True).head(20).to_csv(
        output_dir / "paired_degraded_top20.csv",
        index=False,
    )
    summary: dict[str, Any] = {
        "n_joined_patients": int(len(merged)),
        "missing_in_candidate": missing_in_candidate,
        "missing_in_baseline": missing_in_baseline,
        "mean_delta_patient_macro_f1": float(merged["delta_patient_macro_f1"].mean()) if not merged.empty else 0.0,
        "mean_delta_patient_ez_f1": float(merged["delta_patient_ez_f1"].mean()) if not merged.empty else 0.0,
        "mean_delta_auprc": float(merged["delta_auprc"].mean()) if not merged.empty else 0.0,
        "mean_delta_mrr": float(merged["delta_mrr"].mean()) if not merged.empty else 0.0,
        "mean_delta_recall_lift": float(merged["delta_recall_lift"].mean()) if not merged.empty else 0.0,
    }
    (output_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if missing_in_candidate or missing_in_baseline:
        print(
            "[paired-delta] WARNING: subject sets differ | "
            f"missing_in_candidate={len(missing_in_candidate)} | missing_in_baseline={len(missing_in_baseline)}",
            flush=True,
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired per-patient deltas between baseline and candidate predictions.")
    parser.add_argument("--baseline_pred_dir", required=True, type=Path)
    parser.add_argument("--candidate_pred_dir", required=True, type=Path)
    parser.add_argument("--output_dir", required=True, type=Path)
    args = parser.parse_args()
    summary = run(args.baseline_pred_dir, args.candidate_pred_dir, args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
