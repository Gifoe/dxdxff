"""Aggregate completed P2_ATC runs without changing predictions or profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PROFILE_NAMES = {"A0": "A0_P2_TEMPORAL_Q10", "A1": "A1_P2_ROBUST_TAIL", "A2": "A2_P2_NEZ_TAIL", "A3": "A3_P2_ATC"}


def _summary(path: Path) -> dict | None:
    file = path / "overall_summary.csv"
    if not file.is_file():
        return None
    frame = pd.read_csv(file)
    return frame.iloc[0].to_dict() if len(frame) == 1 else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--bootstrap-replicates", type=int, default=10_000)
    args = parser.parse_args(); root = Path(args.root)
    overall_rows, fold_rows, center_rows, tail_rows, loss_rows, paired_rows, bootstrap_rows = [], [], [], [], [], [], []
    for cohort in ("sensitivity80", "primary90"):
        previous = None
        for key, name in PROFILE_NAMES.items():
            run = root / cohort / name / "seed_42"; row = _summary(run)
            if row is None:
                continue
            row.update({"cohort": cohort, "profile": key, "profile_name": name}); overall_rows.append(row)
            for filename, target in (("fold_summary.csv", fold_rows), ("center_summary.csv", center_rows), ("tail_diagnostics.csv", tail_rows), ("train_history.csv", loss_rows)):
                source = run / filename
                if source.is_file():
                    frame = pd.read_csv(source); frame["cohort"] = cohort; frame["profile"] = key; target.extend(frame.to_dict("records"))
            if previous is not None:
                before = root / cohort / PROFILE_NAMES[previous] / "seed_42" / "outer_test_channel_predictions.csv"
                after = run / "outer_test_channel_predictions.csv"
                if before.is_file() and after.is_file():
                    b, a = pd.read_csv(before), pd.read_csv(after)
                    keys = ["subject_id", "channel"]
                    merged = b.merge(a, on=keys, suffixes=("_before", "_after"))
                    if len(merged):
                        for subject_id, patient in merged.groupby("subject_id"):
                            before_f1 = f1_score(patient.label_nez_before, patient.prediction_nez_before, average="macro", zero_division=0)
                            after_f1 = f1_score(patient.label_nez_after, patient.prediction_nez_after, average="macro", zero_division=0)
                            paired_rows.append({"cohort": cohort, "comparison": f"{key}-{previous}", "subject_id": subject_id, "macro_f1_delta_after_minus_before": float(after_f1 - before_f1)})
            previous = key
        a0 = root / cohort / PROFILE_NAMES["A0"] / "seed_42" / "outer_test_channel_predictions.csv"
        a3 = root / cohort / PROFILE_NAMES["A3"] / "seed_42" / "outer_test_channel_predictions.csv"
        if a0.is_file() and a3.is_file():
            b, a = pd.read_csv(a0), pd.read_csv(a3); merged = b.merge(a, on=["subject_id", "channel"], suffixes=("_a0", "_a3"))
            deltas = np.asarray([f1_score(x.label_nez_a3, x.prediction_nez_a3, average="macro", zero_division=0) - f1_score(x.label_nez_a0, x.prediction_nez_a0, average="macro", zero_division=0) for _, x in merged.groupby("subject_id")], dtype=float)
            rng = np.random.default_rng(42); samples = np.asarray([rng.choice(deltas, len(deltas), replace=True).mean() for _ in range(args.bootstrap_replicates)])
            bootstrap_rows.append({"cohort": cohort, "comparison": "A3-A0", "mean_delta": float(deltas.mean()), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975)), "n_patients": len(deltas)})
    overall = pd.DataFrame(overall_rows)
    if not overall.empty:
        for metric in ("patient_macro_f1", "patient_macro_auprc_ez", "patient_macro_ez_mrr", "patient_oracle_macro_f1"):
            if metric in overall:
                overall[f"delta_{metric}_vs_a0"] = overall.groupby("cohort")[metric].transform(lambda x: x - x.iloc[0])
    overall.to_csv(root / "p2_atc_overall_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(root / "p2_atc_by_fold.csv", index=False); pd.DataFrame(center_rows).to_csv(root / "p2_atc_by_center.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(root / "p2_atc_patient_paired_delta.csv", index=False); pd.DataFrame(tail_rows).to_csv(root / "p2_atc_tail_diagnostics.csv", index=False)
    pd.DataFrame(loss_rows).to_csv(root / "p2_atc_loss_diagnostics.csv", index=False); pd.DataFrame(bootstrap_rows).to_csv(root / "p2_atc_bootstrap_ci.csv", index=False)
    (root / "P2_ATC_FINAL_REPORT.md").write_text("# P2_ATC Final Report\n\nThis report aggregates completed seed-42 runs only. A3-A0 bootstrap confidence intervals are descriptive and were not used for profile selection.\n", encoding="utf-8")


if __name__ == "__main__":
    main()
