"""Read-only comparison audit for the predeclared P2_RTC_SHIFT profiles."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


PROFILE_NAMES = {
    "A0": "A0_P2_TEMPORAL_Q10", "A1": "A1_ROBUST_TAIL", "A2": "A2_TAIL_RELIABILITY",
    "A3": "A3_TAIL_RANK", "A4": "A4_EZ_AWARE_SELECTION", "A5": "A5_P2_RTC_SHIFT",
}
METRICS = (
    "patient_macro_f1", "patient_macro_nez_f1", "patient_macro_ez_f1", "patient_macro_balanced_accuracy",
    "patient_oracle_macro_f1", "patient_macro_auprc_ez", "patient_macro_auprc_nez", "patient_macro_ez_mrr",
    "ez_recall_at_true_count", "classification_threshold",
)


def _read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path) if path.is_file() else pd.DataFrame()


def _patient_f1(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or not {"subject_id", "label_nez", "prediction_nez"}.issubset(frame.columns):
        return pd.DataFrame(columns=["subject_id", "patient_macro_f1"])
    rows = []
    for subject_id, group in frame.groupby("subject_id"):
        labels = group["label_nez"].to_numpy(dtype=int)
        predicted = group["prediction_nez"].to_numpy(dtype=int)
        rows.append({"subject_id": subject_id, "patient_macro_f1": f1_score(labels, predicted, average="macro", zero_division=0)})
    return pd.DataFrame(rows)


def _bootstrap_delta(a0: pd.DataFrame, a5: pd.DataFrame, *, seed: int = 42, draws: int = 10_000) -> dict[str, float]:
    paired = a0.merge(a5, on="subject_id", suffixes=("_a0", "_a5"))
    if paired.empty:
        return {"n_paired_patients": 0, "macro_f1_delta": float("nan"), "ci95_low": float("nan"), "ci95_high": float("nan")}
    values = (paired["patient_macro_f1_a5"] - paired["patient_macro_f1_a0"]).to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(draws, len(values)))
    samples = values[indices].mean(axis=1)
    return {"n_paired_patients": len(values), "macro_f1_delta": float(values.mean()), "ci95_low": float(np.quantile(samples, .025)), "ci95_high": float(np.quantile(samples, .975))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root: Path = args.root
    overall_rows, fold_rows, center_rows, tail_rows, shift_rows, paired_rows = [], [], [], [], [], []
    for cohort in ("sensitivity80", "primary90"):
        cohort_dir = root / cohort
        for short, profile in PROFILE_NAMES.items():
            run_dir = cohort_dir / profile / "seed_42"
            summary = _read(run_dir / "overall_summary.csv")
            if summary.empty:
                continue
            row = dict(summary.iloc[0]); row.update({"cohort": cohort, "profile": short, "profile_name": profile, "run_dir": str(run_dir)})
            overall_rows.append(row)
            for source, collection in (("fold_summary.csv", fold_rows), ("center_summary.csv", center_rows), ("tail_diagnostics.csv", tail_rows), ("shift_calibrator_audit.csv", shift_rows)):
                frame = _read(run_dir / source)
                if not frame.empty:
                    frame.insert(0, "cohort", cohort); frame.insert(1, "profile", short); collection.extend(frame.to_dict("records"))
        a0 = _patient_f1(_read(cohort_dir / PROFILE_NAMES["A0"] / "seed_42" / "outer_test_channel_predictions.csv"))
        a5 = _patient_f1(_read(cohort_dir / PROFILE_NAMES["A5"] / "seed_42" / "outer_test_channel_predictions.csv"))
        paired_rows.append({"cohort": cohort, **_bootstrap_delta(a0, a5)})
    overall = pd.DataFrame(overall_rows)
    if not overall.empty:
        ordered = overall.assign(_order=overall["profile"].map({name: index for index, name in enumerate(PROFILE_NAMES)})).sort_values(["cohort", "_order"])
        adjacent = []
        for cohort, group in ordered.groupby("cohort"):
            group = group.sort_values("_order")
            for (_, previous), (_, current) in zip(group.iloc[:-1].iterrows(), group.iloc[1:].iterrows()):
                row = {"cohort": cohort, "profile": current["profile"], "previous_profile": previous["profile"]}
                for metric in METRICS:
                    row[f"delta_{metric}"] = float(current.get(metric, np.nan)) - float(previous.get(metric, np.nan))
                adjacent.append(row)
        overall = overall.merge(pd.DataFrame(adjacent), on=["cohort", "profile"], how="left")
    overall.to_csv(root / "p2_rtc_shift_comparison.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(root / "p2_rtc_shift_by_fold.csv", index=False)
    pd.DataFrame(center_rows).to_csv(root / "p2_rtc_shift_by_center.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(root / "p2_rtc_shift_patient_paired_delta.csv", index=False)
    pd.DataFrame(tail_rows).to_csv(root / "p2_rtc_shift_tail_audit.csv", index=False)
    pd.DataFrame(shift_rows).to_csv(root / "p2_rtc_shift_calibrator_audit.csv", index=False)
    lines = ["# P2_RTC_SHIFT Report", "", "Held-out values are reporting metrics only; no profile is selected on them.", ""]
    if not overall.empty:
        lines.extend(["## Overall", "", "```text", overall.to_string(index=False), "```", ""])
    if paired_rows:
        lines.extend(["## Patient-Paired Bootstrap: A5 - A0", "", "```text", pd.DataFrame(paired_rows).to_string(index=False), "```", ""])
    lines.extend(["## Cohort Interpretation", "", "sensitivity80 is a predeclared subset of primary90 and is not an independent external validation cohort.", ""])
    (root / "P2_RTC_SHIFT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
