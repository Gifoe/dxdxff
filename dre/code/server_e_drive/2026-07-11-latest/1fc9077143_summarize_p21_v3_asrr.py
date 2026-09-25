from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROFILES = (
    "R0_CURRENT_P2", "R1_BOUNDED_SIMPLEX", "R2_WEIGHTED_RANK_PRESERVE",
    "R3_V3_ANCHORED", "R4_Q10_MULTI_SEIZURE", "R5_FULL",
)


def summarize(run_root: str | Path) -> Path:
    root = Path(run_root)
    output = root / "profile_comparison"
    output.mkdir(parents=True, exist_ok=True)
    kinds = {"overall": "p21_overall_summary.csv", "by_fold": "p21_fold_summary.csv", "by_center": "p21_center_summary.csv", "oracle_comparison": "p21_v3_vs_final_oracle.csv"}
    for kind, filename in kinds.items():
        frames = []
        for profile in PROFILES:
            for source in sorted((root / profile).glob(f"seed_*/{filename}")):
                frame = pd.read_csv(source)
                frame.insert(0, "profile", profile)
                frame.insert(1, "seed", int(source.parent.name.split("_")[-1]))
                frames.append(frame)
        (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame([{"status": "missing"}])).to_csv(output / f"p21_R0_R5_{kind}.csv", index=False)
    ranking = pd.read_csv(output / "p21_R0_R5_overall.csv")
    columns = [name for name in ("profile", "seed", "patient_macro_auprc_nez", "patient_macro_auprc_ez", "patient_macro_ez_mrr", "patient_macro_f1") if name in ranking]
    ranking[columns].to_csv(output / "p21_R0_R5_ranking_comparison.csv", index=False)
    report = ["# P2.1 V3-ASRR Ablation Report", "", "Screening runs are not formal nested results. Patient oracle and true-count diagnostics are not deployable.", "", ranking[columns].to_markdown(index=False) if columns else "No completed runs."]
    (output / "P21_ABLATION_REPORT.md").write_text("\n".join(report), encoding="utf-8")
    reports = root / "reports"; reports.mkdir(exist_ok=True)
    (reports / "P21_V3_ASRR_SENSITIVITY80_REPORT.md").write_text("\n".join(report + ["", "Sensitivity80 is a post-hoc cohort. Ridge-VAR is a proxy, not causal proof. No result is guaranteed to exceed 0.70."]), encoding="utf-8")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True)
    args = parser.parse_args()
    print(summarize(args.run_root))
