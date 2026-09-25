from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROFILES = ("P0", "P1", "P2", "P3", "P4")
METRICS = (
    "patient_macro_f1", "patient_macro_nez_f1", "patient_macro_ez_f1",
    "patient_macro_auprc_nez", "patient_macro_auprc_ez", "patient_macro_ez_mrr",
    "top1_is_ez_rate",
)


def _summary_path(root: Path, profile: str) -> Path | None:
    candidates = (
        [root / "P2" / "ensemble" / "heldout_summary_cane_path_cp.csv"]
        if profile == "P2" else [
            root / profile / "seed_42" / "heldout_summary_cane_path_cp.csv",
            root / profile / "seed_42" / "heldout_summary_neuroez_v3.csv",
        ]
    )
    return next((path for path in candidates if path.is_file()), None)


def summarize(run_root: str | Path, causal_audit_dir: str | Path | None = None) -> Path:
    root = Path(run_root)
    destination = root / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for profile in PROFILES:
        path = _summary_path(root, profile)
        if path is None:
            rows.append({"profile": profile, "status": "missing", "formal_profile": profile == "P2"})
            continue
        source = pd.read_csv(path).iloc[0].to_dict()
        rows.append({
            "profile": profile, "status": "complete", "formal_profile": profile == "P2",
            "selection_role": "predeclared_formal" if profile == "P2" else "reporting_ablation_only",
            "summary_path": str(path), **{metric: source.get(metric, np.nan) for metric in METRICS},
        })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(destination / "profile_comparison.csv", index=False)
    formal = comparison[comparison.profile == "P2"]
    baseline = formal.iloc[0] if len(formal) and formal.iloc[0].get("status") == "complete" else None
    ablations = []
    for profile in ("P0", "P1", "P3", "P4"):
        row = comparison[comparison.profile == profile].iloc[0]
        result = {"formal_profile": "P2", "comparison_profile": profile, "status": row["status"]}
        for metric in METRICS:
            result[f"delta_{metric}_p2_minus_comparison"] = (
                float(baseline[metric]) - float(row[metric])
                if baseline is not None and row["status"] == "complete" and pd.notna(baseline.get(metric)) and pd.notna(row.get(metric))
                else np.nan
            )
        ablations.append(result)
    pd.DataFrame(ablations).to_csv(destination / "ablation_comparison.csv", index=False)

    gap_rows = []
    for profile in ("P1", "P2"):
        path = root / profile / "seed_42" / "inner_oof_summary_by_outer_fold.csv"
        if not path.is_file():
            gap_rows.append({"profile": profile, "status": "missing", "ranking_oracle_macro_f1": np.nan})
            continue
        frame = pd.read_csv(path)
        center_means = {
            f"{center}_legal_path_macro_f1": float(frame[f"{center}_legal_path_macro_f1"].mean())
            for center in ("hup", "lzu", "multicenter", "pediatric")
        }
        gap_rows.append({
            "profile": profile, "status": "complete",
            "n_outer_folds": int(len(frame)),
            "ranking_oracle_macro_f1": float(frame["patient_ranking_oracle_macro_f1"].mean()),
            "legal_path_macro_f1": float(frame["legal_path_macro_f1"].mean()),
            "inner_oof_ez_auprc": float(frame["legal_path_ez_auprc"].mean()),
            "oracle_minus_legal_path_gap": float((frame["patient_ranking_oracle_macro_f1"] - frame["legal_path_macro_f1"]).mean()),
            "n_non_decreasing_outer_folds_vs_p1": np.nan,
            "note": "All values use outer-train inner-OOF ranking records and deterministic PATH validation patients only.",
            **center_means,
        })
    gap = pd.DataFrame(gap_rows)
    if set(gap.loc[gap.status == "complete", "profile"]) == {"P1", "P2"}:
        p1 = gap[gap.profile == "P1"].iloc[0]
        for metric in (
            "ranking_oracle_macro_f1", "legal_path_macro_f1", "inner_oof_ez_auprc",
            "hup_legal_path_macro_f1", "lzu_legal_path_macro_f1",
            "multicenter_legal_path_macro_f1", "pediatric_legal_path_macro_f1",
        ):
            delta = float(gap.loc[gap.profile == "P2", metric].iloc[0] - p1[metric])
            gap.loc[gap.profile == "P2", f"delta_{metric}_p2_minus_p1"] = delta
        p1_fold = pd.read_csv(root / "P1" / "seed_42" / "inner_oof_summary_by_outer_fold.csv")
        p2_fold = pd.read_csv(root / "P2" / "seed_42" / "inner_oof_summary_by_outer_fold.csv")
        aligned = p1_fold.merge(p2_fold, on="outer_fold", suffixes=("_p1", "_p2"), validate="one_to_one")
        non_decreasing = int((aligned.legal_path_macro_f1_p2 >= aligned.legal_path_macro_f1_p1).sum())
        gap.loc[gap.profile == "P2", "n_non_decreasing_outer_folds_vs_p1"] = non_decreasing
    gap.to_csv(destination / "oracle_gap_recovery.csv", index=False)

    audit_dir = Path(causal_audit_dir) if causal_audit_dir else None
    copies = {
        "causal_feature_redundancy_by_fold.csv": "causal_feature_redundancy.csv",
        "causal_feature_center_predictability.csv": "causal_feature_center_predictability.csv",
        "causal_feature_stability.csv": "causal_feature_stability.csv",
    }
    for source_name, output_name in copies.items():
        source = audit_dir / source_name if audit_dir else None
        if source is not None and source.is_file():
            shutil.copyfile(source, destination / output_name)
        else:
            pd.DataFrame([{"status": "missing", "source": str(source or "")}]).to_csv(destination / output_name, index=False)
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description="Create fixed CANE-PATH-CP profile and audit comparisons.")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--causal-audit-dir")
    args = parser.parse_args()
    print(summarize(args.run_root, args.causal_audit_dir))


if __name__ == "__main__":
    main()
