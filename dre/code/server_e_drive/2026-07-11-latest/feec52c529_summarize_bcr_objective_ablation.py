#!/usr/bin/env python3
"""Summarize BCR objective ablations and their locked PRQ+BCR fusions.

This is ledger-only: every CDEL row uses the existing formal evaluator with
the locked PRQ/BCR probability weights (0.80/0.20) and a threshold selected
from that fold's validation patients only.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from task1_confirmatory.component_ablation import BCR_PROFILES, collect_reports
from task1_confirmatory.evaluate import evaluate_pair


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, help="Component-ablation output root.")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--prq-profile", default="P2_TEMPORAL_Q10")
    parser.add_argument(
        "--prq-root-template",
        default="",
        help="Optional root template containing {seed}; defaults to <root>/prq/<profile>/seed_{seed}.",
    )
    return parser


def _mean_std(rows: pd.DataFrame, *, group_columns: list[str]) -> pd.DataFrame:
    metrics = [
        "patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_accuracy",
        "patient_ez_auroc", "patient_ez_auprc", "patient_ez_ndcg", "ez_fraction_bias",
    ]
    available = [metric for metric in metrics if metric in rows.columns]
    grouped = rows.groupby(group_columns, sort=False)[available]
    return grouped.mean().add_suffix("_mean").join(grouped.std(ddof=1).add_suffix("_std")).reset_index()


def _format(mean_std: pd.DataFrame) -> pd.DataFrame:
    entries = []
    for _, row in mean_std.iterrows():
        def metric(name: str) -> str:
            return f"{row[f'{name}_mean']:.4f} +/- {row[f'{name}_std']:.4f}"
        entries.append({
            "Method": row["Method"],
            "Macro-F1": metric("patient_macro_f1"),
            "EZ-F1": metric("patient_ez_f1"),
            "Accuracy": metric("patient_accuracy"),
            "AUROC": metric("patient_ez_auroc"),
            "EZ-AUPRC": metric("patient_ez_auprc"),
            "NDCG-EZ": metric("patient_ez_ndcg"),
            "EZ-Frac. Bias": metric("ez_fraction_bias"),
        })
    return pd.DataFrame(entries)


def main() -> None:
    args = _parser().parse_args()
    root = Path(args.root)
    seeds = [int(item) for item in args.seeds.split(",") if item.strip()]
    if not seeds or seeds != sorted(set(seeds)):
        raise ValueError("--seeds must be a unique sorted comma-separated list")
    audit_path = root / "audit" / "component_preflight.json"
    if not audit_path.is_file():
        raise FileNotFoundError(f"Missing component-ablation audit: {audit_path}")
    audit = __import__("json").loads(audit_path.read_text(encoding="utf-8"))

    # Rebuild the standalone four-objective table from the shared evaluator.
    collect_reports(
        root,
        profiles={"bcr": list(BCR_PROFILES)},
        seeds=seeds,
        repo=REPO,
        audit=audit,
    )

    rows = []
    for seed in seeds:
        if args.prq_root_template:
            prq_root = Path(args.prq_root_template.format(seed=seed))
        else:
            prq_root = root / "prq" / args.prq_profile / f"seed_{seed}"
        if not prq_root.is_dir():
            raise FileNotFoundError(f"Missing formal PRQ root for seed {seed}: {prq_root}")
        for profile, spec in BCR_PROFILES.items():
            bcr_root = root / "bcr" / profile / f"seed_{seed}"
            if not bcr_root.is_dir():
                raise FileNotFoundError(f"Missing BCR objective run: {bcr_root}")
            destination = root / "bcr_objective_fusion" / profile / f"seed_{seed}"
            evaluate_pair(
                p2_root=prq_root,
                v3_root=bcr_root,
                folds=range(1, 6),
                output_dir=destination,
                analysis_status="BCR_OBJECTIVE_ABLATION_LOCKED_80_20",
                require_provenance=False,
            )
            overall = pd.read_csv(destination / "metrics" / "overall.csv")
            cdel = overall.loc[overall.experiment.eq("CDEL")].copy()
            if len(cdel) != 1:
                raise RuntimeError(f"Expected one CDEL summary for {profile}/seed_{seed}")
            cdel["Method"] = f"PRQ-Full + {spec['display_name']}"
            cdel["profile"] = profile
            cdel["seed"] = seed
            rows.append(cdel)
            if profile == "BCR_BC_ONLY":
                prq = overall.loc[overall.experiment.eq("PRQ-Net")].copy()
                if len(prq) != 1:
                    raise RuntimeError(f"Expected one PRQ-Net summary for seed {seed}")
                prq["Method"] = "Single PRQ-Full"
                prq["profile"] = "PRQ_FULL"
                prq["seed"] = seed
                rows.append(prq)
    by_seed = pd.concat(rows, ignore_index=True)
    mean_std = _mean_std(by_seed, group_columns=["profile", "Method"])
    paper = _format(mean_std)
    report = root / "reports"
    by_seed.to_csv(report / "bcr_prq_fusion_ablation_by_seed.csv", index=False)
    mean_std.to_csv(report / "bcr_prq_fusion_ablation_mean_std.csv", index=False)
    paper.to_csv(report / "bcr_prq_fusion_ablation_paper_table.csv", index=False)
    (report / "bcr_prq_fusion_ablation_paper_table.tex").write_text(
        paper.to_latex(index=False, escape=True, column_format="lrrrrrrr"), encoding="utf-8"
    )
    (report / "BCR_OBJECTIVE_ABLATION_REPORT.md").write_text(
        "# BCR Objective Ablation\n\n"
        "All BCR variants are independently trained with Q10 disabled. "
        "CDEL rows use fixed probability weights PRQ=0.80 and BCR=0.20; "
        "each model threshold is selected on its outer-fold validation patients only.\n\n"
        + paper.to_markdown(index=False) + "\n",
        encoding="utf-8",
    )
    print({"status": "passed", "report_dir": str(report), "n_fusion_runs": len(rows) - len(seeds)})


if __name__ == "__main__":
    main()
