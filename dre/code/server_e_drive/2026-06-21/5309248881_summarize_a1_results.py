from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

METRICS = [
    "patient_macro_f1",
    "patient_macro_ez_f1",
    "patient_macro_auroc_ez",
    "patient_macro_auprc_ez",
    "patient_macro_ez_mrr",
    "pooled_macro_f1",
    "pooled_auroc_ez",
    "pooled_auprc_ez",
]


def load_summary(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    df = pd.read_csv(path)
    if df.empty:
        return {}
    return df.iloc[0].to_dict()


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize A1 ranking-loss experiments.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--baseline_summary", type=Path, default=None)
    parser.add_argument("--out_csv", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    for summary_path in sorted(args.root.glob("*/heldout_summary_neuroez_v3.json")):
        run_dir = summary_path.parent
        summary = load_summary(summary_path)
        args_path = run_dir / "run_args_b0_pruned.json"
        run_args = load_summary(args_path) if args_path.exists() else {}
        row = {"config": run_dir.name, "output_dir": str(run_dir)}
        for key in METRICS:
            row[key] = summary.get(key, float("nan"))
        for key in ["ez_ranking_loss_type", "ez_ranking_loss_weight", "ez_ranking_margin", "ez_ranking_tau"]:
            row[key] = run_args.get(key, None)
        rows.append(row)

    df = pd.DataFrame(rows)
    if df.empty:
        raise FileNotFoundError(f"No heldout_summary_neuroez_v3.json files found below {args.root}")

    if args.baseline_summary is not None and args.baseline_summary.exists():
        baseline = load_summary(args.baseline_summary)
        base_f1 = float(baseline.get("patient_macro_f1", float("nan")))
        base_ez_f1 = float(baseline.get("patient_macro_ez_f1", float("nan")))
        df["delta_patient_macro_f1_vs_baseline"] = df["patient_macro_f1"] - base_f1
        df["delta_patient_macro_ez_f1_vs_baseline"] = df["patient_macro_ez_f1"] - base_ez_f1

    df = df.sort_values("patient_macro_f1", ascending=False)
    out_csv = args.out_csv or (args.root / "a1_ranking_grid_summary.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string(index=False))
    print("[A1 summary] wrote", out_csv)


if __name__ == "__main__":
    main()
