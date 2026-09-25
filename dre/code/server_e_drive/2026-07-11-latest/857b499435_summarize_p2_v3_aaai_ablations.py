#!/usr/bin/env python3
"""Validate and print the completed P2/V3 ablation artifacts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(REPO))
from P2_V3_AAAI_ABLATIONS.reporting import grouped_bar_figure


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--ablation_root", required=True); parser.add_argument("--bootstrap_repeats", type=int, default=2000); parser.add_argument("--seed", type=int, default=42); args = parser.parse_args()
    root = Path(args.ablation_root)
    required = [
        root / "audit" / "ablation_reproduction_audit.json", root / "metrics" / "branch_ablation_overall.csv",
        root / "metrics" / "weight_ablation_overall.csv", root / "metrics" / "fusion_operator_overall.csv",
        root / "metrics" / "threshold_decomposition_overall.csv", root / "metrics" / "shuffled_v3_control_summary.csv",
        root / "metrics" / "paired_bootstrap.csv", root / "reports" / "P2_V3_AAAI_ABLATION_REPORT.md",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing: raise FileNotFoundError(f"Incomplete ablation output: {missing}")
    reproduction = json.loads(required[0].read_text(encoding="utf-8"))
    if reproduction.get("status") != "passed": raise RuntimeError("Reproduction gate did not pass")
    branch = pd.read_csv(root / "metrics" / "branch_ablation_overall.csv")
    grouped_bar_figure(pd.read_csv(root / "metrics" / "branch_ablation_by_fold.csv"), x="outer_fold", group="experiment", y="patient_macro_f1", output_stem=root / "figures" / "fold_paired_comparison", ylabel="Patient Macro-F1")
    grouped_bar_figure(pd.read_csv(root / "metrics" / "branch_ablation_by_center.csv"), x="center", group="experiment", y="patient_macro_f1", output_stem=root / "figures" / "center_paired_comparison", ylabel="Patient Macro-F1")
    print(json.dumps({"status": "passed", "reproduction": reproduction["status"], "branch_results": branch[["experiment", "patient_macro_f1"]].to_dict("records"), "report": str(root / "reports" / "P2_V3_AAAI_ABLATION_REPORT.md")}, indent=2))


if __name__ == "__main__": main()
