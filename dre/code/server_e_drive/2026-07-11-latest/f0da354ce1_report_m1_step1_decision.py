from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m1_step1_ablation_common import summarize_decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report the Step 1 M1 ablation decision.")
    parser.add_argument("--results-csv", type=str, required=True)
    parser.add_argument("--b0-pooled-macro-f1", type=float, required=True)
    parser.add_argument("--m1-current-pooled-macro-f1", type=float, required=True)
    parser.add_argument("--m1-current-pooled-auprc-ez", type=float, default=0.3603683291)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_csv(args.results_csv)
    if frame.empty:
        raise SystemExit("Results CSV is empty.")
    best_row = frame.iloc[0].to_dict()
    summary = summarize_decision(
        best_row=best_row,
        b0_pooled_macro_f1=args.b0_pooled_macro_f1,
        current_m1_pooled_macro_f1=args.m1_current_pooled_macro_f1,
        current_m1_pooled_auprc_ez=args.m1_current_pooled_auprc_ez,
    )
    print(f"best_experiment: {summary['best_experiment']}")
    print(f"best_pooled_macro_f1: {summary['best_pooled_macro_f1']:.10f}")
    print(f"best_patient_macro_f1: {summary['best_patient_macro_f1']:.10f}")
    print(f"best_pooled_auprc_ez: {summary['best_pooled_auprc_ez']:.10f}")
    print(f"best_pooled_auroc_ez: {summary['best_pooled_auroc_ez']:.10f}")
    print(f"delta_vs_b0: {summary['delta_vs_b0']:.10f}")
    print(f"delta_vs_current_m1: {summary['delta_vs_current_m1']:.10f}")
    print(f"conclusion: {summary['conclusion']}")


if __name__ == "__main__":
    main()
