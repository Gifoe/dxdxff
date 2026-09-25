from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m1_step1_ablation_common import summarize_multiseed_decision


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Report the final multiseed M1 decision.")
    parser.add_argument("--summary-csv", type=str, required=True)
    parser.add_argument("--current-m1-pooled-auprc-ez", type=float, default=0.3603683291)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = pd.read_csv(args.summary_csv)
    if frame.empty:
        raise SystemExit("Summary CSV is empty.")
    decision = summarize_multiseed_decision(
        frame.to_dict(orient="records"),
        current_m1_pooled_auprc_ez=args.current_m1_pooled_auprc_ez,
    )
    print(f"selected_candidate: {decision['selected_candidate']}")
    print(f"selected_mean_pooled_macro_f1: {decision['selected_mean_pooled_macro_f1']:.10f}")
    print(f"selected_mean_pooled_auprc_ez: {decision['selected_mean_pooled_auprc_ez']:.10f}")
    print(f"reason: {decision['reason']}")
    print(f"conclusion: {decision['conclusion']}")
    print(decision["step"])


if __name__ == "__main__":
    main()
