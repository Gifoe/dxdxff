from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m1_step1_ablation_common import collect_results_rows, write_results_csv


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect M1 Step 1 ablation results into one CSV.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = collect_results_rows(args.output_root)
    frame = write_results_csv(rows, args.out_csv)
    if frame.empty:
        print("No experiment results found.")
        return
    columns = [
        "experiment_name",
        "pooled_macro_f1",
        "patient_macro_f1",
        "pooled_auprc_ez",
        "pooled_auroc_ez",
        "physics_gate_init",
        "physics_state_features",
        "physics_loss_weight",
    ]
    available = [column for column in columns if column in frame.columns]
    print(frame.loc[:, available].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
