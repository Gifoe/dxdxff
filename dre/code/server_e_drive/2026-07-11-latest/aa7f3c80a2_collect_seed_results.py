from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.m1_step1_ablation_common import collect_seed_summary_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect one seed-sweep root into a summary CSV.")
    parser.add_argument("--output-root", type=str, required=True)
    parser.add_argument("--out-csv", type=str, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    frame = collect_seed_summary_frame(args.output_root)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out_csv, index=False)
    if frame.empty:
        print("No seed results found.")
        return
    print(frame.to_string(index=False))


if __name__ == "__main__":
    main()
