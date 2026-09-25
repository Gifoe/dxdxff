#!/usr/bin/env python3
"""Generate patient-level bootstrap CIs from completed Task 1 LOCO outputs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from task1_confirmatory.loco_bootstrap import write_loco_bootstrap_report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Existing Task 1 completion root containing loco/.")
    parser.add_argument("--output-dir", default="", help="Defaults to <root>/statistics.")
    parser.add_argument("--seeds", default="42,52,62")
    parser.add_argument("--bootstrap-repeats", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    output = Path(args.output_dir) if args.output_dir else Path(args.root) / "statistics"
    print(json.dumps(write_loco_bootstrap_report(root=args.root, output_dir=output, seeds=seeds, repeats=args.bootstrap_repeats, seed=args.seed), indent=2))


if __name__ == "__main__":
    main()
