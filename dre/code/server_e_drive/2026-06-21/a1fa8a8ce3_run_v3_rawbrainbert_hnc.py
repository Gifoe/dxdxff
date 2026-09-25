from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neuroez_c.v3_hnc_features import run_hnc_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run V3-RawBrainBERT-HNC hard-negative correction.")
    parser.add_argument("--v3-oof-ledger", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--feature-cache-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--candidate-rule",
        choices=["top20pct", "top12", "top8", "top20pct_plus_neighbors"],
        default="top20pct_plus_neighbors",
    )
    parser.add_argument("--beta-list", default="0.05,0.10,0.20,0.30")
    parser.add_argument("--pca-dim", type=int, default=32)
    parser.add_argument("--classifier", choices=["l2_logreg", "elastic_logreg"], default="l2_logreg")
    parser.add_argument("--split-strategy", default="5fold")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--logit-eps", type=float, default=1e-5)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    audit = run_hnc_pipeline(args)
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
