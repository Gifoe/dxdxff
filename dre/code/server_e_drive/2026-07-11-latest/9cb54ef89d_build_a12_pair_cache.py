"""Build a versioned A12 pair cache from a frozen V3 ledger without raw extraction."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from a12_vcsn.candidate_pool import CandidateConfig, build_candidate_pools
from a12_vcsn.io import load_v3_ledger
from a12_vcsn.pair_dataset import build_pair_dataset, write_pair_cache


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-v3-ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--boundary-width", type=int, default=20)
    parser.add_argument("--selected-tail-frac", type=float, default=0.30)
    parser.add_argument("--selected-tail-min", type=int, default=3)
    parser.add_argument("--selected-tail-max", type=int, default=12)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    ledger, _ = load_v3_ledger(args.old_v3_ledger, strict=True)
    config = CandidateConfig(args.selected_tail_frac, args.selected_tail_min, args.selected_tail_max, args.boundary_width)
    pairs = build_pair_dataset(ledger, build_candidate_pools(ledger, config), include_labels=True)
    result = write_pair_cache(pairs, args.output_dir, config=config.__dict__)
    print(json.dumps({"n_pairs": len(pairs), **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
