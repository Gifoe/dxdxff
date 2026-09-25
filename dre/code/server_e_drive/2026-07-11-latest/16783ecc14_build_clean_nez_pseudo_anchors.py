from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import build_clean_nez_distance_ledger, json_safe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build pseudo clean-NEZ anchors and RawBrainBERT NEZ-distance ledger.")
    parser.add_argument("--v3-ledger", required=True)
    parser.add_argument("--embedding-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--anchor-strategy", choices=["quantile", "topk", "threshold"], default="quantile")
    parser.add_argument("--anchor-quantile", type=float, default=0.70)
    parser.add_argument("--min-anchors", type=int, default=5)
    parser.add_argument("--anchor-topk", type=int, default=None)
    parser.add_argument("--anchor-threshold", type=float, default=None)
    parser.add_argument("--distance-modes", default="cosine,euclidean")
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--allowed-subjects-file", default=None)
    parser.add_argument("--require-n-patients", type=int, default=None)
    parser.add_argument("--label-encoding-mode", choices=["ez1", "ez0"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    modes = [part.strip() for part in str(args.distance_modes).split(",") if part.strip()]
    _, audit = build_clean_nez_distance_ledger(
        args.v3_ledger,
        args.embedding_dir,
        args.output_dir,
        anchor_strategy=args.anchor_strategy,
        anchor_quantile=args.anchor_quantile,
        min_anchors=args.min_anchors,
        anchor_topk=args.anchor_topk,
        anchor_threshold=args.anchor_threshold,
        distance_modes=modes,
        allowed_subjects_ledger=args.allowed_subjects_ledger,
        allowed_subjects_file=args.allowed_subjects_file,
        require_n_patients=args.require_n_patients,
        label_encoding_mode=args.label_encoding_mode,
    )
    print(json.dumps(json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
