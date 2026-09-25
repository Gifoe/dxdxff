from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import json_safe
from neuroez_c.kcalibration import train_predict_kcal


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train fold-safe K calibration for CleanNEZ SetTopo ledgers.")
    parser.add_argument("--settopo-ledger", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--k-min", type=int, default=1)
    parser.add_argument("--k-max", type=int, default=40)
    parser.add_argument(
        "--model",
        choices=["ridge", "poisson", "ensemble_ridge_poisson", "ridge_poisson"],
        default="ensemble_ridge_poisson",
    )
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--allowed-subjects-file", default=None)
    parser.add_argument("--require-n-patients", type=int, default=None)
    parser.add_argument("--label-encoding-mode", choices=["ez1", "ez0"], default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, audit = train_predict_kcal(
        args.settopo_ledger,
        args.output_dir,
        k_min=args.k_min,
        k_max=args.k_max,
        model=args.model,
        allowed_subjects_ledger=args.allowed_subjects_ledger,
        allowed_subjects_file=args.allowed_subjects_file,
        require_n_patients=args.require_n_patients,
        label_encoding_mode=args.label_encoding_mode,
    )
    print(json.dumps(json_safe(audit), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
