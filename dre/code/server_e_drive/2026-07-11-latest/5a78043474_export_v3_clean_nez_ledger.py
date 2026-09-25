from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from neuroez_c.clean_nez_utils import export_v3_clean_nez_ledger, json_safe


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export V3 fold CSVs into a clean-NEZ OOF channel ledger.")
    parser.add_argument("--v3-output-dir", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--fold-start", type=int, default=1)
    parser.add_argument("--fold-end", type=int, default=5)
    parser.add_argument("--allowed-subjects-ledger", default=None)
    parser.add_argument("--allowed-subjects-file", default=None)
    parser.add_argument("--require-n-patients", type=int, default=None)
    parser.add_argument("--label-encoding-mode", choices=["ez1", "ez0"], default="ez1")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, audit = export_v3_clean_nez_ledger(
        args.v3_output_dir,
        args.output_path,
        fold_start=args.fold_start,
        fold_end=args.fold_end,
        allowed_subjects_ledger=args.allowed_subjects_ledger,
        allowed_subjects_file=args.allowed_subjects_file,
        require_n_patients=args.require_n_patients,
        label_encoding_mode=args.label_encoding_mode,
    )
    print(json.dumps(json_safe({"output_path": args.output_path, **audit}), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
