#!/usr/bin/env python3
"""Audit the frozen ablation inputs without running any ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from P2_V3_AAAI_ABLATIONS.input_loader import load_base_folds
from P2_V3_AAAI_ABLATIONS.reporting import write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_manifest", required=True); parser.add_argument("--allowed_subjects_ledger", required=True); parser.add_argument("--fixed_fold_manifest", required=True)
    parser.add_argument("--require_n_patients", type=int, default=80); parser.add_argument("--expected_n_channels", type=int, default=7635); parser.add_argument("--output", required=True); parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()
    _, _, audit = load_base_folds(args.input_manifest, allowed_subjects_ledger=args.allowed_subjects_ledger, fixed_fold_manifest=args.fixed_fold_manifest, require_n_patients=args.require_n_patients, expected_n_channels=args.expected_n_channels, strict=args.strict)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True); write_json(args.output, audit)
    print(json.dumps({"status": audit["status"], "n_patients": audit["n_patients"], "n_channels": audit["n_test_channels"], "n_folds": audit["n_outer_folds"]}, indent=2))


if __name__ == "__main__": main()

