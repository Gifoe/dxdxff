#!/usr/bin/env python3
"""Fail if formal CDEL files still advertise the retired 0.90/0.10 default."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


FORMAL_FILES = (
    "neuroez_c/p2_v3_conservative_fusion.py",
    "task1_confirmatory/evaluate.py",
    "task1_confirmatory/final_summary.py",
    "task1_confirmatory/reporting.py",
    "task1_confirmatory/orchestrator.py",
    "configs/task1_final_confirmatory_v2.yaml",
)
RETIRED = ("0.90", "0.10", "fixed_90_10", "A3_QBC_FULL")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--output", default="")
    args = parser.parse_args()
    root = Path(args.repo)
    findings = []
    for relative in FORMAL_FILES:
        path = root / relative
        text = path.read_text(encoding="utf-8") if path.is_file() else ""
        for token in RETIRED:
            if token in text:
                findings.append({"file": relative, "token": token})
    report = {"status": "passed" if not findings else "failed", "formal_files": list(FORMAL_FILES), "retired_defaults": findings}
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if findings:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
