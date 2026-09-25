"""List CSV files that are valid frozen Task 1 patient-fold ledgers."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from task1_baselines.fold_protocol import Task1ProtocolError, freeze_v3_fold_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--roots", nargs="+", required=True)
    parser.add_argument("--expected-subjects", type=int, default=80)
    args = parser.parse_args()
    matches: list[Path] = []
    for root_text in args.roots:
        root = Path(root_text)
        if not root.exists():
            continue
        for path in root.rglob("*.csv"):
            try:
                manifest = freeze_v3_fold_manifest(pd.read_csv(path), expected_subjects=args.expected_subjects)
            except (OSError, UnicodeDecodeError, pd.errors.ParserError, Task1ProtocolError, ValueError):
                continue
            print(f"VALID\t{path}\tpatients={len(manifest)}\tfolds={','.join(map(str, sorted(manifest.outer_fold.unique())))}")
            matches.append(path)
    if not matches:
        print("No valid frozen ledger found. A valid file needs subject_id, center, and outer_fold/fold_idx; each of 80 subjects must appear in exactly one fold.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
