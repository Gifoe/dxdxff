from __future__ import annotations

import argparse
from pathlib import Path
import sys

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import pandas as pd

from task1_baselines.fold_protocol import freeze_v3_fold_manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3-ledger", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    freeze_v3_fold_manifest(pd.read_csv(args.v3_ledger)).to_csv(args.output, index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
