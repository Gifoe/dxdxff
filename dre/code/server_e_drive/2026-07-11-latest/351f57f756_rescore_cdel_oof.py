#!/usr/bin/env python3
"""Recompute formal CDEL OOF results from frozen PRQ and BCR ledgers only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from task1_confirmatory.evaluate import evaluate_pair


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prq-root", required=True, help="Completed PRQ-Net OOF output root")
    parser.add_argument("--bcr-root", required=True, help="Completed final BCR BC-only OOF output root")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--folds", default="1,2,3,4,5")
    args = parser.parse_args()
    folds = [int(value) for value in args.folds.split(",") if value.strip()]
    result = evaluate_pair(
        p2_root=args.prq_root,
        v3_root=args.bcr_root,
        folds=folds,
        output_dir=args.output_dir,
        analysis_status="OOF_RESCORE_FINAL_CDEL_NO_RETRAIN",
    )
    (Path(args.output_dir) / "CDEL_OOF_RESCORE_AUDIT.json").write_text(
        json.dumps({
            **result,
            "formula": "p_cdel_nez = 0.80 * p_prq_nez + 0.20 * (1 - sigmoid(e_bcr))",
            "threshold_protocol": "outer_fold_validation_only_grid_step_0.005",
            "training_performed": False,
        }, indent=2), encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
