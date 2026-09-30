"""Sequential PRiSM-EZ development runner.  It exposes no test arguments."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("ictal", "omni"), required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--ictal-cache", type=Path); parser.add_argument("--ictal-manifest", type=Path)
    parser.add_argument("--omni-cache", type=Path); parser.add_argument("--omni-official-split", type=Path)
    parser.add_argument("--omni-inner-split", type=Path)
    args = parser.parse_args()
    common = ["--benchmark", args.benchmark, "--protocol", str(args.protocol), "--runtime", str(args.runtime),
              "--feature-cache", str(args.feature_cache)]
    if args.benchmark == "ictal":
        common += ["--ictal-cache", str(args.ictal_cache), "--ictal-manifest", str(args.ictal_manifest)]
        folds = range(1, 6)
    else:
        common += ["--omni-cache", str(args.omni_cache), "--omni-official-split", str(args.omni_official_split),
                   "--omni-inner-split", str(args.omni_inner_split)]
        folds = (1,)
    for fold in folds:
        marker = args.runtime / args.benchmark / f"fold{fold}" / "PRISM_VALIDATION_SELECTION.json"
        if marker.exists() and json.loads(marker.read_text(encoding="utf-8")).get("status") == "TRAIN_VALIDATION_COMPLETE":
            print(json.dumps({"status": "REUSED_COMPLETE", "benchmark": args.benchmark, "fold": fold}), flush=True)
            continue
        subprocess.run([sys.executable, str(args.code / "train_prism.py"), "--fold", str(fold), *common], check=True)
    print(json.dumps({"status": "ALL_DEVELOPMENT_COMPLETE", "benchmark": args.benchmark, "test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
