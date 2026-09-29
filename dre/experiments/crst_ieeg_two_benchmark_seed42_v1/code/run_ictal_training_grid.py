"""Sequential 5-fold, two-variant train/validation-only grid with resume."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--spectral-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--feature-cache", type=Path, required=True)
    args = p.parse_args()
    for fold in range(1, 6):
        for variant in ("CRST-0", "CRST-FULL"):
            target = args.runtime / "ictal" / f"fold{fold}" / variant / "supervised_complete.json"
            if target.is_file():
                print(json.dumps({"fold": fold, "variant": variant, "status": "reused"}), flush=True)
                continue
            argv = [sys.executable, str(args.code / "train_crst.py"),
                    "--benchmark", "ictal", "--fold", str(fold),
                    "--variant", variant, "--spectral-cache", str(args.spectral_cache),
                    "--manifest", str(args.manifest), "--feature-cache", str(args.feature_cache),
                    "--protocol", str(args.code / "PROTOCOL_LOCK.json"),
                    "--training-lock", str(args.code / "TRAINING_LOCK.json"),
                    "--runtime", str(args.runtime)]
            log = args.runtime / f"ictal_fold{fold}_{variant.replace('-', '')}.log"
            error = args.runtime / f"ictal_fold{fold}_{variant.replace('-', '')}.err"
            with log.open("a", encoding="utf-8") as stdout, error.open("a", encoding="utf-8") as stderr:
                exit_code = subprocess.call(argv, stdout=stdout, stderr=stderr)
            if exit_code != 0 or not target.is_file():
                raise RuntimeError(f"Ictal fold{fold}/{variant} failed; inspect private logs; exit={exit_code}")
            print(json.dumps({"fold": fold, "variant": variant, "status": "complete",
                              "utc_epoch": time.time()}), flush=True)
    print(json.dumps({"status": "ALL_ICTAL_TRAIN_VALIDATION_COMPLETE",
                      "outer_test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
