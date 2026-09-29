"""Predeclared B1/B2 validation-only ablation grid; no test access.

Both ablations repeat the train-only SSL and supervised recipe with the same
model parameter topology. They are not selected or changed using test results.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=("ictal", "omni"), required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--spectral-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--feature-cache", type=Path)
    args = p.parse_args()
    if args.benchmark == "ictal" and (not args.manifest or not args.feature_cache):
        raise RuntimeError("Frozen ictal manifest and feature metadata required")
    if args.benchmark == "omni":
        audit = json.loads((args.runtime / "OMNI_TRAIN_CACHE_AUDIT.json").read_text())
        if not audit["pass"] or audit["test_split_accessed"]:
            raise RuntimeError("Omni TRAIN cache audit not clean")
    for ablation in ("A_ONLY", "NO_CHANNEL_ATTENTION"):
        for fold in (range(1, 6) if args.benchmark == "ictal" else [1]):
            work = (args.runtime / args.benchmark / f"fold{fold}" / "ablation" /
                    ablation / "CRST-FULL")
            if (work / "supervised_complete.json").is_file():
                print(json.dumps({"ablation": ablation, "fold": fold,
                                  "status": "reused"}), flush=True)
                continue
            argv = [sys.executable, str(args.code / "train_crst.py"),
                    "--benchmark", args.benchmark, "--fold", str(fold),
                    "--variant", "CRST-FULL", "--ablation", ablation,
                    "--spectral-cache", str(args.spectral_cache),
                    "--protocol", str(args.code / "PROTOCOL_LOCK.json"),
                    "--training-lock", str(args.code / "TRAINING_LOCK.json"),
                    "--runtime", str(args.runtime)]
            if args.benchmark == "ictal":
                argv += ["--manifest", str(args.manifest),
                         "--feature-cache", str(args.feature_cache)]
            else:
                argv += ["--train-val-split", str(args.code / "TRAIN_VAL_SPLIT.csv")]
            log = args.runtime / f"ablation_{args.benchmark}_{ablation}_fold{fold}.log"
            error = args.runtime / f"ablation_{args.benchmark}_{ablation}_fold{fold}.err"
            with log.open("a", encoding="utf-8") as out, error.open("a", encoding="utf-8") as err:
                exit_code = subprocess.call(argv, stdout=out, stderr=err)
            if exit_code or not (work / "supervised_complete.json").is_file():
                raise RuntimeError(f"Ablation {ablation} fold{fold} failed, exit {exit_code}")
            print(json.dumps({"ablation": ablation, "fold": fold,
                              "status": "complete", "test_accessed": False}), flush=True)
    print(json.dumps({"status": "VALIDATION_ONLY_ABLATIONS_COMPLETE",
                      "benchmark": args.benchmark, "test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
