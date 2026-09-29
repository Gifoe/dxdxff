"""Sequential, resume-safe TRAIN/validation orchestration; no test access.

This process must start only when no independent B0/PC-CNN process is already
training on the same GPU. Each completed unit is verified before proceeding.
An engineering failure stops the chain without deleting or overwriting the
last/best checkpoints; it does not retry or adjust the scientific protocol.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from train_rawcnn import digest


def complete(path: Path, protocol_sha: str, benchmark: str, fold: int):
    if not path.is_file():
        return False
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj["status"] != "TRAIN_VALIDATION_COMPLETE" or \
            obj["protocol_sha256"] != protocol_sha or \
            obj["benchmark"] != benchmark or obj["fold"] != fold or \
            obj["test_accessed"]:
        raise RuntimeError(f"Unexpected completion marker: {path}")
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=("ictal", "omni"), required=True)
    p.add_argument("--code", type=Path, required=True)
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--ictal-cache", type=Path)
    p.add_argument("--ictal-manifest", type=Path)
    p.add_argument("--omni-cache", type=Path)
    p.add_argument("--omni-official-split", type=Path)
    p.add_argument("--omni-inner-split", type=Path)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    protocol_sha = digest(args.protocol)
    shared = ["--benchmark", args.benchmark, "--official-cnn", str(args.official_cnn),
              "--protocol", str(args.protocol), "--runtime", str(args.runtime)]
    if args.benchmark == "ictal":
        shared += ["--ictal-cache", str(args.ictal_cache),
                   "--ictal-manifest", str(args.ictal_manifest)]
        folds = range(1, 6)
    else:
        shared += ["--omni-cache", str(args.omni_cache),
                   "--omni-official-split", str(args.omni_official_split),
                   "--omni-inner-split", str(args.omni_inner_split)]
        folds = (1,)
    for fold in folds:
        work = args.runtime / args.benchmark / f"fold{fold}"
        for stage, script, marker in (
            ("B0", "train_rawcnn.py", work / "B0_RawCNN" / "RAW_B0_SELECTION.json"),
            ("PC", "train_pccnn.py", work / "PC_VALIDATION_SELECTION.json"),
        ):
            if complete(marker, protocol_sha, args.benchmark, fold):
                print(json.dumps({"status": "REUSED_COMPLETE", "benchmark": args.benchmark,
                                  "fold": fold, "stage": stage}), flush=True)
                continue
            command = [sys.executable, str(args.code / script), *shared,
                       "--fold", str(fold)]
            print(json.dumps({"status": "STARTING", "benchmark": args.benchmark,
                              "fold": fold, "stage": stage}), flush=True)
            subprocess.run(command, check=True)
            if not complete(marker, protocol_sha, args.benchmark, fold):
                raise RuntimeError(f"Stage returned without valid completion marker: {marker}")
    print(json.dumps({"status": "ALL_TRAIN_VALIDATION_COMPLETE",
                      "benchmark": args.benchmark, "test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
