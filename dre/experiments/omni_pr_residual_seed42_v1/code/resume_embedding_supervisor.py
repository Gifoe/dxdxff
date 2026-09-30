"""Bounded engineering recovery for native Windows/CUDA extractor exits."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from common import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--extractor", type=Path, required=True)
    parser.add_argument("--mode", choices=("train", "test"), required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--official-cnn", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--public-audit", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--maximum-attempts", type=int, default=200)
    args = parser.parse_args()
    expected = 296 if args.mode == "train" else 237
    status_path = args.runtime / f"{args.mode}_embedding_supervisor_status.json"
    no_progress = 0
    for attempt in range(1, args.maximum_attempts + 1):
        before = len(list(args.output.glob("*.npz")))
        if before == expected:
            atomic_json(status_path, {"status": "COMPLETE", "mode": args.mode,
                                      "completed_files": before, "attempts": attempt - 1})
            return
        stdout = args.runtime / f"{args.mode}_embedding_attempt_{attempt:03d}.log"
        stderr = args.runtime / f"{args.mode}_embedding_attempt_{attempt:03d}.err"
        command = [str(args.python), str(args.extractor), "--mode", args.mode,
                   "--source", str(args.source), "--official-cnn", str(args.official_cnn),
                   "--checkpoint", str(args.checkpoint), "--protocol", str(args.protocol),
                   "--output", str(args.output), "--public-audit", str(args.public_audit),
                   "--batch-size", str(args.batch_size)]
        atomic_json(status_path, {"status": "RUNNING", "mode": args.mode,
                                  "attempt": attempt, "completed_files": before})
        with stdout.open("w", encoding="utf-8") as out, stderr.open("w", encoding="utf-8") as err:
            process = subprocess.run(command, stdout=out, stderr=err)
        after = len(list(args.output.glob("*.npz")))
        if after == expected and process.returncode == 0:
            atomic_json(status_path, {"status": "COMPLETE", "mode": args.mode,
                                      "completed_files": after, "attempts": attempt})
            return
        if stderr.stat().st_size:
            atomic_json(status_path, {"status": "PYTHON_ERROR", "mode": args.mode,
                                      "attempt": attempt, "returncode": process.returncode,
                                      "completed_files": after})
            raise RuntimeError(f"Extractor wrote Python stderr on attempt {attempt}")
        no_progress = no_progress + 1 if after == before else 0
        atomic_json(status_path, {"status": "NATIVE_EXIT_RESUMABLE", "mode": args.mode,
                                  "attempt": attempt, "returncode": process.returncode,
                                  "before": before, "after": after,
                                  "consecutive_no_progress": no_progress})
        if no_progress >= 3:
            raise RuntimeError("Three consecutive native exits without progress")
        time.sleep(2)
    raise RuntimeError("Embedding supervisor exhausted bounded attempts")


if __name__ == "__main__":
    main()
