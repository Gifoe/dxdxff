"""Resumable sequential Stage-2 launcher for frozen-gate eligible I1 folds."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def complete(runtime: Path, fold: int) -> bool:
    path = runtime / f"fold_{fold}/stage2/summary.json"
    if not path.is_file():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    return row.get("fold") == fold and row.get("stage2_complete") is True and row.get("epochs") == 15


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    a = parser.parse_args()
    gate = json.loads(a.gate_audit.read_text(encoding="utf-8"))
    eligible = gate["eligible_folds"]
    if not eligible or any(fold not in range(1, 6) for fold in eligible):
        raise RuntimeError("Invalid frozen Stage-2 eligible fold list")
    if 1 in eligible and not complete(a.runtime, 1):
        raise RuntimeError("I1 Stage-2 fold 1 has not completed")
    for fold in range(2, 6):
        if fold not in eligible:
            print(f"SKIP I1 Stage2 fold={fold} frozen gate false", flush=True)
            continue
        if complete(a.runtime, fold):
            print(f"SKIP I1 Stage2 fold={fold} complete", flush=True)
            continue
        command = [sys.executable, str(Path(__file__).with_name("train_tf_ictal_stage2.py")),
                   "--fold", str(fold), "--raw-cache", str(a.raw_cache),
                   "--runtime", str(a.runtime), "--lock", str(a.lock),
                   "--amendment", str(a.amendment), "--gate-audit", str(a.gate_audit)]
        log, error = (a.runtime / f"ictal_fold{fold}_stage2.log",
                      a.runtime / f"ictal_fold{fold}_stage2.err")
        print(f"START I1 Stage2 fold={fold}", flush=True)
        with log.open("a", encoding="utf-8") as stdout, error.open("a", encoding="utf-8") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if result.returncode or not complete(a.runtime, fold):
            raise RuntimeError(f"I1 Stage2 fold={fold} failed exit={result.returncode}")
        print(f"DONE I1 Stage2 fold={fold}", flush=True)
    print("I1_STAGE2_ALL_ELIGIBLE_FOLDS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
