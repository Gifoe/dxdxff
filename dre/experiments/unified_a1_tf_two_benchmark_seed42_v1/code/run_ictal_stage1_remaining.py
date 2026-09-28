"""Resumable sequential FIT-only I1 Stage-1 launcher for folds 2-5."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def complete(runtime: Path, fold: int) -> bool:
    path = runtime / f"fold_{fold}/stage1/summary.json"
    if not path.is_file():
        return False
    value = json.loads(path.read_text(encoding="utf-8"))
    return value.get("fold") == fold and value.get("stage1_complete") is True and value.get("epochs") == 15


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--amendment", type=Path, required=True)
    a = p.parse_args()
    if not complete(a.runtime, 1):
        raise RuntimeError("I1 fold 1 must have a verified complete Stage-1 summary")
    for fold in range(2, 6):
        if complete(a.runtime, fold):
            print(f"SKIP I1 fold={fold} complete", flush=True)
            continue
        command = [sys.executable, str(Path(__file__).with_name("train_tf_ictal_stage1.py")),
                   "--fold", str(fold), "--raw-cache", str(a.raw_cache),
                   "--runtime", str(a.runtime), "--lock", str(a.lock),
                   "--amendment", str(a.amendment)]
        log, error = (a.runtime / f"ictal_fold{fold}_stage1.log",
                      a.runtime / f"ictal_fold{fold}_stage1.err")
        print(f"START I1 fold={fold}", flush=True)
        with log.open("a", encoding="utf-8") as stdout, error.open("a", encoding="utf-8") as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
        if result.returncode or not complete(a.runtime, fold):
            raise RuntimeError(f"I1 fold={fold} failed (exit={result.returncode}); inspect private logs")
        print(f"DONE I1 fold={fold}", flush=True)
    print("I1_STAGE1_ALL_5_FOLDS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
