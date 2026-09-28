"""Sequential resumable Stage B launcher; never duplicates an active first cell."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


VARIANTS = ("M1_RAWTINY_NOPR", "M2_RAWTINY_PR", "M3_HYBRID_PR")


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    row = json.loads(path.read_text(encoding="utf-8"))
    return row.get("epochs") == 30 and len(row.get("epoch_rows", [])) == 30


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--wait-first-pid", type=int, default=0)
    a = p.parse_args()
    jobs = [(variant, fold) for variant in VARIANTS for fold in range(1, 6)]
    for variant, fold in jobs:
        summary = a.runtime / variant / f"fold_{fold}" / "summary.json"
        if complete(summary):
            print(f"[SKIP] {variant} fold={fold} already complete", flush=True)
            continue
        if variant == VARIANTS[0] and fold == 1 and a.wait_first_pid:
            import psutil
            while psutil.pid_exists(a.wait_first_pid) and not complete(summary):
                time.sleep(10)
            if not complete(summary):
                raise RuntimeError("Active M1 fold1 exited without a verified 30-epoch summary")
            print("[WAIT_FIRST] M1 fold1 complete", flush=True)
            continue
        log = a.runtime / f"{variant}_fold{fold}.log"
        err = a.runtime / f"{variant}_fold{fold}.err"
        command = [sys.executable, str(Path(__file__).with_name("train_stage_b.py")),
                   "--fold", str(fold), "--variant", variant,
                   "--raw-cache", str(a.raw_cache), "--runtime", str(a.runtime), "--lock", str(a.lock)]
        print(f"[START] {variant} fold={fold}", flush=True)
        with log.open("a", encoding="utf-8") as out, err.open("a", encoding="utf-8") as error:
            result = subprocess.run(command, stdout=out, stderr=error, check=False)
        if result.returncode != 0 or not complete(summary):
            raise RuntimeError(f"Stage B cell failed: {variant} fold={fold} return={result.returncode}")
        print(f"[DONE] {variant} fold={fold}", flush=True)
    print("STAGE_B_ALL_15_CELLS_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
