"""Resumable 4 variants x 5 folds x 4 locked optimizer settings.

Stops on the first engineering failure. Never opens any target evaluation file.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from stage0_source import file_sha, write_json

VARIANTS = ("S1_ABS_SPECTRAL", "S2_ABS_SELF", "S3_DUAL_REFERENCE", "S4_DRST_PR")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    a = p.parse_args()
    if not (a.runtime / "S4_DRST_PR" / "fold_1" / "lr_0.0001_wd_0.0001" /
            "overfit_sanity" / "OVERFIT_SANITY_AUDIT.json").is_file():
        raise RuntimeError("Mandatory source-only overfit sanity has not passed")
    lock_sha = file_sha(a.lock)
    for fold in range(1, 6):
        normalizer = json.loads((a.runtime / f"fold_{fold}" / "FIT_GLOBAL_SPECTRAL_NORMALIZER.json").read_text(encoding="utf-8"))
        if normalizer["lock_sha256"] != lock_sha or not normalizer["fit_only"]:
            raise RuntimeError(f"Missing locked FIT-only normalizer fold={fold}")
    completed = 0
    started = time.time()
    for variant in VARIANTS:
        for fold in range(1, 6):
            for lr in (1e-4, 3e-4):
                for wd in (1e-4, 1e-3):
                    cell = a.runtime / variant / f"fold_{fold}" / f"lr_{lr:g}_wd_{wd:g}"
                    cell.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, str(Path(__file__).with_name("train_cell.py")),
                               "--fold", str(fold), "--variant", variant, "--lr", str(lr), "--wd", str(wd),
                               "--raw-cache", str(a.raw_cache), "--runtime", str(a.runtime), "--lock", str(a.lock)]
                    # Windows/CUDA can terminate a process with 0xC0000005
                    # without a Python traceback. Retry only this process-level
                    # failure from the last atomically saved model+optimizer+RNG
                    # checkpoint. Other errors require engineering inspection.
                    for attempt in range(4):
                        with (cell / "train.log").open("a", encoding="utf-8") as out, (cell / "train.err").open("a", encoding="utf-8") as err:
                            status = subprocess.run(command, stdout=out, stderr=err, env=os.environ.copy(), check=False).returncode
                        if status != 3221225477:
                            break
                        if not (cell / "resume_private.pt").is_file() or attempt == 3:
                            break
                        print(f"TRANSIENT_WINDOWS_ACCESS_VIOLATION attempt={attempt+1} resume={variant} fold={fold}", flush=True)
                    if status != 0:
                        write_json(a.runtime / "TRAINING_GRID_STATUS.json", {"complete": False, "completed_cells": completed,
                                   "expected_cells": 80, "failed_cell": [variant, fold, lr, wd], "exit_code": status,
                                   "lock_sha256": lock_sha, "target_outcomes_accessed": False})
                        raise RuntimeError(f"Training cell failed: {variant} fold={fold} lr={lr} wd={wd}; exit={status}")
                    summary = json.loads((cell / "summary.json").read_text(encoding="utf-8"))
                    if not summary["complete"] or summary["lock_sha256"] != lock_sha:
                        raise RuntimeError("Training cell returned without complete locked summary")
                    completed += 1
                    write_json(a.runtime / "TRAINING_GRID_STATUS.json", {"complete": completed == 80,
                               "completed_cells": completed, "expected_cells": 80,
                               "last_cell": [variant, fold, lr, wd], "elapsed_seconds": time.time() - started,
                               "lock_sha256": lock_sha, "target_outcomes_accessed": False})
                    print(f"GRID {completed}/80 {variant} fold={fold} lr={lr:g} wd={wd:g}", flush=True)


if __name__ == "__main__":
    main()
