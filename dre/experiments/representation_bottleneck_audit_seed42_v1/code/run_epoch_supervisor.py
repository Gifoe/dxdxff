"""Run each frozen fold/epoch in an isolated process; retry only native crashes.

The scientific protocol and cached completed cells are unchanged. This isolates
intermittent Windows access violations seen after long NumPy-heavy runs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from reproduce_source import RUNTIME, ensure_source

SCRIPT = Path(__file__).with_name("run_frozen_probes.py")
VARIANTS = ("A1", "A2", "B1", "B2", "C1", "C2")


def complete(fold: int, epoch: int) -> bool:
    folder = RUNTIME / "private" / f"fold_{fold}" / f"epoch_{epoch:02d}"
    return (folder / "representation.pt").is_file() and all(
        (folder / f"{variant}_probe.pt").is_file() and (folder / f"{variant}_validation_grid.json").is_file()
        for variant in VARIANTS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-fold", type=int, default=1)
    parser.add_argument("--end-fold", type=int, default=5)
    parser.add_argument("--max-retries", type=int, default=3)
    options = parser.parse_args()
    if not (1 <= options.start_fold <= options.end_fold <= 5):
        raise ValueError("Fold range must be within 1..5")
    ensure_source()
    for fold in range(options.start_fold, options.end_fold + 1):
        for epoch in range(1, 31):
            if complete(fold, epoch):
                print(f"[KEEP] fold={fold} epoch={epoch} already complete", flush=True)
                continue
            for attempt in range(1, options.max_retries + 1):
                result = subprocess.run([sys.executable, "-X", "faulthandler", str(SCRIPT), "--fold", str(fold),
                                         "--epoch", str(epoch)], capture_output=True, text=True,
                                        encoding="utf-8", errors="replace", check=False)
                if result.stdout:
                    print("\n".join(result.stdout.splitlines()[-10:]), flush=True)
                if result.returncode == 0 and complete(fold, epoch):
                    print(f"[DONE] fold={fold} epoch={epoch} attempt={attempt}", flush=True)
                    break
                print(f"[RETRY] fold={fold} epoch={epoch} attempt={attempt} returncode={result.returncode}", flush=True)
                if result.stderr:
                    print("\n".join(result.stderr.splitlines()[-18:]), file=sys.stderr, flush=True)
            else:
                raise RuntimeError(f"Frozen probe cell failed after {options.max_retries} attempts: fold={fold} epoch={epoch}")
    print(f"[SUPERVISOR] requested folds {options.start_fold}..{options.end_fold} complete", flush=True)


if __name__ == "__main__":
    main()
