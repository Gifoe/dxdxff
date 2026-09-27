"""Resumable, process-isolated FIT selection and full-FIT score production."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from train import LOCK_SHA, RUNTIME, VARIANTS, configs, preflight

HERE = Path(__file__).resolve().parent
NATIVE_CRASH = {0xC0000005, 0xC000001D, 0xC0000096, 0x80000003,
                -1073741819, -1073741795, -1073741674, -2147483645}


def completed_full(fold, variant, index):
    """Avoid importing the native ML stack only to skip an already-frozen unit."""
    folder = RUNTIME / "full" / f"fold_{fold}" / variant / f"config_{index:02d}"
    summary = folder / "summary.json"
    if not summary.is_file(): return False
    row = json.loads(summary.read_text(encoding="utf-8"))
    if (row.get("lock_sha") != LOCK_SHA or row.get("fold") != fold or
            row.get("variant") != variant or row.get("config_index") != index or
            row.get("checkpoints") != 30 or not row.get("validation_scores_frozen")):
        raise RuntimeError(f"Invalid completed full-FIT marker: {summary}")
    if len(row.get("score_snapshot_hashes", [])) != 30:
        raise RuntimeError(f"Incomplete frozen score list: {summary}")
    if any(not (folder / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl").is_file() for epoch in range(1, 31)):
        raise RuntimeError(f"Completed marker has missing scores: {summary}")
    return True


def call(*args):
    command = [sys.executable, "-u", str(HERE / "train.py"), *map(str, args)]
    for attempt in range(3):
        status = subprocess.run(command, check=False).returncode
        if status == 0: return
        if status not in NATIVE_CRASH or attempt == 2:
            raise RuntimeError(f"Unit failed status={status}: {' '.join(command)}")
        print(f"[NATIVE_RETRY] status={status} attempt={attempt+1}/2; private epoch resume preserved", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("fit", "full", "all"), default="all")
    args = parser.parse_args()
    preflight()
    if args.stage in ("fit", "all"):
        for fold in range(1, 6):
            for variant in VARIANTS:
                for index in range(len(configs(variant, fold))):
                    call("--stage", "fit", "--fold", fold, "--variant", variant,
                         "--config-index", index)
                call("--stage", "select", "--fold", fold, "--variant", variant)
        print("[FIT_SELECTION_COMPLETE]", flush=True)
    if args.stage in ("full", "all"):
        from train import selected_config
        for fold in range(1, 6):
            for variant in VARIANTS:
                index = selected_config(fold, variant)["config_index"]
                if completed_full(fold, variant, index):
                    print(f"[SKIP_COMPLETE] full fold={fold} {variant} config={index}", flush=True)
                    continue
                call("--stage", "full", "--fold", fold, "--variant", variant,
                     "--config-index", index)
        print("[ALL_TARGET_SCORES_FROZEN_PENDING_HASH]", flush=True)


if __name__ == "__main__": main()
