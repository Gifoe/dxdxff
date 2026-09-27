"""Resumable, process-isolated FIT selection and full-FIT score production."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from train import VARIANTS, configs, preflight

HERE = Path(__file__).resolve().parent
NATIVE_CRASH = {0xC0000005, 0xC000001D, 0x80000003,
                -1073741819, -1073741795, -2147483645}


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
                call("--stage", "full", "--fold", fold, "--variant", variant,
                     "--config-index", index)
        print("[ALL_TARGET_SCORES_FROZEN_PENDING_HASH]", flush=True)


if __name__ == "__main__": main()
