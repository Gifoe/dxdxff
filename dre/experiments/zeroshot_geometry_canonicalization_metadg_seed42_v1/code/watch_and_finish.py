"""Start target evaluation only after complete score production and driver exit."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

import psutil
from train import RUNTIME, preflight


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--driver-pid",type=int,required=True)
    args=parser.parse_args()
    preflight()
    log=RUNTIME/"DRIVE.log";err=RUNTIME/"DRIVE.err"
    deadline=time.monotonic()+18*3600
    while True:
        lines=log.read_text(encoding="utf-8",errors="replace") if log.is_file() else ""
        complete="[ALL_TARGET_SCORES_FROZEN_PENDING_HASH]" in lines
        running=psutil.pid_exists(args.driver_pid)
        if complete and not running:break
        if not running and not complete:
            raise RuntimeError("Training driver exited without complete FIT/full score marker")
        if time.monotonic()>deadline:raise TimeoutError("Training driver exceeded 18-hour watcher bound")
        time.sleep(30)
    if err.is_file() and err.stat().st_size:
        raise RuntimeError("Training driver stderr nonempty; inspect before target labels")
    print("[PRELABEL] full score production complete; starting hash freeze",flush=True)
    command=[sys.executable,"-u",str(Path(__file__).with_name("drive_evaluate.py"))]
    status=subprocess.run(command,check=False).returncode
    if status:raise RuntimeError(f"Target evaluation/finalizer failed status={status}")
    print("[WATCHER_COMPLETE]",flush=True)


if __name__=="__main__":main()
