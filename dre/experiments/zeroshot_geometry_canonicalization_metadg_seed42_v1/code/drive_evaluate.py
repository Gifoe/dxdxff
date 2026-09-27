"""Run pre-label freeze, fixed VLOO evaluation, aggregate finalization."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from train import VARIANTS, preflight

HERE=Path(__file__).resolve().parent
NATIVE_CRASH={0xC0000005,0xC000001D,0xC0000096,0x80000003,-1073741819,-1073741795,-1073741674,-2147483645}


def call(script,*args):
    command=[sys.executable,"-u",str(HERE/script),*map(str,args)]
    for attempt in range(3):
        status=subprocess.run(command,check=False).returncode
        if status==0:return
        if status not in NATIVE_CRASH or attempt==2:
            raise RuntimeError(f"Evaluation stage failed status={status}: {' '.join(command)}")
        print(f"[NATIVE_RETRY] {script} status={status} attempt={attempt+1}/2",flush=True)


def main():
    preflight()
    call("evaluate.py","--stage","freeze")
    for fold in range(1,6):
        for variant in VARIANTS:
            call("evaluate.py","--stage","evaluate","--fold",fold,"--variant",variant)
    call("finalize.py")
    call("validate.py")
    print("[ZEROSHOT_STUDY_COMPLETE]",flush=True)


if __name__=="__main__":main()
