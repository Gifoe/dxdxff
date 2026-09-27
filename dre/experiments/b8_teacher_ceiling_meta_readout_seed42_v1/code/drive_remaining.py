"""Resume-safe bounded-memory sequential meta/prototype/target evaluation."""
import subprocess
import sys
import time
from pathlib import Path

import teacher_core as tc


def call(script,*args):
    argv=[sys.executable,"-u",str(Path(__file__).with_name(script)),*map(str,args)]
    for attempt in range(1,6):
        result=subprocess.run(argv,check=False)
        if result.returncode==0:return
        if result.returncode not in (-1073741819,3221225477,2147483651,-2147483645) or attempt==5:
            raise subprocess.CalledProcessError(result.returncode,argv)
        print(f"[NATIVE_RETRY] {script} args={args} attempt={attempt} exit={result.returncode}",flush=True)
        time.sleep(2)


def main():
    tc.preflight()
    for fold in range(1,6):
        for d in tc.DIMS:
            call("meta_train.py","--fold",fold,"--dimension",d)
        call("prototype_select.py","--fold",fold)
    call("freeze.py")
    print("[ALL_FIT_MODEL_SELECTION_FROZEN_BEFORE_TARGET_OUTCOMES]",flush=True)
    for fold in range(1,6):
        for cell in range(13):
            call("evaluate.py","--fold",fold,"--cell",cell)
    call("finalize.py")
    print("[B8_STUDY_COMPLETE]",flush=True)


if __name__=="__main__":main()
