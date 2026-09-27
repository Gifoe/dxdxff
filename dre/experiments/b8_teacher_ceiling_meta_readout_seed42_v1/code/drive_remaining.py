"""Resume-safe bounded-memory sequential meta/prototype/target evaluation."""
import subprocess
import sys
from pathlib import Path

import teacher_core as tc


def call(script,*args):
    subprocess.run([sys.executable,"-u",str(Path(__file__).with_name(script)),*map(str,args)],check=True)


def main():
    tc.preflight()
    for fold in range(1,6):
        for d in tc.DIMS:
            call("meta_train.py","--fold",fold,"--dimension",d)
        call("prototype_select.py","--fold",fold)
    print("[ALL_FIT_MODEL_SELECTION_FROZEN_BEFORE_TARGET_OUTCOMES]",flush=True)
    for fold in range(1,6):
        for cell in range(13):
            call("evaluate.py","--fold",fold,"--cell",cell)
    call("finalize.py")
    print("[B8_STUDY_COMPLETE]",flush=True)


if __name__=="__main__":main()
