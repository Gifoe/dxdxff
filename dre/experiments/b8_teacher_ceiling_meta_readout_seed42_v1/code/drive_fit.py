"""Bounded-memory FIT-only hyperparameter selection; one context per child process."""
import subprocess
import sys
from pathlib import Path

import teacher_core as tc


def main():
    tc.preflight()
    script=Path(__file__).with_name("fit_select.py")
    for fold in range(1,6):
        n=len({(int(r["selected_epoch"]),float(r["selected_threshold"])) for r in tc.selected_rows(fold)})
        print(f"[FOLD] {fold} contexts={n}",flush=True)
        for i in range(n):
            subprocess.run([sys.executable,"-u",str(script),"--fold",str(fold),"--context",str(i)],check=True)
    print("[FIT_SELECTION_DONE]",flush=True)


if __name__=="__main__":main()
