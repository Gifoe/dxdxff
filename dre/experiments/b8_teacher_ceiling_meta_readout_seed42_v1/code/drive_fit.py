"""Bounded-memory FIT-only hyperparameter selection; one context per child process."""
import subprocess
import sys
import time
from pathlib import Path

import teacher_core as tc
from fit_select import context_key


def main():
    tc.preflight()
    script=Path(__file__).with_name("fit_select.py")
    for fold in range(1,6):
        keys=list(dict.fromkeys((int(r["selected_epoch"]),float(r["selected_threshold"])) for r in tc.selected_rows(fold)))
        n=len(keys)
        print(f"[FOLD] {fold} contexts={n}",flush=True)
        for i in range(n):
            epoch,tau=keys[i]
            key=context_key({"fold":fold,"epoch":epoch,"tau":tau})
            if (tc.RUNTIME/"private"/tc.FIT_SELECTION_FOLDER/f"{key}.pkl").exists():
                print(f"[RESUME] fold={fold} context={i} key={key}",flush=True)
                continue
            for variant in ("R64","PCA4","PCA8","PCA16"):
                for mode in ("b8","fullpool"):
                    argv=[sys.executable,"-u",str(script),"--fold",str(fold),"--context",str(i),
                          "--variant",variant,"--mode",mode]
                    for attempt in range(1,4):
                        result=subprocess.run(argv,check=False)
                        if result.returncode==0:break
                        if result.returncode not in (-1073741819,3221225477) or attempt==3:
                            raise subprocess.CalledProcessError(result.returncode,argv)
                        print(f"[NATIVE_RETRY] fold={fold} context={i} {variant} {mode} attempt={attempt}",flush=True)
                        time.sleep(2)
    print("[FIT_SELECTION_DONE]",flush=True)


if __name__=="__main__":main()
