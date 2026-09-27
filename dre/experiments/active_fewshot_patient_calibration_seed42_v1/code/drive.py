"""Resume-safe one-cell subprocess driver; bounds Windows worker memory growth."""
from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

from common import PRIOR_RUNTIME, RUNTIME, preflight, read_csv


def completed(fold):
    rows=read_csv(PRIOR_RUNTIME/"private"/f"fold_{fold}"/"A1_VLOO_PRIVATE.csv")
    folder=RUNTIME/"private"/f"fold_{fold}"
    count=0
    for row in rows:
        stem=hashlib.sha256(row["subject_id"].encode()).hexdigest()[:16]
        count+=(folder/(stem+".pkl")).is_file()
    return count


def main():
    preflight()
    runner=Path(__file__).with_name("run.py")
    for fold in range(1,6):
        failures=0
        while completed(fold)<13:
            before=completed(fold)
            result=subprocess.run([sys.executable,"-u",str(runner),"--fold",str(fold),"--max-new-cells","1"],
                                  check=False)
            after=completed(fold)
            print(f"[DRIVE] fold={fold} before={before} after={after} exit={result.returncode}",flush=True)
            if after==before:
                failures+=1
                if failures>=3:
                    raise RuntimeError(f"Repeated cell failure without progress in fold {fold}")
            else:
                failures=0
        print(f"[FOLD_COMPLETE] fold={fold} 13/13",flush=True)
    print("[ALL_COMPLETE] 65/65",flush=True)


if __name__=="__main__":
    main()
