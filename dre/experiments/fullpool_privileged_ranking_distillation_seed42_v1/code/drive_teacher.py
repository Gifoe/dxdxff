"""Resumable process-isolated FIT Teacher crossfit driver."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import protocol as p

HERE=Path(__file__).resolve().parent
NATIVE={-1073741819,-1073741795,-1073741674,-1073740791,-2147483645,
        0xC0000005,0xC000001D,0xC0000096,0xC0000409,0x80000003}


def run(script,args):
    command=[sys.executable,"-u",str(HERE/script),*args]
    for attempt in range(4):
        status=subprocess.run(command,check=False).returncode
        if status==0:return
        if status not in NATIVE or attempt==3:
            raise RuntimeError(f"FIT Teacher stage failed status={status}: {script} {args}")
        print(f"[NATIVE_RETRY] {script} {args} status={status}",flush=True)


def main():
    p.preflight()
    for index,ctx in enumerate(p.all_contexts()):
        run("teacher.py",["--stage","grid","--context-index",str(index)])
        run("teacher.py",["--stage","oof","--context-index",str(index)])
        print(f"[TEACHER_CONTEXT_DONE] index={index} fold={ctx['fold']} {ctx['context_id']}",flush=True)
    run("audit_teacher.py",[])
    print("[ALL_FIT_TEACHER_COMPLETE]",flush=True)


if __name__=="__main__":main()
