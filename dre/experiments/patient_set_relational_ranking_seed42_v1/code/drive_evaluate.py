"""Process-isolated, resumable evaluation for Windows native-library faults."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import protocol as p

HERE=Path(__file__).resolve().parent
NATIVE={-1073741819,-1073741795,-1073741674,-1073740791,-2147483645,
        0xC0000005,0xC000001D,0xC0000096,0xC0000409,0x80000003}


def run(args):
    command=[sys.executable,"-u",str(HERE/"evaluate.py"),*args]
    for attempt in range(4):
        status=subprocess.run(command,check=False).returncode
        if status==0:return
        if status not in NATIVE or attempt==3:
            raise RuntimeError(f"Context evaluation failed status={status}: {args}")
        print(f"[NATIVE_RETRY] args={args} status={status}",flush=True)


def main():
    p.preflight()
    for index,ctx in enumerate(p.all_contexts()):
        path=p.RUNTIME/"private"/"evaluated_contexts"/f"{index:02d}.pkl"
        if path.is_file():
            print(f"[SKIP_EVALUATED] index={index}",flush=True)
            continue
        run(["--context-index",str(index)])
    run(["--combine"])
    status=json.loads((p.RUNTIME/"EVALUATION_STATUS.json").read_text(encoding="utf-8"))
    if status["unique_patient_ids"]!=47:raise RuntimeError("Incomplete evaluation")
    print("[ALL_CONTEXT_METRICS_COMBINED]",flush=True)


if __name__=="__main__":main()
