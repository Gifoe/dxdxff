"""Execution-only foreground owner with preserved per-stage native logs."""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('tag');parser.add_argument('--phase',choices=['all','select','seal'],default='all');args=parser.parse_args()
    import psutil
    psutil.Process().cpu_affinity([16])
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[name]='1'
    here=Path(__file__).resolve().parent;root=Path(r'D:\m8_nonlocal6_conditional_selection_seed42_v1_runtime');root.mkdir(exist_ok=True)
    stages=[('prepare',['prepare_inputs.py']),('select',['select_and_seal.py','select']),('seal',['select_and_seal.py','seal'])]
    if args.phase!='all':stages=[s for s in stages if s[0]==args.phase]
    for stage,command in stages:
        out=root/f'{stage}_{args.tag}.log';err=root/f'{stage}_{args.tag}.err';assert not out.exists() and not err.exists()
        with out.open('xb') as stdout,err.open('xb') as stderr:
            process=subprocess.Popen([sys.executable,'-u',str(here/command[0]),*command[1:]],stdout=stdout,stderr=stderr,cwd=here)
            print('STAGE_START',stage,'PID',process.pid,flush=True)
            while True:
                try:code=process.wait(timeout=30);break
                except subprocess.TimeoutExpired:print('STAGE_LIVE',stage,'PID',process.pid,flush=True)
        print('STAGE_EXIT',stage,code,flush=True)
        if code:
            print(err.read_text(encoding='utf-8',errors='replace'),flush=True);raise SystemExit(code)
    print('REQUESTED_STAGES_COMPLETE',flush=True)
