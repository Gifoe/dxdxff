#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO))
from task1_confirmatory.audit import repository_entrypoint_audit, reproduce_seed42
from task1_confirmatory.config import load_config
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--output-root'); p.add_argument('--reproduce-seed42',action='store_true'); a=p.parse_args(); c=load_config(a.config); output=Path(a.output_root or c['output_root']); result=repository_entrypoint_audit(c,output_root=output)
    if a.reproduce_seed42: result['seed42_reproduction']=reproduce_seed42(c,output_root=output)
    print(json.dumps(result,indent=2,default=str))
if __name__=='__main__': main()
