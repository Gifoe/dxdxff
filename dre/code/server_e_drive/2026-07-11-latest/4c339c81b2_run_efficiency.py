#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]; sys.path.insert(0,str(REPO))
from task1_confirmatory.config import load_config
from task1_confirmatory.efficiency import collect_efficiency
def main() -> None:
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); a=p.parse_args(); c=load_config(a.config); print(json.dumps(collect_efficiency(output_root=c['output_root']),indent=2))
if __name__=='__main__': main()
