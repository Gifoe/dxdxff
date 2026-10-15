"""Resume-safe sequential full protocol, all five folds; stop on real failure."""
import os
import subprocess
import sys
from pathlib import Path
ROOT=Path(__file__).parent

def run(runtime):
    runtime=Path(runtime);runtime.mkdir(parents=True,exist_ok=True)
    for name in ['TimeConvCNN','EEGNet','SEEGNet','SEEGformer']:
        for phase in ['train.py','outer.py']:
            print('START_RAW_PIPELINE',name,phase,flush=True)
            subprocess.run([sys.executable,'-u',str(ROOT/phase),'--runtime',str(runtime),'--model',name],check=True,cwd=ROOT)
    subprocess.run([sys.executable,'-u',str(ROOT/'tests/verify_results.py'),'--runtime',str(runtime)],check=True,cwd=ROOT)
    print('ALL_EIGHT_RAW_CONFIGURATIONS_COMPLETE',flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--runtime',default=r'D:\m8_raw_backbones_pr_seed42_v1_runtime');run(p.parse_args().runtime)
