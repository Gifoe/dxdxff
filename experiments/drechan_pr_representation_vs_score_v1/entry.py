"""Pinned-runtime entry; establish process-local paths before parent imports."""
import argparse
import os
import sys
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['probe','admit','train','freeze','evaluate','verify','report','all'])
    p.add_argument('--runtime',default=r'D:\Codex_Research\experiments\drechan_pr_representation_vs_score_v1_runtime')
    p.add_argument('--source-commit',default='ENGINEERING_UNFROZEN');a=p.parse_args()
    os.environ['PR_REPR_RUNTIME']=a.runtime
    for name in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[name]='1'
    sys.dont_write_bytecode=True
    from path_adapter import install
    install(r'D:\Codex_Research\_maintenance\cleanup_20261028\LEGACY_PATHS.json',[Path(__file__).parent,Path(a.runtime)])
    import pipeline
    pipeline.run(a.stage,a.source_commit)

if __name__=='__main__':main()

