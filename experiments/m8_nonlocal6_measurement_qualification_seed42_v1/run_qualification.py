"""Execution-only native stdio capture; no change to frozen qualification."""
import os
import subprocess
import sys
from pathlib import Path

if __name__=='__main__':
    root=Path(r'D:\m8_nonlocal6_measurement_qualification_seed42_v1_runtime')
    tag=sys.argv[1];out=root/f'qualification_{tag}.log';err=root/f'qualification_{tag}.err'
    assert not out.exists() and not err.exists()
    with out.open('xb') as stdout,err.open('xb') as stderr:
        result=subprocess.run([sys.executable,'-u',str(Path(__file__).with_name('qualify.py'))],stdout=stdout,stderr=stderr)
    print('QUALIFICATION_EXIT',result.returncode,flush=True)
    if result.returncode:print(err.read_text(encoding='utf-8',errors='replace'),flush=True)
    raise SystemExit(result.returncode)
