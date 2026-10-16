"""Fresh accepted execution or verification-only for a completed atlas."""
import os
from pathlib import Path
import subprocess
import sys


if __name__=='__main__':
    for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
    root=Path(r'D:\m8_physiological_evidence_atlas_seed42_v1_runtime');accepted=root/'rolecount_checked'
    root.mkdir(exist_ok=True)
    if (accepted/'public/ATLAS_STATUS.json').exists():
        raise SystemExit(subprocess.run([sys.executable,str(Path(__file__).with_name('verify_atlas.py'))]).returncode)
    if (accepted/'INPUT_FREEZE_PRIVATE.json').exists():
        raise RuntimeError('Partial accepted runtime: inspect preserved logs and source binding before an engineering-only resume; no blind restart.')
    import run_rolecount_checked
    run_rolecount_checked.main()
