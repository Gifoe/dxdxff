"""Own-worker isolation only; no server-wide environment/system changes."""
import os
from pathlib import Path
import runpy
import sys
import psutil

assert len(sys.argv) >= 2
allowed = {'admit.py', 'train.py', 'verify_internal.py', 'evaluate_outer.py',
           'report_outer.py', 'verify_outer.py', 'package_public.py'}
target = Path(sys.argv[1]).name
assert target in allowed
assert not os.environ.get('KMP_DUPLICATE_LIB_OK')
for name in ['OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS']:
    assert os.environ.get(name) == '1'
psutil.Process().cpu_affinity(list(range(16, 32)))
sys.argv = sys.argv[1:]
sys.argv[0] = str(Path(__file__).resolve().parent/target)
runpy.run_path(sys.argv[0], run_name='__main__')
