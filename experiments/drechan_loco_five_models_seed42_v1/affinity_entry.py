"""Only this worker's CPU affinity; no unrelated process/GPU changes."""
import os
from pathlib import Path
import runpy
import sys
import psutil
assert not os.environ.get('KMP_DUPLICATE_LIB_OK')
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:assert os.environ[n]=='1'
psutil.Process().cpu_affinity(list(range(16,32)))
assert Path(sys.argv[1]).name in ['admit.py','pipeline.py']
sys.argv=sys.argv[1:];sys.argv[0]=str(Path(__file__).resolve().parent/sys.argv[0]);runpy.run_path(sys.argv[0],run_name='__main__')
