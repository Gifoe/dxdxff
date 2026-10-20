"""Hash-locked execution overlay; original13 scientific files/inputs stay unchanged."""
import hashlib
import json
import os
from pathlib import Path
import runpy
import sys
import psutil
HERE=Path(__file__).resolve().parent;CODE=Path(r'D:\drechan_pooling_capacity_ablation_seed42_v1')
ROOT=Path(r'D:\drechan_pooling_capacity_ablation_seed42_v1_runtime')
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
manifest=json.loads((HERE/'EXECUTION_MANIFEST.json').read_text())
for n,h in manifest['execution_source'].items():assert sha(HERE/n)==h,('EXECUTION_CHANGED',n)
for n,h in manifest['scientific_source'].items():assert sha(CODE/n)==h,('SCIENTIFIC_CHANGED',n)
assert len(sys.argv)==2 and sys.argv[1] in ['admit_repair.py','train.py','verify_internal.py','evaluate_outer.py','report_outer.py','verify_outer.py','package_public.py']
target=sys.argv[1]
for n in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:assert os.environ.get(n)=='1'
assert not os.environ.get('KMP_DUPLICATE_LIB_OK');psutil.Process().cpu_affinity(list(range(16,32)))
sys.path.insert(0,str(CODE));import common
common.ORIGINAL_ATTENTION=common.MODEL.attention_mean
sys.path.insert(0,str(HERE));from repair import attention_mean
common.MODEL.attention_mean=attention_mean
common.EXECUTION_MANIFEST_SHA256=sha(HERE/'EXECUTION_MANIFEST.json')
if target!='admit_repair.py':
    proof=common.jr(ROOT/'public/ENGINEERING_REPAIR_ADMISSION.json')
    assert proof['status']=='PASS' and proof['execution_manifest_sha256']==common.EXECUTION_MANIFEST_SHA256
    lock=common.jr(ROOT/'EXECUTION_OVERLAY_PRIVATE.json');assert lock['execution_manifest_sha256']==common.EXECUTION_MANIFEST_SHA256
    for rel,h in lock['preserved_completed_files'].items():assert sha(ROOT/rel)==h
sys.argv=[str((HERE if target=='admit_repair.py' else CODE)/target)]
runpy.run_path(sys.argv[0],run_name='__main__')
