"""Explicit anonymous-only transfer allowlist; no runtime-directory archive."""
import csv
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED

runtime=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
pub=runtime/'public'
names=['NUMERICAL_CONTROL_EXTRACTION_STATUS.json','NONLINEAR_INDEPENDENT_VERIFICATION.json',
    'NONLINEAR_INNER_AND_CENTER_METRICS.csv','FEATURE_DIRECTION_STABILITY.csv','NONLINEAR_PATIENT_CENTER_ISOLATION.csv',
    'NONLINEAR_FIT_CROSSFIT_SUMMARY.csv','NONLINEAR_PAIRED_SCREEN_DIFFERENCES.csv','NONLINEAR_PAIRED_FIT_INTERVALS.csv',
    'NONLINEAR_FEATURE_ELIGIBILITY.csv','PER_FIT_FEATURE_SELECTION.json','NONLINEAR_SCREEN_STATUS.json',
    'INTEGRATED_MODEL_SYNTHETIC_AUDIT.json']
assert json.loads((pub/'NONLINEAR_INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
assert json.loads((pub/'INTEGRATED_MODEL_SYNTHETIC_AUDIT.json').read_text())['status']=='PASS'
manifest={}
for name in names:
    p=pub/name
    if p.suffix=='.csv':
        with p.open(encoding='utf-8',newline='') as f:headers=next(csv.reader(f))
        assert not set(h.lower() for h in headers)&{'patient','channel','id','path','label','y','prediction','probability','checkpoint'}
    manifest[name]=hashlib.sha256(p.read_bytes()).hexdigest()
target=runtime/'phase2_public_20261016_01.zip';assert not target.exists()
with ZipFile(target,'x',compression=ZIP_DEFLATED) as z:
    for name in names:z.write(pub/name,'phase2_results/'+name)
    z.writestr('phase2_results/PUBLIC_SHA256.json',json.dumps(manifest,indent=2,sort_keys=True))
print('PHASE2_PUBLIC_ONLY_PACKAGE_PASS',len(names),target.stat().st_size)
