"""Export an explicit compact aggregate allowlist; never traverse private runtime."""
import csv
import hashlib
import json
import zipfile
from pathlib import Path

ROOT=Path(r'D:\m8_7d_frozen_candidate_outer_audit_seed42_v1_runtime');PUBLIC=ROOT/'public'
ALLOW=['ADMISSION.json','ADAPTER_REPLAY_CHECKS.csv','FROZEN_MODELS.csv','PREDICTION_STATUS.json','RAW_COVERAGE.csv',
       'MAIN_RESULTS.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv',
       'MACRO_FAMILY_INTERVALS.csv','POOLED_RESULTS.csv','OBSERVED_TARGETS.json','INDEPENDENT_OUTER_VERIFICATION.json','OUTER_STATUS.json']

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    status=json.loads((PUBLIC/'OUTER_STATUS.json').read_text());v=json.loads((PUBLIC/'INDEPENDENT_OUTER_VERIFICATION.json').read_text())
    assert status['status']=='COMPLETE_VERIFIED_ALL11_FULL80_FROZEN_OUTER_AUDIT' and v['status']=='PASS'
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    for name in ALLOW:
        path=PUBLIC/name;assert path.is_file() and path.stat().st_size<1000000
        if path.suffix=='.csv':
            with path.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden)
    manifest=dict(status='AGGREGATE_ALLOWLIST_ONLY',files={n:sha(PUBLIC/n) for n in ALLOW},private_artifacts_exported=False)
    text=json.dumps(manifest,indent=2,sort_keys=True)+'\n';path=ROOT/'public_aggregate_export_v1.zip'
    assert not path.exists(),'Never overwrite an export; preserve and diagnose'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in ALLOW:z.write(PUBLIC/name,name)
        z.writestr('PACKAGE_MANIFEST.json',text)
    print(json.dumps(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),files=len(ALLOW)+1)),flush=True)

if __name__=='__main__':main()
