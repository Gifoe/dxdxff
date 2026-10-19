"""Explicit anonymous export allowlist; never scan private runtime directories."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile
ROOT=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1_runtime');PUB=ROOT/'public'
ALLOW=['ADMISSION.json','TABULAR_ADMISSION.json','SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv',
'TABULAR_TRAIN_STATUS.json','TABULAR_TRAIN_CELLS.csv','TABULAR_CALIBRATION_AUDIT.csv','TABULAR_VALIDATION_CHECKS.csv',
'INTERNAL_VERIFICATION.json','VALIDATION_REPLAY_CHECKS.csv','FREEZE_STATUS.json','PREDICTION_STATUS.json','RAW_COVERAGE.csv','TABULAR_COVERAGE.csv',
'MAIN_RESULTS.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv','MACRO_FAMILY_INTERVALS.csv','POOLED_RESULTS.csv',
'REPORT_STATUS.json','INDEPENDENT_OUTER_VERIFICATION.json','OUTER_STATUS.json','CORE_ABLATION_TABLE.csv','FEATURE_ABLATION_TABLE.csv','CLASSIFIER_TABLE.csv','PR_PAIRED_INTERVALS.csv','REPORT.md','RENDER_MANIFEST.json']
ALLOW += ['figures/'+n+'.'+e for n in ['FEATURE_CONTRIBUTION','CORE_PAIRED_INTERVALS','CLASSIFIER_PR_INTERVALS'] for e in ['svg','pdf','png']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    assert json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'))['status']=='COMPLETE_VERIFIED_F0_6D_30_GROUP_FIVE_FOLD_OUTER'
    assert json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))['status']=='PASS'
    names=ALLOW+(['NATIVE_RESUME_AUDIT.json'] if (PUB/'NATIVE_RESUME_AUDIT.json').exists() else [])
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','score_nez','y','path'}
    for n in names:
        p=PUB/n;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden)
    manifest=dict(status='EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST',files={n:sha(PUB/n) for n in names},private_artifacts_exported=False)
    out=ROOT/'public_aggregate_export_v1.zip';assert not out.exists(),'Preserve previous export'
    with zipfile.ZipFile(out,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for n in names:z.write(PUB/n,n)
        z.writestr('PACKAGE_MANIFEST.json',json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(file=out.name,bytes=out.stat().st_size,sha256=sha(out),files=len(names)+1)),flush=True)
if __name__=='__main__':main()
