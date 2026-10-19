"""Explicit public allowlist; never recurse over private runtime artifacts."""
import csv
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(r'D:\m8_7d_f0_complete_ablation_seed42_v1_runtime');PUB=ROOT/'public'
ALLOW=['ADMISSION.json','SOURCE_PUSH_BEFORE_TRAIN.json','TRAIN_STATUS.json','TRAIN_CELLS.csv','TRAIN_HISTORY.csv','INTERNAL_VERIFICATION.json','VALIDATION_REPLAY_CHECKS.csv','FREEZE_STATUS.json',
 'PREDICTION_STATUS.json','RAW_COVERAGE.csv','MAIN_RESULTS.csv','FOLD_RESULTS.csv','FOLD_SD.csv','SOURCE_RESULTS.csv','METRIC_COVERAGE.csv','INTERVALS.csv','MACRO_FAMILY_INTERVALS.csv','POOLED_RESULTS.csv',
 'REPORT_STATUS.json','INDEPENDENT_OUTER_VERIFICATION.json','OUTER_STATUS.json','FEATURE_CONTRIBUTION.csv','CORE_ABLATION_TABLE.csv','RAW_COMPLEMENTARITY.csv','GUIDANCE_INTERACTION.csv','REPORT.md','RENDER_MANIFEST.json']
ALLOW += ['figures/'+name+'.'+ext for name in ['FEATURE_CONTRIBUTION','PAIRED_CORE_INTERVALS'] for ext in ['svg','pdf','png']]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    status=json.loads((PUB/'OUTER_STATUS.json').read_text(encoding='utf-8'));v=json.loads((PUB/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))
    assert status['status']=='COMPLETE_VERIFIED_F0_18_ARM_FIVE_FOLD_OUTER' and v['status']=='PASS'
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    names=ALLOW+(['NATIVE_RESUME_AUDIT.json'] if (PUB/'NATIVE_RESUME_AUDIT.json').exists() else [])
    for name in names:
        p=PUB/name;assert p.is_file() and p.stat().st_size<5000000
        if p.suffix=='.csv':
            with p.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden)
    manifest=dict(status='EXPLICIT_ANONYMOUS_AGGREGATE_ALLOWLIST',files={n:sha(PUB/n) for n in names},private_artifacts_exported=False)
    text=json.dumps(manifest,indent=2,sort_keys=True)+'\n';path=ROOT/'public_aggregate_export_v1.zip';assert not path.exists(),'Preserve export; use a new reviewed version if needed'
    with zipfile.ZipFile(path,'x',compression=zipfile.ZIP_DEFLATED) as z:
        for name in names:z.write(PUB/name,name)
        z.writestr('PACKAGE_MANIFEST.json',text)
    print(json.dumps(dict(file=path.name,bytes=path.stat().st_size,sha256=sha(path),files=len(names)+1)),flush=True)

if __name__=='__main__':main()
