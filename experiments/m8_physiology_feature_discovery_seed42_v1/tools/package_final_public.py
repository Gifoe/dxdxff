"""Allowlist anonymous reports only. No recursive runtime/checkpoint/cache archive."""
import csv
import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
runtime=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
outer=runtime/'final_Outer/public';integrated=runtime/'integrated_M8_v2/public';old=runtime/'old7_matched_M8/public'
def jr(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert jr(outer/'INDEPENDENT_FINAL_VERIFICATION.json')['status']=='PASS'
assert jr(outer/'REPORTING_RECOVERY_AUDIT.json')['status']=='PASS'
assert jr(integrated/'INDEPENDENT_TRAINING_VERIFICATION.json')['status']=='PASS'
assert jr(old/'INDEPENDENT_RETRAIN_VERIFICATION.json')['status']=='PASS'
items=[]
def add(root,prefix,names):items.extend((root/n,prefix+'/'+n) for n in names)
add(outer,'outer',['main_results.csv','pooled_metrics.csv','fold_metrics.csv','center_metrics.csv','metric_coverage.csv',
    'confusion_matrices.csv','fold_mean_std.csv','paired_intervals.csv','INFERENCE_TIMING.csv','PRIMARY_SUCCESS_GATE.json','STATUS.json',
    'INDEPENDENT_METRIC_VERIFICATION.json','INDEPENDENT_FINAL_CACHE_VERIFICATION.json','INDEPENDENT_FINAL_VERIFICATION.json','REPORTING_RECOVERY_AUDIT.json'])
add(integrated,'integrated',['MODEL_SELECTION.csv','TRAINING_HISTORY.csv','MODEL_FREEZE_AUDIT.json','INDEPENDENT_TRAINING_VERIFICATION.json'])
add(old,'old7',['INDEPENDENT_RETRAIN_VERIFICATION.json'])
add(runtime/'compute_public','compute',['AFFINE_COMPUTE_COUNTS.csv','MEASURED_MODULE_SHAPE_MACS.csv','PARAMETER_AND_AFFINE_PROFILE.csv',
    'MATCHED_PATIENT_INFERENCE_TIMING.csv','OFFLINE_FEATURE_EXTRACTION_TIMING.csv','ACTUAL_TRAINING_WORK.csv','COMPUTE_AUDIT.json'])
add(runtime/'public','execution',['CPU_STATISTICS_RUNTIME_PARITY.json','RANKING_SORT_EQUIVALENCE.json','FINAL_CACHE_PROTOCOL_UNIT_TEST.json'])
for p,name in items:
    assert p.exists(),name
    if p.suffix=='.csv':
        with p.open(newline='',encoding='utf-8') as f:headers=next(csv.reader(f))
        assert not set(headers)&{'patient','channel','path','y','label','probability','prediction','indices','checkpoint','D1_checkpoint','M8_checkpoint','bank','validation_path'},name
# Preserve and compare partial native-runtime aggregates; no model/test rerun.
partial=runtime/'report_py313_partial_preserved_01';comparisons=[];maximum=0.;mismatches=0
for p in sorted(partial.glob('*.csv')):
    q=outer/p.name
    with p.open(newline='',encoding='utf-8') as f:a=list(csv.reader(f))
    with q.open(newline='',encoding='utf-8') as f:b=list(csv.reader(f))
    assert len(a)==len(b) and a[0]==b[0],p.name
    for left,right in zip(a[1:],b[1:]):
        assert len(left)==len(right)
        for x,y in zip(left,right):
            if x==y:continue
            try:xx=float(x);yy=float(y)
            except ValueError:mismatches+=1;continue
            if math.isnan(xx) and math.isnan(yy):continue
            error=abs(xx-yy);maximum=max(maximum,error)
            if error>1e-12:mismatches+=1
    comparisons.append(p.name)
parity=dict(status='PASS' if mismatches==0 else 'PARTIAL_OLD_RUNTIME_DIFFERENCES',files=comparisons,
    maximum_absolute_numeric_difference=maximum,cells_differing_above_1e_12=mismatches,
    final_values_independently_verified_against_sealed_scores=True,no_new_inference=True)
fr=jr(runtime/'final_Outer/FROZEN_BEFORE_OUTER_PRIVATE.json')
audit=dict(status='PASS',freeze_sha256=sha(runtime/'final_Outer/FROZEN_BEFORE_OUTER_PRIVATE.json'),
    seal_sha256=sha(runtime/'final_Outer/OUTER_PREDICTIONS_SEAL_PRIVATE.json'),arms=26,armfolds=130,
    source=fr['binding']['source'],metric_engine_sha256=fr['binding']['metric_engine_sha256'],
    selections=[{k:r[k] for k in ['model','fold','kind','checkpoint_sha256','bank_sha256','base_sha256','threshold']} for r in fr['selections']])
manifest={name:sha(p) for p,name in items}
extra={'execution/PARTIAL_TO_FINAL_AGGREGATE_PARITY.json':json.dumps(parity,indent=2,sort_keys=True),
    'outer/COMPACT_FREEZE_AUDIT.json':json.dumps(audit,indent=2,sort_keys=True)}
for name,data in extra.items():manifest[name]=hashlib.sha256(data.encode()).hexdigest()
target=runtime/'final_public_20261016_01.zip';assert not target.exists()
with ZipFile(target,'x',compression=ZIP_DEFLATED) as z:
    for p,name in items:z.write(p,'final_results/'+name)
    for name,data in extra.items():z.writestr('final_results/'+name,data)
    z.writestr('final_results/PUBLIC_SHA256.json',json.dumps(manifest,indent=2,sort_keys=True))
print('FINAL_ANONYMOUS_ONLY_PACKAGE_PASS',len(manifest),target.stat().st_size,parity['status'])
