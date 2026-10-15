"""Verify all downloaded hashes, source bindings, actual cells and public arithmetic."""
import csv
import hashlib
import json
import math
from pathlib import Path
root=Path(__file__).resolve().parents[1];pub=root/'final_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jr(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
for n,h in jr(pub/'PUBLIC_SHA256.json').items():assert sha(pub/n)==h,n
sources=jr(root/'FROZEN_SCIENTIFIC_SOURCE_AUDIT.json');assert sources['status']=='PASS'
assert sources['source_sha256']==sha(root/'tools/audit_frozen_sources.py')
for phase,files in sources['phase_source_sha256'].items():
    for n,h in files.items():assert sha(root/n)==h,(phase,n)
freeze=jr(pub/'outer/COMPACT_FREEZE_AUDIT.json');assert freeze['armfolds']==130 and freeze['arms']==26
for n,h in freeze['source'].items():assert sha(root/n)==h,n
audit=jr(pub/'outer/INDEPENDENT_FINAL_VERIFICATION.json');assert audit['status']=='PASS' and audit['patient_rows']==2080
assert audit['source_sha256']==sha(root/'verify_final_cpu_metrics.py')
assert audit['original_verifier_sha256']==sha(root/'verify_final.py')
cache=jr(pub/'outer/INDEPENDENT_FINAL_CACHE_VERIFICATION.json');assert cache['source_sha256']==sha(root/'verify_final_cache_stage.py')
assert audit['cache_audit_sha256']==sha(pub/'outer/INDEPENDENT_FINAL_CACHE_VERIFICATION.json')
assert audit['freeze_sha256']==freeze['freeze_sha256'] and audit['seal_sha256']==freeze['seal_sha256']
assert audit['aliases_exact']==9 and audit['metric_process_did_not_import_torch']
recover=jr(pub/'outer/REPORTING_RECOVERY_AUDIT.json');assert recover['status']=='PASS'
assert recover['reporter_sha256']==sha(root/'report_final_cpu_v2.py')
assert recover['exact_frozen_report_and_classification_AST'] and recover['original_ranking_function_unmodified']
assert jr(pub/'execution/CPU_STATISTICS_RUNTIME_PARITY.json')['maximum_metric_difference']==0
assert jr(pub/'execution/PARTIAL_TO_FINAL_AGGREGATE_PARITY.json')['status']=='PASS'
old=jr(pub/'old7/INDEPENDENT_RETRAIN_VERIFICATION.json');assert old['status']=='PASS' and old['actual_models']==160
assert old['verifier_sha256']==sha(root/'tools/verify_old7_training.py')
new=jr(pub/'integrated/INDEPENDENT_TRAINING_VERIFICATION.json');assert new['status']=='PASS'
assert (new['logical_cells'],new['actual_M8_cells'],new['actual_D1_cells'],new['aliases'])==(45,36,26,9)
assert new['verifier_sha256']==sha(root/'verify_integrated_training.py')
assert jr(pub/'compute/COMPUTE_AUDIT.json')['source_sha256']==sha(root/'audit_compute.py')
main=rows(pub/'outer/main_results.csv');folds=rows(pub/'outer/fold_metrics.csv');assert len(main)==26
assert len(freeze['selections'])==len({(r['model'],r['fold']) for r in freeze['selections']})==130
for r in main:
    assert int(r['patients'])==80 and int(r['channels'])==7635
    f=[s for s in folds if s['model']==r['model'] and s['unit']=='patient_macro'];assert len(f)==5 and sum(int(s['patients']) for s in f)==80
    for k in ['auroc','auprc','macro_f1','f1','nez_f1','accuracy','ez_fraction_bias','mrr','hit_at_1']:
        expected=sum(float(s[k])*int(s['patients']) for s in f)/80
        assert math.isclose(expected,float(r[k]),abs_tol=1e-12,rel_tol=0),(r['model'],k)
values={r['model']:r for r in main};primary=jr(pub/'outer/PRIMARY_SUCCESS_GATE.json');truth=[]
for r in primary['comparisons']:
    delta={k:float(values['SLOT10_ELIGIBLE_ADD'][k])-float(values[r['reference']][k]) for k in ['auroc','auprc','macro_f1','f1']}
    for k,v in delta.items():assert math.isclose(v,r['delta_'+k],abs_tol=1e-12,rel_tol=0)
    passed=all(delta[k]>=.005 for k in ['auroc','auprc','macro_f1']) and delta['f1']>=0;assert passed==r['point_estimate_gate'];truth.append(passed)
assert primary['status']==('PASS_POINT_ESTIMATE_GATE' if all(truth) else 'NO_PREDECLARED_PRIMARY_IMPROVEMENT')
intervals=rows(pub/'outer/paired_intervals.csv');pairs={(r['model'],r['reference']) for r in intervals};assert len(pairs)==33
assert len([r for r in intervals if r['bonferroni_ci_low']])==6
for r in intervals:
    assert int(r['draws'])==10000 and r['threshold_reoptimized'].lower()=='false'
    delta=float(values[r['model']][r['metric']])-float(values[r['reference']][r['metric']])
    # Undefined metrics can have different patient sets. Their paired mean delta
    # is not the difference of two separately available-case public means.
    if int(r['paired_valid_patients'])==80 and math.isfinite(delta):
        assert math.isclose(delta,float(r['delta']),abs_tol=1e-12,rel_tol=0),(r['model'],r['reference'],r['metric'])
for p in pub.rglob('*.csv'):
    with p.open(newline='',encoding='utf-8') as f:headers=next(csv.reader(f))
    assert not set(headers)&{'patient','channel','path','y','label','prediction','probability','indices','checkpoint','D1_checkpoint','M8_checkpoint','bank','validation_path'},p.name
print('ALL_FINAL_PUBLIC_HASHES_SOURCES_222_ACTUAL_FITS_26_ARMS_AND_PRIMARY_ARITHMETIC_PASS')
