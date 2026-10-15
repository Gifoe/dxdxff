"""Anonymous development transfer audit; not independent neural replay or final Outer."""
import csv
import hashlib
import json
import math
from pathlib import Path
root=Path(__file__).resolve().parents[1];pub=root/'phase3_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jr(p):return json.loads(p.read_text(encoding='utf-8'))
def rows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
for name,h in jr(pub/'PUBLIC_SHA256.json').items():assert sha(pub/name)==h,name
old=pub/'old7_development';state=jr(old/'STATUS.json')
assert state['status']=='ALL80_MATCHED_OLD7_D1_AND_M8_DEVELOPMENT_CELLS_COMPLETE'
assert state['base_cells']==state['M8_cells']==80 and state['no_Outer'] and not state['full_goal_complete']
select=rows(old/'MODEL_SELECTION.csv');assert len(select)==80 and len({(r['variant'],r['fold']) for r in select})==80
for r in select:
    for head in ['D1','M8']:assert 6<=int(r[head+'_epochs_run'])<=30 and 1<=int(r[head+'_epoch'])<=int(r[head+'_epochs_run'])
summary=rows(old/'DEVELOPMENT_SUMMARY.csv');folds=rows(old/'DEVELOPMENT_BY_FOLD.csv');assert len(summary)==32
for r in summary:
    assert int(r['patient_appearances'])==65 and int(r['unique_patients'])==47
    f=[s for s in folds if s['variant']==r['variant'] and s['head']==r['head']];assert len(f)==5
    assert sum(int(s['patient_appearances']) for s in f)==65
    for k in ['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1']:
        v=sum(float(s[k])*int(s['patient_appearances']) for s in f)/65
        assert math.isclose(v,float(r[k]),abs_tol=1e-12,rel_tol=0),(r['variant'],r['head'],k)
new=pub/'integrated_admission';ad=jr(new/'REAL_FIT_ADMISSION.json')
assert ad['status']=='PASS' and ad['logical_cells']==45 and ad['unique_M8_inputs']==36 and ad['unique_D1_inputs']==26
assert ad['all_actual_admissions_before_training'] and ad['total_parameters']==11690 and ad['relative_parameters']<=1.1
assert len(ad['audits'])==45 and all(a['status']=='PASS' and a['base_frozen'] for a in ad['audits'])
assert all(a['original_Raw_prefix_max_error']==0 for a in ad['audits'] if a['variant']=='SLOT10_OLD7')
axis=jr(new/'WRONG_PAIR_AXIS_REGRESSION.json');assert axis['status']=='PASS'
assert axis['source_sha256']==sha(root/'integrated_data_v2.py') and axis['test_sha256']==sha(root/'test_pairing_index.py')
coverage=rows(new/'WRONG_PAIR_COVERAGE.csv');total=next(r for r in coverage if r['center']=='ALL')
assert int(total['channels'])==7635 and int(total['moved_channels'])==7633 and int(total['unchanged_singleton_channels'])==2
for p in pub.rglob('*.csv'):
    with p.open(newline='',encoding='utf-8') as f:headers=next(csv.reader(f))
    assert not set(headers)&{'patient','channel','path','y','label','prediction','probability','checkpoint','D1_checkpoint','M8_checkpoint','bank','validation_path'}
print('PHASE3_DEVELOPMENT_HASH_COUNTS_WEIGHTED_FOLDS_AND_ADMISSION_PASS_NOT_FINAL_OUTER')
