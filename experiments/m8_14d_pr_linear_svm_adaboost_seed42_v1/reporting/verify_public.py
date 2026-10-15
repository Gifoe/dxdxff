"""Local anonymous-artifact verification; no server or private data needed."""
import csv
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parents[1]
pub=root/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):
    with (pub/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
status=json.loads((pub/'OUTER_STATUS.json').read_text())
assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE' and status['new_models']==20
for name,h in status['public_sha256'].items():assert sha(pub/name)==h,name
source=json.loads((pub/'SOURCE_HASHES.json').read_text())
for name,h in source['source'].items():assert sha(root/name)==h,name
repair=json.loads((pub/'REPORTING_REPAIR_AUDIT.json').read_text())
assert sha(root/'reporting/finalize.py')==repair['reporting_helper_sha256']
assert json.loads((pub/'INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
main=rows('main_results.csv');folds=rows('outer_fold_metrics.csv')
assert len(main)==6 and len(rows('MODEL_SELECTION.csv'))==20 and len(rows('VALIDATION_SEARCH.csv'))==50
for r in main:
    assert int(r['patients'])==80 and int(r['channels'])==7635
    selected=[q for q in folds if q['model']==r['model'] and q['version']==r['version'] and q['unit']=='patient_macro']
    assert len(selected)==5 and sum(int(q['patients']) for q in selected)==80
    for m in ['auroc','auprc','macro_f1','f1','nez_f1','accuracy','ez_fraction_bias']:
        expected=sum(float(q[m])*int(q['patients']) for q in selected)/80
        assert math.isclose(expected,float(r[m]),rel_tol=0,abs_tol=1e-12),(r['model'],r['version'],m)
for path in pub.glob('*.csv'):
    with path.open(newline='',encoding='utf-8') as f:
        columns=next(csv.reader(f))
    assert not set(columns)&{'patient','channel','global_index','path','probability','prediction','y'}
print('PUBLIC_HASH_SOURCE_PRIVACY_AND_WEIGHTED_FOLD_VERIFICATION_PASS')
