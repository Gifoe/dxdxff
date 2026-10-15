"""Verify downloaded anonymous files and weighted fold means without private data."""
import csv
import hashlib
import json
import math
from pathlib import Path

root=Path(__file__).resolve().parents[1];pub=root/'results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):
    with (pub/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
for name,h in json.loads((pub/'PUBLIC_SHA256.json').read_text()).items():assert sha(pub/name)==h,name
state=json.loads((pub/'FINAL_STATUS.json').read_text())
assert state['status']=='M8_MECHANISM_5FOLD_OUTER_COMPLETE' and state['verified_patient_rows']==640
assert json.loads((pub/'INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
fusion=json.loads((pub/'INDEPENDENT_FUSION_AUDIT.json').read_text())
assert fusion['status']=='PASS' and fusion['validation_candidates_reconstructed']==105
assert sha(root/'tools/verify_fusion.py')==fusion['verifier_sha256']
for name,h in json.loads((pub/'INPUT_AUDIT.json').read_text())['source'].items():assert sha(root/name)==h,name
main=rows('main_results.csv');folds=rows('fold_metrics.csv');assert len(main)==8
assert len(rows('MODEL_SELECTION.csv'))==20 and len(rows('FUSION_SELECTION.csv'))==5
assert len(rows('FUSION_VALIDATION_SEARCH.csv'))==105
for r in main:
    assert int(r['patients'])==80 and int(r['channels'])==7635
    ff=[q for q in folds if q['model']==r['model'] and q['unit']=='patient_macro']
    assert len(ff)==5 and sum(int(q['patients']) for q in ff)==80
    for k in ['auroc','auprc','macro_f1','f1','nez_f1','accuracy','ez_fraction_bias','mrr','hit_at_1']:
        expected=sum(float(q[k])*int(q['patients']) for q in ff)/80
        assert math.isclose(float(r[k]),expected,abs_tol=1e-12,rel_tol=0),(r['model'],k)
for p in pub.glob('*.csv'):
    with p.open(newline='',encoding='utf-8') as f:columns=next(csv.reader(f))
    assert not set(columns)&{'patient','channel','indices','path','checkpoint','probability','prediction','y'}
print('ALL8_ARMS_PUBLIC_HASHES_SOURCE_PRIVACY_AND_WEIGHTED_FOLDS_PASS')
