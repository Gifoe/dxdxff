"""Post-transfer hashes, source bindings, exact cell counts and privacy checks."""
import csv
import hashlib
import json
from pathlib import Path
root=Path(__file__).resolve().parents[1];pub=root/'phase2_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def jr(p):return json.loads(p.read_text(encoding='utf-8'))
for name,h in jr(pub/'PUBLIC_SHA256.json').items():assert sha(pub/name)==h,name
for name in ['NUMERICAL_CONTROL_EXTRACTION_STATUS','NONLINEAR_SCREEN_STATUS']:
    for src,h in jr(pub/(name+'.json'))['source'].items():assert sha(root/src)==h,src
syn=jr(pub/'INTEGRATED_MODEL_SYNTHETIC_AUDIT.json');assert syn['status']=='PASS'
assert syn['source_sha256']==sha(root/'integrated_model.py') and syn['test_sha256']==sha(root/'test_integrated_model.py')
assert syn['total_parameters']==11690 and syn['relative_to_legacy']<=1.1
v=jr(pub/'NONLINEAR_INDEPENDENT_VERIFICATION.json');assert v['status']=='PASS' and v['maximum_probability_replay_error']==0
assert v['verifier_sha256']==sha(root/'tools/verify_nonlinear.py')
assert v['cells']=={'patient':705,'center':520}
with (pub/'NONLINEAR_INNER_AND_CENTER_METRICS.csv').open(newline='',encoding='utf-8') as f:cells=list(csv.DictReader(f))
assert len(cells)==1225 and len({(r['outer_fold'],r['task'],r['split'],r['variant']) for r in cells})==1225
selection=jr(pub/'PER_FIT_FEATURE_SELECTION.json');assert selection['original_validation_Outer_excluded']
assert selection['selection_freeze_sha256']==jr(root/'M8_INTEGRATION_LOCK.json')['FIT_feature_freeze_sha256']
assert all(not r['deleted_old_group_indices'] for r in selection['folds'].values())
assert {f:r['added_new_group_indices'] for f,r in selection['folds'].items()}=={'1':[],'2':[17],'3':[],'4':[],'5':[]}
for p in pub.glob('*.csv'):
    with p.open(encoding='utf-8',newline='') as f:headers=next(csv.reader(f))
    assert not set(headers)&{'patient','channel','path','id','y','label','prediction','probability','checkpoint'}
print('PHASE2_PUBLIC_HASH_SOURCE_COUNTS_SELECTION_AND_PRIVACY_PASS')
