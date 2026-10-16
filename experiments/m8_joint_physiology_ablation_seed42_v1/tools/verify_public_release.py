"""Local standard-library release hash/schema/count checks; no trusted data needed."""
import csv
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1];RESULTS=ROOT/'results'
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def jr(path):return json.loads(path.read_text(encoding='utf-8'))
def main():
    proof=jr(RESULTS/'INDEPENDENT_COMPLETION_AUDIT.json');assert proof['status']=='PASS'
    for name,h in proof['files'].items():
        path=(RESULTS/name).resolve();assert RESULTS.resolve() in path.parents and sha(path)==h,name
    training=jr(RESULTS/'TRAINING_VERIFICATION.json');assert training['status']=='PASS' and training['maximum_logit_error']==0
    for name,h in training['source'].items():assert sha(ROOT/name)==h,name
    intervals=jr(RESULTS/'INDEPENDENT_ALL_INTERVALS_AUDIT.json');assert intervals['status']=='PASS'
    assert (intervals['paired_metric_intervals'],intervals['adjusted_primary_intervals'],intervals['absolute_metric_intervals'])==(1116,84,288)
    assert intervals['source_sha256']==sha(ROOT/'tools/verify_all_intervals.py')
    grouped=jr(RESULTS/'INDEPENDENT_GROUPED_AGGREGATES_AUDIT.json')
    assert grouped['status']=='PASS' and grouped['verified_numeric_fields']==5508
    assert grouped['source_sha256']==sha(ROOT/'tools/verify_grouped_aggregates.py')
    for code in ['011','101','110','111']:assert jr(RESULTS/('ADMISSION_'+code+'.json'))['status']=='PASS'
    final=jr(RESULTS/'COMPLETION_STATUS.json');assert final['patient_rows']==1280 and final['choices_before_Outer']=={'D1':'000','M8':'000'} and not final['Outer_selection']
    forbidden={'patient','channel','path','checkpoint','prediction','probability','y','indices'}
    count=0
    for path in RESULTS.rglob('*.csv'):
        with path.open(newline='',encoding='utf-8') as f:
            reader=csv.DictReader(f);assert not forbidden.intersection(reader.fieldnames),path
            rows=list(reader)
        if path.name=='MAIN_RESULTS.csv':
            assert len(rows)==16 and len({(r['config'],r['head']) for r in rows})==16
            assert {r['config'] for r in rows}=={f'{i:03b}' for i in range(8)}
            assert all(int(r['appearances'])==(80 if path.parent.name=='outer' else 65) for r in rows)
        if path.name=='MODEL_SELECTION.csv':assert len(rows)==40 and sum(r['reused']=='True' for r in rows)==20
        if path.name=='NEW_TRAINING_HISTORY.csv':assert len(rows)==training['new_complete_epochs']==434
        count+=1
    print('JOINT_PUBLIC_RELEASE_HASHES_SCHEMAS_AND_16_ROUTE_TABLES_PASS',count)

if __name__=='__main__':main()
