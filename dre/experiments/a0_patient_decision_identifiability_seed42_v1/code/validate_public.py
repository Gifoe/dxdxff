"""Public release validation: aggregates only, binding consistency, report parity."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',type=Path,required=True);a=p.parse_args();root=a.experiment
    audits={q.name:json.loads(q.read_text(encoding='utf-8')) for q in (root/'audit').glob('*.json')}
    forbidden={'patient','patient_id','subject_id','channel','channel_name','y_nez','score_nez','oracle_threshold','coefficients'}
    rows={}
    for q in (root/'results').glob('*.csv'):
        with q.open(encoding='utf-8',newline='') as f:
            reader=csv.DictReader(f);assert not forbidden.intersection(reader.fieldnames);r=list(reader);assert r;rows[q.name]=r
            for row in r:
                if 'center' in row and 'unique_patients' in row:assert int(row['unique_patients'])>=5
                if 'development_unique_patients' in row:assert int(row['development_unique_patients'])>=5
    for q in root.rglob('*'):
        if q.is_file():assert q.suffix not in ['.pt','.npz','.npy','.pkl','.log','.err','.edf','.h5','.pyc']
    status=audits['RUN_STATUS.json'];assert status['status']=='COMPLETE' and status['outer_test_accessed'] is False
    assert sha(root/'PROTOCOL_LOCK.json')==status['binding']['protocol']
    for name,digest in status['binding']['code'].items():assert sha(root/'code'/name)==digest
    assert sha(root/'audit/REGULARIZATION_SELECTION_AUDIT.json')==status['regularization_sha256']
    replay=audits['A0_REPRODUCTION.json'];assert replay['status']=='PASS' and not replay['retrained']
    assert abs(replay['metrics']['macro_f1']-.6380797828499001)<1e-12
    original=[r for r in rows['PATIENT_ORACLE_SUMMARY.csv'] if r['method']=='baseline'][0]
    ceiling=[r for r in rows['PATIENT_ORACLE_SUMMARY.csv'] if r['method']=='oracle'][0]
    assert abs(float(original['macro_f1'])-replay['metrics']['macro_f1'])<1e-12
    decision=audits['FINAL_DECISION_MATRIX.json'];assert decision['terminal']==status['terminal']=='PATIENT_ORACLE_HEADROOM_NOT_RECOVERABLE'
    assert abs(float(ceiling['macro_f1'])-decision['A']['oracle_macro_f1'])<1e-12
    assert not decision['B8']['bias_pass'] and not decision['B8']['direction_pass']
    assert not decision['half_capacity']['bias_pass'] and not decision['half_capacity']['direction_pass']
    integrity=[audits[f'FINAL_INTEGRITY_FOLD{i}.json'] for i in range(1,6)]
    assert sum(v['episodes_replayed_exactly'] for v in integrity)==2015 and max(v['max_metric_drift'] for v in integrity)==0
    assert all(v['status']=='PASS' and v['all_original_A0_checkpoint_and_bank_hashes_unchanged'] for v in integrity)
    for r in audits['CLINICAL_SOURCE_INVENTORY.json']['label_source_counts']:assert r['patients']>=5
    assert sum(int(r['patients']) for r in rows['LABEL_SOURCE_COUNTS.csv'])==80
    assert sum(int(r['channels']) for r in rows['LABEL_SOURCE_COUNTS.csv'])==7635
    required=['README.md','SOURCE_AUDIT.md','PROTOCOL_LOCK.json','IMPLEMENTATION_AUDIT.md','FINAL_REPORT.md','CLINICAL_LABEL_PROVENANCE.md','LABEL_AUDIT_LIMITATIONS.md']
    assert all((root/n).exists() for n in required)
    report=(root/'FINAL_REPORT.md').read_text(encoding='utf-8')
    assert all(k in report for k in ['0.706285','0.638080','0.068205','0.001739','0.019536','2,015'])
    print('PUBLIC_RELEASE_VALIDATION_PASS',len(audits),'JSON audits',len(rows),'aggregate CSVs','2015 exact private replays',flush=True)

if __name__=='__main__':main()
