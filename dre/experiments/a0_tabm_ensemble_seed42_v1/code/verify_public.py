"""Read-only publication preflight: hashes, aggregates and private-row exclusion."""
import csv
import hashlib
import json
from pathlib import Path


def main():
    root=Path(__file__).resolve().parents[1]
    read=lambda name:json.loads((root/'audit'/name).read_text(encoding='utf-8'))
    rows=lambda name:list(csv.DictReader((root/'results'/name).open(encoding='utf-8',newline='')))
    sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
    status=read('RUN_STATUS.json'); gate=read('VALIDATION_GATE.json')
    assert status['status']=='COMPLETE_DEVELOPMENT_GATE_FAILED'
    assert status['terminal']==gate['terminal']=='TABM_NOT_SUPPORTED'
    assert status['new_student_models']==10 and not status['outer_evaluation'] and not status['A0_retrained']
    for name,digest in status['binding'].items():
        path=root/'PROTOCOL_LOCK.json' if name=='protocol' else root/'code'/name
        assert sha(path)==digest,(name,'source binding mismatch')
    assert sha(root/'code'/'finalize.py')==status['finalizer_sha256']
    smoke=read('TEST_AND_SMOKE_AUDIT.json')
    assert sha(root/'tests'/'test_tabm.py')==smoke['tests_sha256']
    for name in ['A0_REPRODUCTION.json','DATA_PREPROCESSING_AUDIT.json','TEST_AND_SMOKE_AUDIT.json']:
        item=read(name); assert item['status']=='PASS' and item['binding']==status['binding']
    assert read('ARTIFACT_INTEGRITY_AUDIT.json')['status']=='PASS'
    selection=read('SELECTION_AUDIT.json')['models']
    assert len(selection)==10 and {(r['arm'],r['fold']) for r in selection}=={(a,f) for a in ['W1','W2'] for f in range(1,6)}
    hist=rows('TRAINING_HISTORY.csv'); assert len(hist)==76
    for r in selection:
        group=[h for h in hist if h['arm']==r['arm'] and int(h['fold'])==r['fold']]
        assert len(group)==r['completed_epochs'] and [int(h['epoch']) for h in group]==list(range(1,len(group)+1))
    summary={r['arm']:r for r in rows('VALIDATION_SUMMARY.csv')}
    fold=rows('VALIDATION_BY_FOLD.csv'); assert len(fold)==15
    boot=rows('PAIRED_BOOTSTRAP.csv'); assert len(boot)==33
    for r in boot:
        left,right=r['comparison'].split('-'); metric=r['metric']
        assert abs(float(r['delta'])-(float(summary[left][metric])-float(summary[right][metric])))<1e-12
        assert int(r['draws'])==10000 and int(r['seed'])==42 and int(r['cluster_patients'])==47 and int(r['paired_patient_fold_cells'])==65
        assert float(r['ci_low'])<=float(r['ci_high'])
    for arm in summary:
        assert int(summary[arm]['unique_patients'])==47 and int(summary[arm]['patient_fold_cells'])==65
        for metric in [r['metric'] for r in boot if r['comparison']=='W2-A0']:
            assert abs(float(summary[arm][metric])-sum(float(r[metric]) for r in fold if r['arm']==arm)/5)<1e-12
    assert abs(float(summary['A0']['macro_f1'])-.6380797828499001)<1e-12
    assert not gate['pass'] and len(gate['failed_conditions'])==4
    for directory in ['audit','results']:
        for path in (root/directory).iterdir():
            assert path.suffix in ['.csv','.json'] and 'PRIVATE' not in path.name
            if path.suffix=='.csv':
                with path.open(encoding='utf-8',newline='') as f:header=next(csv.reader(f))
                assert not set(header)&{'patient','patient_id','channel','channel_id','y_nez','score_nez','edf_name'}
    print('PUBLIC_PREFLIGHT_PASS: source hashes, 10 students / 76 epochs, 33 contrasts; no private prediction columns')


if __name__=='__main__':main()
