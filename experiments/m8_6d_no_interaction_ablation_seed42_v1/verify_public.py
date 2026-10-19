"""Portable aggregate-only release verifier; never imports trusted private inputs."""
import csv
import hashlib
import json
import math
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent;RESULTS=HERE/'results'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def obj(n):return json.loads((RESULTS/n).read_text(encoding='utf-8'))
def table(n):
    with (RESULTS/(n+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def main():
    lock=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));arms=lock['arms'];metrics=lock['metrics']
    assert len(arms)==14 and len(set(arms))==14 and 'WO_LATE_GATE' not in arms and 'WO_HEAD_INTERACTION' not in arms
    manifest=obj('PACKAGE_MANIFEST.json');assert manifest['private_artifacts_exported'] is False
    for n,h in manifest['files'].items():
        assert Path(n).name==n and sha(RESULTS/n)==h
        assert Path(n).suffix not in ['.pkl','.pt','.npz','.npy','.log','.err']
    admission=obj('ADMISSION.json');assert admission['status']=='PASS' and admission['new_fits_started']==0
    assert admission['tests_passed']==5 and len(admission['reference_replays'])==15 and len(admission['actual_full_patient_backward'])==11
    assert all(r['initial_state_exact'] and r['validation_logits_bit_exact'] for r in admission['reference_replays'])
    pushed=obj('SOURCE_PUSH_BEFORE_TRAIN.json');assert pushed['source']==admission['source'] and pushed['source_pushed_before_formal_fits']
    for n,h in pushed['source'].items():assert sha(HERE/n)==h
    status=obj('OUTER_STATUS.json');assert status['status']=='COMPLETE_VERIFIED_NI_14_GROUP_FIVE_FOLD_OUTER'
    assert status['logical_model_cells']==70 and status['new_neural_fits']==55 and status['reused_cells']==15
    assert status['patient_rows']==1120 and status['metric_checks']==10080 and status['scientific_status']=='EXPLORATORY_REPEATED_TEST'
    assert obj('FREEZE_STATUS.json')['freeze_sha256']==status['freeze_sha256']
    assert obj('PREDICTION_STATUS.json')['score_seal_sha256']==status['score_seal_sha256']
    independent=obj('INDEPENDENT_OUTER_VERIFICATION.json')
    assert independent['status']=='PASS' and independent['patient_metric_checks']==10080 and independent['nominal_interval_checks']==243 and independent['macro_family_interval_checks']==13
    cells=table('TRAIN_CELLS');assert len(cells)==70 and {(r['arm'],int(r['fold'])) for r in cells}=={(a,f) for a in arms for f in range(1,6)}
    assert sum(r['reused']=='True' for r in cells)==15
    histories=table('TRAIN_HISTORY');assert len(histories)==obj('TRAIN_STATUS.json')['epochs']
    for r in cells:
        if r['reused']=='True':
            assert r['arm'] in lock['references'] and int(r['epochs'])==0;continue
        h=[q for q in histories if q['arm']==r['arm'] and q['fold']==r['fold']]
        assert 6<=len(h)==int(r['epochs'])<=30 and [int(q['epoch']) for q in h]==list(range(1,len(h)+1))
        winner=max(h,key=lambda q:(float(q['macro_f1']),float(q['ez_f1']),-int(q['epoch'])))
        assert winner['epoch']==r['epoch'] and winner['threshold']==r['threshold']
    mainrows=table('MAIN_RESULTS');assert [r['arm'] for r in mainrows]==arms
    main={r['arm']:r for r in mainrows};folds=table('FOLD_RESULTS');sd=table('FOLD_SD');raw=table('RAW_COVERAGE')
    assert len(folds)==len(raw)==70 and len(sd)==14
    for a in arms:
        r=main[a];assert int(r['patients'])==80
        assert abs(float(r['macro_f1'])-.5*(float(r['f1'])+float(r['nez_f1'])))<1e-12
        f=sorted([q for q in folds if q['arm']==a],key=lambda q:int(q['fold']))
        counts=np.array([int(q['patients']) for q in f]);assert counts.tolist()==[16,16,17,15,16]
        qsd=next(q for q in sd if q['arm']==a)
        for m in metrics:
            vals=np.array([float(q[m]) for q in f]);assert np.isfinite(vals).all()
            assert abs(np.average(vals,weights=counts)-float(r[m]))<1e-12
            assert abs(np.std(vals,ddof=1)-float(qsd[m]))<1e-12
        rr=[q for q in raw if q['arm']==a]
        assert sum(int(q['patients']) for q in rr)==80 and sum(int(q['channels']) for q in rr)==7635
        assert sum(int(q['valid_windows']) for q in rr)==(0 if a=='PHYSIOLOGY_ONLY' else 1471965)
    ci=table('INTERVALS');family=table('MACRO_FAMILY_INTERVALS');assert len(ci)==243 and len(family)==13
    for r in ci:
        m=r['metric'];assert m in metrics and int(r['patients'])==80 and int(r['draws'])==10000
        if r['kind']=='absolute':point=float(main[r['contrast']][m])
        else:point=float(main['FULL_F0'][m])-float(main[r['contrast'].removeprefix('FULL_minus_')][m])
        assert abs(float(r['mean'])-point)<1e-12 and float(r['low'])<=float(r['high'])
    for r in family:
        nominal=next(q for q in ci if q['metric']=='macro_f1' and q['contrast']==r['contrast'] and q['kind']=='paired')
        assert int(r['comparisons'])==13 and abs(float(r['tail'])-.05/26)<1e-15
        assert abs(float(r['mean'])-float(nominal['mean']))<1e-12
        assert float(r['low'])<=float(nominal['low']) and float(r['high'])>=float(nominal['high'])
    assert abs(float(main['FULL_F0']['macro_f1'])-.6538278218104221)<1e-12
    assert all(int(r['finite_patients'])==80 for r in table('METRIC_COVERAGE'))
    for r in table('SOURCE_RESULTS'):
        assert (r['suppressed']=='True')==(int(r['patients'])<5)
        if r['suppressed']=='True':assert all(math.isnan(float(r[m])) for m in metrics)
    for p in RESULTS.glob('*.csv'):
        with p.open(encoding='utf-8',newline='') as f:
            assert not(set(next(csv.reader(f)))&{'patient','patient_id','channel','channel_id','archive_indices','score_nez','prediction','probability','y','path'})
    historical=table('HISTORICAL_CLASSIFIER_REFERENCE');assert len(historical)==14
    parent=HERE.parent/'m8_6d_no_variance_ablation_baselines_seed42_v1/results/MAIN_RESULTS.csv'
    with parent.open(encoding='utf-8',newline='') as f:old={r['arm']:r for r in csv.DictReader(f)}
    assert sha(parent)==obj('HISTORICAL_CLASSIFIER_AUDIT.json')['source_table_sha256']
    assert all(r==old[r['arm']] for r in historical)
    print('NI_14_GROUP_PUBLIC_SOURCE_METRICS_INTERVALS_AND_PRIVACY_PASS')

if __name__=='__main__':main()
