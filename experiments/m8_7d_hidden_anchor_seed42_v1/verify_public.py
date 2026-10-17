"""Anonymous aggregate consistency and exact release bytes; no private replay."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ARMS=['C2_REFERENCE','P_WARM_ONLY','H0_SCRATCH','H1_WARM_JOINT','H2_WARM_FIXED']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def table(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def main():
    protocol=json.loads((HERE/'PROTOCOL_LOCK.json').read_text());assert protocol['arms']==ARMS
    result=HERE/'results/internal_v1';admitted=(result/'ADMISSION.json').exists();complete=(result/'INTERNAL_TERMINAL.json').exists()
    if admitted:
        a=json.loads((result/'ADMISSION.json').read_text());assert a['status']=='PASS' and a['tests_passed']==5
        assert len(a['reference_checks'])==5 and all(r['maximum_replay_error']<=1e-6 and r['all201_decisions_exact'] for r in a['reference_checks'])
        assert len(a['actual_FIT_gradient_checks'])==3 and a['all3_initial_states_exact']
        assert all(r['parameters']==13961 and r['all7_coordinates_retained'] and r['original_Raw_summary_exact'] and r['original_physiological_normalization_exact'] and r['discarded_steps']==3 for r in a['actual_FIT_gradient_checks'])
        assert a['new_Outer_passes']==a['formal_epochs_discarded']==0
    if complete:
        t=json.loads((result/'INTERNAL_TERMINAL.json').read_text());v=json.loads((result/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())
        assert t['status']=='COMPLETE_VERIFIED_HIDDEN_ANCHOR_INTERNAL' and v['status']=='PASS'
        assert t['actual_new_fits']==20 and t['warm_fits']==5 and t['joint_fits']==15 and t['reused_controls']==5
        assert t['epochs']==t['warm_epochs']+t['joint_epochs'] and not t['goal_established'] and t['new_Outer_passes']==0
        assert v['patient_metric_rows']==325 and v['independent_metric_checks']==2925 and v['independent_interval_checks']==135 and v['maximum_interval_difference']<1e-12
        assert v['maximum_replay_error']<=1e-6 and v['all7_features_unchanged'] and v['all_stage_transfer_and_frozen_states_verified']
        means=table(result/'DEVELOPMENT_MAIN.csv');cells=table(result/'DEVELOPMENT_CELLS.csv');folds=table(result/'DEVELOPMENT_FOLD.csv');sd=table(result/'DEVELOPMENT_FOLD_SD.csv');hist=table(result/'TRAIN_HISTORY.csv')
        assert len(means)==5 and len(cells)==len(folds)==25 and {r['arm'] for r in means}==set(ARMS)
        assert len(hist)==t['epochs']==sum(int(r['epochs']) for r in cells)
        assert sum(r['phase']=='phys' for r in hist)==t['warm_epochs'] and sum(r['phase']=='joint' for r in hist)==t['joint_epochs']
        for r in cells:
            assert int(r['selected_epoch'])>=1
            if r['arm']=='C2_REFERENCE':assert int(r['epochs'])==0 and r['execution_kind']=='REUSED_EXACT_C2'
            else:assert int(r['epochs'])>=int(r['selected_epoch']) and int(r['parameters'])==13961 and int(r['trainable'])==({'P_WARM_ONLY':1537,'H2_WARM_FIXED':12424}.get(r['arm'],13961))
        for r in means:
            ff=[q for q in folds if q['arm']==r['arm']];s=next(q for q in sd if q['arm']==r['arm'])
            for m in METRICS:
                assert abs(math.fsum(float(q[m]) for q in ff)/5-float(r[m]))<1e-12
                assert abs(statistics.stdev(float(q[m]) for q in ff)-float(s[m]))<1e-12
            assert abs(float(r['macro_f1'])-.5*(float(r['f1'])+float(r['nez_f1'])))<1e-12
        ci=table(result/'DEVELOPMENT_INTERVALS.csv');assert len(ci)==135;points={r['arm']:r for r in means}
        for r in ci:
            expected=float(points[r['left']][r['metric']])-(float(points[r['right']][r['metric']]) if r['kind']=='paired' else 0)
            assert abs(float(r['mean'])-expected)<1e-12 and int(r['appearances'])==65 and int(r['unique_patients'])==47 and int(r['draws'])==10000
        c=json.loads((result/'CANDIDATE_SELECTION.json').read_text());assert c['selected']==t['selected'] and c['new_qualified']==t['final_stage_warranted'] and not c['new_Outer_used']
        assert t['selected']!='P_WARM_ONLY'
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    for p in (HERE/'results').rglob('*.csv'):
        with p.open(encoding='utf-8',newline='') as f:assert not(set(csv.DictReader(f).fieldnames)&forbidden),p.name
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes') and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json')
    assert all(p.stat().st_size<10_000_000 for p in files)
    manifest=dict(status='PASS',scope='public_hashes_aggregate_consistency_not_private_replay',admission=admitted,internal=complete,files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    target=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv:target.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    else:assert json.loads(target.read_text())==manifest
    print('HIDDEN_ANCHOR_PUBLIC_RELEASE_PASS',len(files),'files;admission',admitted,'internal',complete)


if __name__=='__main__':main()
