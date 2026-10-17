"""Stdlib source hashes, privacy and public aggregate checks, not private replay."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ARMS=['C2_REFERENCE','G0_GLOBAL','G1_PATIENT']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def table(path):
    with path.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def main():
    protocol=json.loads((HERE/'PROTOCOL_LOCK.json').read_text());assert protocol['models']==ARMS and protocol['cohort']['patients']==80
    root=HERE/'results/internal_v1';admitted=(root/'ADMISSION.json').exists();complete=(root/'INTERNAL_TERMINAL.json').exists()
    if admitted:
        a=json.loads((root/'ADMISSION.json').read_text());assert a['status']=='PASS' and a['tests_passed']==5
        assert a['formal_epochs_discarded']==a['new_Outer_passes']==0
        assert len(a['initial_checks'])==10 and all(z['initial_replay_error']<=1e-6 and z['all201_decisions_exact'] for z in a['initial_checks'])
        assert len(a['actual_FIT_checks'])==2
        for z in a['actual_FIT_checks']:
            assert z['exact_full_model_cached_initial_logits'] and z['full_model_has_one_scalar_readout'] and z['readout_unchanged'] and z['discarded_steps']==2
            assert z['full_parameters']==(10506 if z['arm']=='G0_GLOBAL' else 12562)
            assert z['trainable_parameters']==(128 if z['arm']=='G0_GLOBAL' else 2184)
        r=json.loads((root/'REPRESENTATION_STATUS.json').read_text())
        assert r['status']=='PASS' and r['patient_appearances']==320 and r['folds']==5 and r['new_Outer_passes']==0
        assert all(z['maximum_replay_error']<=1e-6 and z['all201_decisions_exact'] and z['body_unchanged'] for z in r['checks'])
    if complete:
        t=json.loads((root/'INTERNAL_TERMINAL.json').read_text());v=json.loads((root/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())
        assert t['status']=='COMPLETE_VERIFIED_CONTEXT_AFFINE_INTERNAL' and v['status']=='PASS'
        assert t['actual_new_fits']==10 and t['reused_controls']==5 and not t['goal_established'] and t['new_Outer_passes']==0
        assert v['patient_metric_rows']==195 and v['independent_metric_checks']==1755 and v['independent_interval_checks']==54
        assert v['maximum_replay_error']<=1e-6 and v['maximum_interval_difference']<1e-12 and v['all7_features_unchanged']
        means=table(root/'DEVELOPMENT_MAIN.csv');cells=table(root/'DEVELOPMENT_CELLS.csv')
        folds=table(root/'DEVELOPMENT_FOLD.csv');sd=table(root/'DEVELOPMENT_FOLD_SD.csv')
        assert len(means)==3 and len(cells)==len(folds)==15 and {z['arm'] for z in means}==set(ARMS)
        assert len(table(root/'TRAIN_HISTORY.csv'))==t['epochs']==sum(int(float(z['epochs'])) for z in cells)
        for z in means:
            f=[r for r in folds if r['arm']==z['arm']];counts={r['fold']:int(r['patients']) for r in cells if r['arm']==z['arm']}
            assert sum(counts.values())==65 and set(counts.values())=={13}
            s=next(r for r in sd if r['arm']==z['arm'])
            for metric in METRICS:
                assert abs(math.fsum(float(r[metric])*counts[r['fold']] for r in f)/65-float(z[metric]))<1e-12
                assert abs(statistics.stdev(float(r[metric]) for r in f)-float(s[metric]))<1e-12
            assert abs(float(z['macro_f1'])-.5*(float(z['f1'])+float(z['nez_f1'])))<1e-12
        points={r['arm']:r for r in means};ci=table(root/'DEVELOPMENT_INTERVALS.csv');assert len(ci)==54
        for z in ci:
            assert int(z['appearances'])==65 and int(z['unique_patients'])==47 and int(z['draws'])==10000
            expected=float(points[z['left']][z['metric']])-(float(points[z['right']][z['metric']]) if z['kind']=='paired' else 0)
            assert abs(float(z['mean'])-expected)<1e-12
        c=json.loads((root/'CANDIDATE_SELECTION.json').read_text())
        assert c['selected']==t['selected'] and c['new_qualified']==t['final_stage_warranted'] and not c['new_Outer_used']
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    for path in (HERE/'results').rglob('*.csv'):
        with path.open(encoding='utf-8',newline='') as f:assert not(set(csv.DictReader(f).fieldnames)&forbidden),path.name
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes')
        and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json')
    assert all(p.stat().st_size<10_000_000 for p in files)
    manifest=dict(status='PASS',scope='public_hashes_and_aggregate_consistency_not_private_replay',admission=admitted,internal=complete,
        files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    path=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv:path.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    else:assert json.loads(path.read_text())==manifest
    print('CONTEXT_AFFINE_PUBLIC_RELEASE_PASS',len(files),'files;admission',admitted,'internal',complete)


if __name__=='__main__':main()
