"""Stdlib release hash/privacy/aggregate consistency, not private replay."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
ARMS=['C0_CONCAT','C1_BALANCED','C2_INTERACTION']
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']


def table(p):
    with p.open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))


def main():
    protocol=json.loads((HERE/'PROTOCOL_LOCK.json').read_text());assert protocol['models']==ARMS and protocol['cohort']['patients']==80
    result=HERE/'results/internal_v1';admitted=(result/'ADMISSION.json').exists();complete=(result/'INTERNAL_TERMINAL.json').exists()
    if admitted:
        a=json.loads((result/'ADMISSION.json').read_text());assert a['status']=='PASS' and a['tests_passed']==4
        assert len(a['actual_control_replays'])==5 and all(r['maximum_replay_error']<=1e-6 and r['all201_CAL_decisions_exact'] for r in a['actual_control_replays'])
        assert len(a['largest_complete_FIT_backwards'])==3 and all(r['all7_coordinates_retained'] and r['exact_original_Raw_summary'] and r['discarded_legal_FIT_steps']==3 for r in a['largest_complete_FIT_backwards'])
        assert a['formal_epochs_discarded']==a['new_Outer_passes']==0 and a['shared_CNN_Gate_head_initial_state_exact']
    if complete:
        t=json.loads((result/'INTERNAL_TERMINAL.json').read_text());v=json.loads((result/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())
        assert t['status']=='COMPLETE_VERIFIED_BALANCED_7D_INTERNAL' and v['status']=='PASS'
        assert t['actual_new_fits']==10 and t['reused_controls']==5 and not t['goal_established'] and t['new_Outer_passes']==0
        assert v['maximum_replay_error']<=1e-6 and v['independent_interval_checks']==54 and v['maximum_interval_difference']<1e-12 and v['all7_features_unchanged']
        means=table(result/'DEVELOPMENT_MAIN.csv');cells=table(result/'DEVELOPMENT_CELLS.csv');folds=table(result/'DEVELOPMENT_FOLD.csv');sd=table(result/'DEVELOPMENT_FOLD_SD.csv')
        assert len(means)==3 and len(cells)==len(folds)==15 and {r['arm'] for r in means}==set(ARMS)
        assert len(table(result/'TRAIN_HISTORY.csv'))==t['epochs']==sum(int(r['epochs']) for r in cells)
        for a in means:
            z=[r for r in folds if r['arm']==a['arm']];n={r['fold']:int(r['patients']) for r in cells if r['arm']==a['arm']};assert sum(n.values())==65 and set(n.values())=={13}
            s=next(r for r in sd if r['arm']==a['arm'])
            for m in METRICS:
                assert abs(math.fsum(float(r[m])*n[r['fold']] for r in z)/65-float(a[m]))<1e-12
                assert abs(statistics.stdev(float(r[m]) for r in z)-float(s[m]))<1e-12
            assert abs(float(a['macro_f1'])-.5*(float(a['f1'])+float(a['nez_f1'])))<1e-12
        ci=table(result/'DEVELOPMENT_INTERVALS.csv');assert len(ci)==54
        assert all(int(r['appearances'])==65 and int(r['unique_patients'])==47 and int(r['draws'])==10000 for r in ci)
        points={r['arm']:r for r in means}
        for r in ci:
            expected=float(points[r['left']][r['metric']])-(float(points[r['right']][r['metric']]) if r['kind']=='paired' else 0)
            assert abs(float(r['mean'])-expected)<1e-12
        c=json.loads((result/'CANDIDATE_SELECTION.json').read_text());assert c['selected']==t['selected'] and c['new_qualified']==t['final_stage_warranted'] and not c['new_Outer_used']
    outer=HERE/'results/final_stage_v1';outer_complete=(outer/'OUTER_STATUS.json').exists()
    if outer_complete:
        t=json.loads((outer/'OUTER_STATUS.json').read_text());v=json.loads((outer/'INDEPENDENT_OUTER_VERIFICATION.json').read_text());d=json.loads((outer/'FINAL_DECISION.json').read_text())
        assert t['status']=='COMPLETE_VERIFIED_FULL80_BALANCED_7D' and v['status']=='PASS' and t['patients']==80 and t['features']==7
        assert t['selected']=='C2_INTERACTION' and v['patient_rows']==240 and v['patient_metric_checks']==2160 and v['interval_checks']==54 and v['maximum_interval_difference']<1e-12
        assert v['original7_features_unchanged'] and v['one_classification_head'] and not t['independent_confirmation']
        rows=table(outer/'MAIN_RESULTS.csv');points={r['arm']:r for r in rows};assert set(points)==set(ARMS)
        a=float(points['C2_INTERACTION']['macro_f1']);b=float(points['C0_CONCAT']['macro_f1'])
        assert abs(a-t['selected_patient_Macro_F1'])<1e-12 and abs(b-t['control_patient_Macro_F1'])<1e-12
        assert t['target_met']==(d['status']=='POINT_TARGET_MET')==(a>.64 and a>b)
        for r in rows:assert abs(float(r['macro_f1'])-.5*(float(r['f1'])+float(r['nez_f1'])))<1e-12
        ci=table(outer/'INTERVALS.csv');assert len(ci)==54
        for r in ci:
            expected=float(points[r['left']][r['metric']])-(float(points[r['right']][r['metric']]) if r['kind']=='paired' else 0)
            assert abs(float(r['mean'])-expected)<1e-12 and int(r['appearances'])==int(r['unique_patients'])==80 and int(r['draws'])==10000
        cov=table(outer/'METRIC_COVERAGE.csv');assert len(cov)==27 and all(int(r['finite_patients'])==int(r['total_patients'])==80 for r in cov)
        census=table(outer/'RAW_COVERAGE.csv');assert len(census)==10
        for arm in ARMS[1:]:
            q=[r for r in census if r['arm']==arm];assert len(q)==5 and sum(int(r['patients']) for r in q)==80 and sum(int(r['channels']) for r in q)==7635 and sum(int(r['valid_windows']) for r in q)==1471965
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    for p in (HERE/'results').rglob('*.csv'):
        with p.open(encoding='utf-8',newline='') as f:assert not(set(csv.DictReader(f).fieldnames)&forbidden),p.name
    files=sorted(p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes')
        and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json')
    assert all(p.stat().st_size<10_000_000 for p in files)
    manifest=dict(status='PASS',scope='public_hashes_and_aggregate_consistency_not_private_replay',admission=admitted,internal=complete,outer=outer_complete,
        files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    target=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv:target.write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n',encoding='utf-8')
    else:assert json.loads(target.read_text())==manifest
    print('BALANCED_7D_PUBLIC_RELEASE_PASS',len(files),'files;admission',admitted,'internal',complete,'outer',outer_complete)


if __name__=='__main__':main()
