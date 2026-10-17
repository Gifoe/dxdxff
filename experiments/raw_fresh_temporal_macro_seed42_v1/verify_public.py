"""Public consistency and release hashes, not a substitute for private replay."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
ARMS=['NATIVE','FROZEN_PRE','COMPACT_PRE','PRIOR_PRESTATE','FRESH_BASE','FRESH_POINTWISE','FRESH_TEMPORAL']


def table(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def main():
    protocol=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));assert protocol['arms']==ARMS
    result=HERE/'results/internal_v1';admitted=(result/'ADMISSION.json').exists();complete=(result/'INTERNAL_TERMINAL.json').exists()
    if admitted:
        admission=json.loads((result/'ADMISSION.json').read_text());assert admission['status']=='PASS' and admission['tests_passed']==5
        assert admission['formal_updates']==0 and not admission['original_Development_Outer_used']
        assert all(r['optimizer_updates']==0 and r['parameters']==(8433 if r['arm']=='FRESH_BASE' else 9417) and r['all_parameter_gradients_finite'] for r in admission['largest_complete_patient_backward'])
        assert len(admission['initial_states'])==5 and all(r['shared_CNN_head_states_exact'] for r in admission['initial_states'])
        assert all(r['exact_initial_logits_match'] for r in admission['largest_complete_patient_backward'])
        assert all(r['adapter_output_projection_gradient_nonzero'] for r in admission['largest_complete_patient_backward'] if r['arm']!='FRESH_BASE')
    if complete:
        terminal=json.loads((result/'INTERNAL_TERMINAL.json').read_text());audit=json.loads((result/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())
        assert terminal['status']=='COMPLETE_VERIFIED_FRESH_TEMPORAL_RAW_INTERNAL' and audit['status']=='PASS'
        assert terminal['new_Outer_passes']==0 and not terminal['goal_established'] and not audit['original_Development_Outer_used']
        assert audit['independent_CI_metric_checks']==198 and audit['maximum_bootstrap_difference']<1e-12
        assert audit['maximum_replication_interval_error']<1e-12
        assert 15<=terminal['actual_neural_fits']==audit['actual_neural_fits']<=45
        cells=table(result/'INTERNAL_CELL_RESULTS.csv');means=table(result/'FIRST_PATH_MAIN.csv');folds=table(result/'FIRST_PATH_FOLD.csv')
        assert len(means)==7 and len(folds)==35 and {r['arm'] for r in means}==set(ARMS)
        assert len(table(result/'TRAIN_HISTORY.csv'))==terminal['epochs']
        fresh=[r for r in cells if r['arm'] in ARMS[4:]]
        assert len(fresh)==terminal['actual_neural_fits'] and all(int(r['selected_epoch'])>=1 and int(r['parameters'])==(8433 if r['arm']=='FRESH_BASE' else 9417) for r in fresh)
        points={r['arm']:r for r in means}
        for row in means:
            count={r['fold']:int(r['patients']) for r in cells if r['arm']==row['arm'] and r['screen']=='1'}
            ff=[r for r in folds if r['arm']==row['arm']];assert len(ff)==len(count)==5 and sum(count.values())==86
            for m in METRICS:assert abs(math.fsum(float(r[m])*count[r['fold']] for r in ff)/86-float(row[m]))<1e-12
            assert abs(float(row['macro_f1'])-.5*(float(row['f1'])+float(row['nez_f1'])))<1e-12
            sd=next(r for r in table(result/'FIRST_PATH_FOLD_SD.csv') if r['arm']==row['arm'])
            src=[r for r in table(result/'FIRST_PATH_SOURCE.csv') if r['arm']==row['arm']]
            assert sum(int(r['appearances']) for r in src)==86 and sum(int(r['unique_patients']) for r in src)==59
            for m in METRICS:
                assert abs(statistics.stdev(float(r[m]) for r in ff)-float(sd[m]))<1e-12
                if all(r['suppressed']=='False' for r in src):
                    assert abs(math.fsum(float(r[m])*int(r['appearances']) for r in src)/86-float(row[m]))<1e-12
        coverage=table(result/'FIRST_PATH_METRIC_COVERAGE.csv');assert len(coverage)==63
        assert all(int(r['finite_appearances'])==int(r['total_appearances'])==86 and int(r['unique_patients'])==59 for r in coverage)
        assert len(table(result/'FIRST_PATH_ABSOLUTE_CI.csv'))==63 and len(table(result/'FIRST_PATH_PAIRED_CI.csv'))==135
        for row in table(result/'FIRST_PATH_PAIRED_CI.csv'):
            assert int(row['appearances'])==86 and int(row['unique_patients'])==59 and int(row['draws'])==10000
            assert abs(float(row['mean'])-(float(points[row['left']][row['metric']])-float(points[row['right']][row['metric']])))<1e-12
        choices=table(result/'OWN_FIT_SELECTION.csv');assert len(choices)==5
        fresh_selected=[int(r['fold']) for r in choices if r['selected_arm'] in ARMS[4:]]
        assert fresh_selected==terminal['fresh_qualified_folds'] and terminal['final_stage_warranted']==bool(fresh_selected)
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','raw','score_nez','logit_nez','pred_ez','y'}
    for path in (HERE/'results').rglob('*.csv'):
        with path.open(encoding='utf-8',newline='') as stream:assert not(set(csv.DictReader(stream).fieldnames)&forbidden),path.name
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes')
                 and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json')
    assert all(p.stat().st_size<10_000_000 for p in paths)
    manifest=dict(status='PASS',scope='public_consistency_not_private_Raw_replay',admission_verified=admitted,internal_verified=complete,
        files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    destination=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv[1:]:destination.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    else:assert json.loads(destination.read_text(encoding='utf-8'))==manifest,'Public source/artifact hash mismatch'
    print('PUBLIC_FRESH_TEMPORAL_RAW_DELIVERY_PASS',len(paths),'files;admission',admitted,'internal',complete)


if __name__=='__main__':main()
