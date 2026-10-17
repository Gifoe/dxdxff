"""Public source/statistic/reuse checks; private checkpoint replay is separate."""
import csv
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
REFS=['NATIVE','FROZEN_PRE','LEGACY_PRE_MEAN','PRIOR_PRESTATE']
FAMILY=['SHORT_MEAN','SHORT_ENERGY','LONG_MEAN','LONG_ENERGY'];NOVEL=FAMILY[1:];ARMS=REFS+FAMILY


def table(path):
    with path.open(encoding='utf-8',newline='') as stream:return list(csv.DictReader(stream))


def main():
    protocol=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'));assert protocol['arms']==ARMS
    result=HERE/'results/internal_v1';admitted=(result/'ADMISSION.json').exists();complete=(result/'INTERNAL_TERMINAL.json').exists()
    if admitted:
        a=json.loads((result/'ADMISSION.json').read_text());assert a['status']=='PASS' and a['tests_passed']==6
        assert a['formal_updates']==0 and not a['original_Development_Outer_used']
        assert len(a['initial_states'])==5 and all(r['all4_collapsed_random_core_states_exact'] for r in a['initial_states'])
        assert a['reused_control_cells']==7 and a['maximum_reused_control_error']<=1e-6
        assert all(r['all201_CAL_decisions_exact'] for r in a['reused_control_checks'])
        assert len(a['largest_complete_patient_backward'])==4
        assert all(r['parameters']==(9873 if r['arm'].startswith('LONG') else 8433) and r['optimizer_updates']==0 and r['all_parameter_gradients_finite']
                   and r['matching_pool_pair_initial_logits_exact'] for r in a['largest_complete_patient_backward'])
        assert all(r['outer_filter_gradient_nonzero'] for r in a['largest_complete_patient_backward'] if r['arm'].startswith('LONG'))
    if complete:
        t=json.loads((result/'INTERNAL_TERMINAL.json').read_text());v=json.loads((result/'INDEPENDENT_INTERNAL_VERIFICATION.json').read_text())
        assert t['status']=='COMPLETE_VERIFIED_EARLY_FIR_INTERNAL' and v['status']=='PASS'
        assert t['new_Outer_passes']==0 and not t['goal_established'] and not v['original_Development_Outer_used']
        assert v['independent_CI_metric_checks']==261 and v['maximum_bootstrap_difference']<1e-12 and v['maximum_replication_interval_error']<1e-12
        assert 15<=t['actual_neural_fits']==v['actual_neural_fits']<=55 and 5<=t['reused_control_cells']==v['reused_control_cells']<=7
        assert t['family_model_cells']==t['actual_neural_fits']+t['reused_control_cells']<=60
        cells=table(result/'INTERNAL_CELL_RESULTS.csv');means=table(result/'FIRST_PATH_MAIN.csv');folds=table(result/'FIRST_PATH_FOLD.csv')
        assert len(means)==8 and len(folds)==40 and {r['arm'] for r in means}==set(ARMS)
        assert len(table(result/'TRAIN_HISTORY.csv'))==t['epochs']==sum(int(r['epochs']) for r in cells)
        fresh=[r for r in cells if r['execution_kind']=='NEW_FIT'];reused=[r for r in cells if r['execution_kind']=='REUSED_EXACT_SHORT_MEAN']
        assert len(fresh)==t['actual_neural_fits'] and len(reused)==t['reused_control_cells']
        assert all(r['arm'] in FAMILY and int(r['selected_epoch'])>=1 and int(r['parameters'])==(9873 if r['arm'].startswith('LONG') else 8433) for r in fresh+reused)
        assert all(r['arm']=='SHORT_MEAN' and int(r['epochs'])==0 and int(r['prior_training_epochs'])>=int(r['selected_epoch']) for r in reused)
        points={r['arm']:r for r in means}
        for row in means:
            count={r['fold']:int(r['patients']) for r in cells if r['arm']==row['arm'] and r['screen']=='1'}
            ff=[r for r in folds if r['arm']==row['arm']];assert len(ff)==len(count)==5 and sum(count.values())==86
            sd=next(r for r in table(result/'FIRST_PATH_FOLD_SD.csv') if r['arm']==row['arm'])
            source=[r for r in table(result/'FIRST_PATH_SOURCE.csv') if r['arm']==row['arm']]
            assert sum(int(r['appearances']) for r in source)==86 and sum(int(r['unique_patients']) for r in source)==59
            for m in METRICS:
                assert abs(math.fsum(float(r[m])*count[r['fold']] for r in ff)/86-float(row[m]))<1e-12
                assert abs(statistics.stdev(float(r[m]) for r in ff)-float(sd[m]))<1e-12
                if all(r['suppressed']=='False' for r in source):assert abs(math.fsum(float(r[m])*int(r['appearances']) for r in source)/86-float(row[m]))<1e-12
            assert abs(float(row['macro_f1'])-.5*(float(row['f1'])+float(row['nez_f1'])))<1e-12
        coverage=table(result/'FIRST_PATH_METRIC_COVERAGE.csv');assert len(coverage)==72
        assert all(int(r['finite_appearances'])==int(r['total_appearances'])==86 and int(r['unique_patients'])==59 for r in coverage)
        aa=table(result/'FIRST_PATH_ABSOLUTE_CI.csv');pp=table(result/'FIRST_PATH_PAIRED_CI.csv');fc=table(result/'FACTORIAL_CONTRASTS.csv')
        assert len(aa)==72 and len(pp)==162 and len(fc)==27
        for r in pp:
            assert int(r['appearances'])==86 and int(r['unique_patients'])==59 and int(r['draws'])==10000
            assert abs(float(r['mean'])-(float(points[r['left']][r['metric']])-float(points[r['right']][r['metric']])))<1e-12
        for r in fc:
            sm,se,lm,le=[float(points[a][r['metric']]) for a in FAMILY]
            expected={'kernel_main':.5*((lm-sm)+(le-se)),'energy_main':.5*((se-sm)+(le-lm)),'interaction':(le-lm)-(se-sm)}[r['contrast']]
            assert abs(float(r['mean'])-expected)<1e-12 and int(r['draws'])==10000 and int(r['unique_patients'])==59
        choices=table(result/'OWN_FIT_SELECTION.csv');assert len(choices)==5
        selected=[int(r['fold']) for r in choices if r['selected_arm'] in NOVEL]
        assert selected==t['fresh_qualified_folds'] and t['final_stage_warranted']==bool(selected)
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','raw','score_nez','logit_nez','pred_ez','y'}
    for p in (HERE/'results').rglob('*.csv'):
        with p.open(encoding='utf-8',newline='') as stream:assert not(set(csv.DictReader(stream).fieldnames)&forbidden),p.name
    paths=sorted(p for p in HERE.rglob('*') if p.is_file() and (p.suffix in {'.py','.ps1','.json','.csv','.md'} or p.name=='.gitattributes')
        and '__pycache__' not in p.parts and '.pytest_cache' not in p.parts and p.name!='PUBLIC_ARTIFACT_MANIFEST.json')
    assert all(p.stat().st_size<10_000_000 for p in paths)
    manifest=dict(status='PASS',scope='public_consistency_not_private_Raw_replay',admission_verified=admitted,internal_verified=complete,
        files={p.relative_to(HERE).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
    target=HERE/'PUBLIC_ARTIFACT_MANIFEST.json'
    if '--write-manifest' in sys.argv[1:]:target.write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n',encoding='utf-8')
    else:assert json.loads(target.read_text(encoding='utf-8'))==manifest,'Public source/artifact hash mismatch'
    print('PUBLIC_EARLY_FIR_DELIVERY_PASS',len(paths),'files;admission',admitted,'internal',complete)


if __name__=='__main__':main()
