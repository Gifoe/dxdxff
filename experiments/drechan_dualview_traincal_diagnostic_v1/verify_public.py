"""Public hashes and aggregate structure only; no private diagnostic recomputation."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import sys


def main(root):
    own=Path(__file__).resolve().parent
    def read(name):return json.loads((root/name).read_text(encoding='utf-8'))
    def table(name):
        with (root/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    def keyed(rows,fields):
        d={tuple(r[k] for k in fields):r for r in rows};assert len(d)==len(rows);return d
    tree=ast.parse((own/'verify.py').read_text(encoding='utf-8'))
    allowed=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOW' for t in n.targets))
    manifest=read('PACKAGE_MANIFEST.json');assert len(allowed)==len(set(allowed))==10 and set(manifest['files'])==set(allowed)
    assert manifest['status']=='EXPLICIT_ANONYMOUS_ALLOWLIST' and not manifest['private_artifacts_exported']
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','prediction','probability','y','path'}
    for n,h in manifest['files'].items():
        assert Path(n).name==n and '..' not in Path(n).parts and (root/n).stat().st_size<5000000
        assert hashlib.sha256((root/n).read_bytes()).hexdigest()==h
        if n.endswith('.csv'):
            with (root/n).open(encoding='utf-8',newline='') as f:assert not set(next(csv.reader(f)))&forbidden
    push=read('SOURCE_PUSH_BEFORE_AUDIT.json');assert push['source_pushed_before_audit'] and len(push['source'])==8
    assert len(push['commit'])==40 and all(c in '0123456789abcdef' for c in push['commit'])
    assert all(hashlib.sha256((own/n).read_bytes()).hexdigest()==h for n,h in push['source'].items())
    assert read('TESTS.json')==dict(status='PASS',changed_critical_path_tests=5,no_performance_evidence=True)
    expected={(a,str(s),str(f)) for a in ['B0','PRUNED_REF','FULL'] for s in [42,43,44] for f in range(1,6)}
    cal=keyed(table('CAL_STABILITY'),['arm','seed','fold']);replay=keyed(table('CAL_SELECTED_REPLAY'),['arm','seed','fold'])
    assert set(cal)==set(replay)==expected
    roles=keyed(table('ROLE_SCOPE'),['fold']);assert set(roles)=={(str(f),) for f in range(1,6)}
    for r in roles.values():
        assert int(r['CAL_patients'])==8 and 34<=int(r['TRAIN_patients'])<=36
        assert r['ASSESS_labels_indexed']=='False' and r['Outer_scored']=='False'
    for k,r in cal.items():
        assert int(r['cal_patients'])==8 and int(r['cal_channels'])==int(roles[(r['fold'],)]['CAL_channels'])
        assert r['checkpoint_selected_on_all_CAL']=='True' and r['not_independent_validation']=='True'
        assert abs((float(r['leave_one_CAL_threshold_macro_f1'])-float(r['cal_macro_f1']))-float(r['difference']))<1e-12
        assert 0<=float(r['cal_macro_f1'])<=1 and 0<=float(r['leave_one_CAL_threshold_macro_f1'])<=1
        assert 0<=float(r['loo_threshold_min'])<=float(r['loo_threshold_q25'])<=float(r['loo_threshold_median'])<=float(r['loo_threshold_q75'])<=float(r['loo_threshold_max'])<=1
        assert replay[k]['stored_CAL_hash_verified']=='True' and replay[k]['no_model_or_threshold_change']=='True'
        assert (replay[k]['new_CAL_forward_bit_exact']=='True')==(r['arm']!='B0')
        assert (replay[k]['reused_stored_CAL_only']=='True')==(r['arm']=='B0')
    raw=keyed(table('RAW_ANCHOR_DRIFT'),['arm','seed','fold','phase'])
    scores=keyed(table('LABEL_FREE_SCORE_SUMMARY'),['arm','seed','fold','phase'])
    assert set(raw)=={(a,str(s),str(f),p) for a in ['PRUNED_REF','FULL'] for s in [42,43,44] for f in range(1,6) for p in ['TRAIN','CAL']}
    assert set(scores)=={(a,str(s),str(f),p) for a in ['B0','PRUNED_REF','FULL'] for s in [42,43,44] for f in range(1,6) for p in (['CAL'] if a=='B0' else ['TRAIN','CAL'])}
    for k,r in raw.items():
        role=roles[(r['fold'],)];phase=r['phase'];assert int(r['channels'])==int(role[phase+'_channels'])
        assert int(r['patients'])==int(role[phase+'_patients']) and int(r['valid_windows'])>0
        assert 0<=float(r['abs_q50'])<=float(r['abs_q95'])<=float(r['abs_q99'])<=float(r['abs_max'])
        assert 0<=float(r['fraction_abs_above10'])<=float(r['fraction_abs_above5'])<=1
        assert 0<float(r['anchor_scale_min'])<=float(r['anchor_scale_median'])<=float(r['anchor_scale_max'])
    for k,r in scores.items():
        role=roles[(r['fold'],)];phase=r['phase'];assert int(r['channels'])==int(role[phase+'_channels']) and int(r['patients'])==int(role[phase+'_patients'])
        assert all(0<=float(r[n])<=1 for n in ['probability_saturation_fraction','fraction_abs_logit_above20','unique_logit_fraction','unique_probability_fraction'])
        assert all(math.isfinite(float(r[n])) for n in ['logit_mean','logit_population_sd','patient_logit_median_sd'])
    audit=read('INDEPENDENT_VERIFICATION.json');assert audit['status']=='PASS'
    assert audit['selected_CAL_cells']==45 and audit['leave_one_CAL_checks']==360 and audit['latent_phase_cells']==60 and audit['score_phase_cells']==75
    assert audit['maximum_CAL_metric_discrepancy']<1e-12 and audit['all_CAL_forward_or_prior_reuse_bindings_verified'] and audit['TRAIN_CAL_only']
    assert audit['neural_fits']==audit['assessment_metrics']==0 and not audit['new_Outer'] and not audit['active_goal_achieved']
    status=read('RUN_STATUS.json');assert status['status']=='ALL45_DIAGNOSTIC_CELLS_REPORTED'
    assert status['neural_fits']==status['new_deployment_thresholds']==status['assessment_metrics']==0 and not status['new_Outer'] and not status['active_goal_achieved']
    print(json.dumps(dict(status='PASS',public_package_members=11,source_files=8,CAL_cells=45,latent_phase_cells=60,score_phase_cells=75,
                          private_data_used=False,private_360_case_recomputation=False,private_verifier_required=True)))


if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'results')
