"""Standalone public-only release checks, independent of private metric engines."""
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
ARMS=['B0','JOINT_PRUNED','FULL','WO_FEATURE_PR','WO_RAW_PR','WO_BOTH_PR','WO_PRE_REFERENCE']
CONTRASTS={
 'FULL_MINUS_B0':{'FULL':1,'B0':-1},'PRUNING_EFFECT':{'JOINT_PRUNED':1,'B0':-1},
 'FUSION_EFFECT':{'FULL':1,'JOINT_PRUNED':-1},'FEATURE_PR_WITH_RAW':{'FULL':1,'WO_FEATURE_PR':-1},
 'RAW_PR_WITH_FEATURE':{'FULL':1,'WO_RAW_PR':-1},'BOTH_PR_JOINT':{'FULL':1,'WO_BOTH_PR':-1},
 'PRE_REFERENCE':{'FULL':1,'WO_PRE_REFERENCE':-1},
 'PR_INTERACTION':{'FULL':1,'WO_FEATURE_PR':-1,'WO_RAW_PR':-1,'WO_BOTH_PR':1}}
PRIMARY=['FULL_MINUS_B0','FUSION_EFFECT','FEATURE_PR_WITH_RAW','RAW_PR_WITH_FEATURE','PRE_REFERENCE']

def read_json(path):return json.loads(path.read_text(encoding='utf-8'))

def main(root):
    own=Path(__file__).resolve().parent
    def table(name):
        with (root/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    manifest=read_json(root/'PACKAGE_MANIFEST.json');assert len(manifest['files'])==30
    tree=ast.parse((own/'package_public.py').read_text(encoding='utf-8'))
    allowed=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ALLOW' for t in node.targets))
    assert len(allowed)==len(set(allowed))==30 and set(manifest['files'])==set(allowed)
    assert manifest['status']=='EXPLICIT_ANONYMOUS_ALLOWLIST' and not manifest['private_artifacts_exported']
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','score_nez','prediction','probability','y','path'}
    for n,h in manifest['files'].items():
        assert not Path(n).is_absolute() and '..' not in Path(n).parts
        p=root/n;assert Path(n).name==n and p.stat().st_size<5000000 and hashlib.sha256(p.read_bytes()).hexdigest()==h
        if p.suffix=='.csv':
            with p.open(newline='',encoding='utf-8') as f:assert not set(next(csv.reader(f)))&forbidden
    audit=json.loads((root/'INDEPENDENT_VERIFICATION.json').read_text())
    assert audit['status']=='PASS' and audit['patient_fold_seed_rows']==840 and audit['patient_metric_checks']==7560
    assert audit['unique_patients']==33 and audit['nominal_interval_checks']==135 and audit['primary_family_interval_checks']==10
    assert audit['maximum_interval_discrepancy']<1e-12 and audit['no_new_Outer'] and not audit['active_goal_achieved']
    status=json.loads((root/'TRAIN_STATUS.json').read_text());assert status['fresh_fits']==90 and status['reused']==15
    assert status['status']=='ALL105_COMPLETE'
    expected={(a,seed,fold) for a in ARMS for seed in [42,43,44] for fold in range(1,6)}
    cells=table('TRAIN_CELLS');assert len(cells)==105
    assert {(r['arm'],int(r['seed']),int(r['fold'])) for r in cells}==expected
    for r in cells:
        assert (r['reused']=='True')==(r['arm']=='B0')
        assert int(r['parameters'])==(11425 if r['arm']=='B0' else 9329)
        assert 1<=int(r['epoch'])<=int(r['epochs'])<=30 and int(r['epochs'])>=6
    history=table('TRAIN_HISTORY')
    assert sum(r['reused']=='False' for r in history)==status['new_epochs']
    assert sum(r['reused']=='True' for r in history)==status['reused_epochs']
    for r in cells:
        h=[q for q in history if (q['arm'],q['seed'],q['fold'])==(r['arm'],r['seed'],r['fold'])]
        assert len(h)==int(r['epochs']) and sorted(int(q['epoch']) for q in h)==list(range(1,len(h)+1))
    coverage=table('COVERAGE');assert len(coverage)==105
    assert {(r['arm'],int(r['seed']),int(r['fold'])) for r in coverage}==expected
    assert all(int(r['patients'])==8 and int(r['valid_windows'])>0 for r in coverage)
    replay=table('CALIBRATION_REPLAY');assert len(replay)==105
    assert all(r['CAL_logits_bit_exact']=='True' for r in replay)
    assert read_json(root/'FREEZE_STATUS.json')['status']=='ALL105_FROZEN'
    assert read_json(root/'PREDICTION_STATUS.json')['status']=='ALL105_SEALED_BEFORE_METRICS'
    assert read_json(root/'SELECTION_VERIFICATION.json')['initial_TRAIN_anchor_replays']==15
    assert read_json(root/'FREEZE_STATUS.json')['freeze_sha256']==read_json(root/'PREDICTION_STATUS.json')['freeze_sha256']
    mainrows={r['arm']:r for r in table('MAIN_RESULTS')};seeds={(r['arm'],int(r['seed'])):r for r in table('SEED_RESULTS')}
    assert set(mainrows)==set(ARMS) and len(seeds)==21
    for a in ARMS:
        assert int(mainrows[a]['unique_patients'])==33 and int(mainrows[a]['patient_fold_seed_rows'])==120
        sd=next(r for r in table('SEED_SD') if r['arm']==a)
        for m in METRICS:
            values=[float(seeds[(a,s)][m]) for s in [42,43,44]]
            assert abs(statistics.mean(values)-float(mainrows[a][m]))<1e-12
            assert abs(statistics.stdev(values)-float(sd[m]))<1e-12
        for s in [42,43,44]:
            ff=[r for r in table('FOLD_RESULTS') if r['arm']==a and int(r['seed'])==s];assert len(ff)==5
            ss=next(r for r in table('FOLD_SD') if r['arm']==a and int(r['seed'])==s)
            for m in METRICS:assert abs(statistics.stdev(float(r[m]) for r in ff)-float(ss[m]))<1e-12
        sources=[r for r in table('SOURCE_RESULTS') if r['arm']==a];assert sum(int(r['unique_patients']) for r in sources)==33
        for r in sources:
            suppressed=int(r['unique_patients'])<5;assert (r['suppressed']=='True')==suppressed
            assert all(math.isnan(float(r[m])) if suppressed else math.isfinite(float(r[m])) for m in METRICS)
    ci=table('INTERVALS');family=table('PRIMARY_FAMILY_INTERVALS');assert len(ci)==135 and len(family)==10
    assert {(r['kind'],r['contrast'],r['metric']) for r in ci}==({('absolute',a,m) for a in ARMS for m in METRICS}|{('paired',c,m) for c in CONTRASTS for m in METRICS})
    for r in ci:
        weights={r['contrast']:1} if r['kind']=='absolute' else CONTRASTS[r['contrast']]
        value=sum(float(mainrows[a][r['metric']])*w for a,w in weights.items())
        assert abs(value-float(r['mean']))<1e-12 and math.isfinite(float(r['low'])) and math.isfinite(float(r['high'])) and float(r['low'])<=float(r['high'])
        assert int(r['unique_patients'])==33 and int(r['seeds'])==3 and int(r['draws'])==10000
    for r in family:
        value=sum(float(mainrows[a][r['metric']])*w for a,w in CONTRASTS[r['contrast']].items())
        assert abs(value-float(r['mean']))<1e-12 and math.isfinite(float(r['low'])) and math.isfinite(float(r['high'])) and float(r['low'])<=float(r['high'])
        assert int(r['comparisons'])==10 and int(r['draws'])==10000
    assert len(table('COMPONENT_EVIDENCE'))==5 and {r['contrast'] for r in table('COMPONENT_EVIDENCE')}==set(PRIMARY)
    assert {(r['contrast'],r['metric']) for r in family}=={(n,m) for n in PRIMARY for m in ['auroc','macro_f1']}
    for r in table('COMPONENT_EVIDENCE'):
        weights=CONTRASTS[r['contrast']]
        point=all(sum(float(mainrows[a][m])*w for a,w in weights.items())>0 for m in ['auroc','macro_f1'])
        count=sum(all(sum(float(seeds[(a,s)][m])*w for a,w in weights.items())>0 for m in ['auroc','macro_f1']) for s in [42,43,44])
        assert (r['mean_AUROC_and_Macro_both_positive']=='True')==point and int(r['seeds_both_positive'])==count
        assert (r['stable_positive_point_evidence']=='True')==(point and count>=2)
        nominal=all(float(next(q for q in ci if q['contrast']==r['contrast'] and q['metric']==m)['low'])>0 for m in ['auroc','macro_f1'])
        simultaneous=all(float(next(q for q in family if q['contrast']==r['contrast'] and q['metric']==m)['low'])>0 for m in ['auroc','macro_f1'])
        assert (r['both_nominal_lowers_positive']=='True')==nominal and (r['both_family_lowers_positive']=='True')==simultaneous
    gate=json.loads((root/'QUALIFICATION_GATE.json').read_text());primary=['auroc','auprc','macro_f1','f1']
    delta={m:float(mainrows['FULL'][m])-float(mainrows['B0'][m]) for m in primary}
    count=0
    for s in [42,43,44]:
        d={m:float(seeds[('FULL',s)][m])-float(seeds[('B0',s)][m]) for m in primary}
        count+=d['auroc']>0 and all(d[m]>=-.01 for m in primary[1:])
    performance=delta['auroc']>0 and all(delta[m]>=-.01 for m in primary[1:]) and count>=2
    points={r['contrast']:r['stable_positive_point_evidence']=='True' for r in table('COMPONENT_EVIDENCE')}
    pr=points['FEATURE_PR_WITH_RAW'] and points['RAW_PR_WITH_FEATURE']
    fusion=points['FUSION_EFFECT'];pre=points['PRE_REFERENCE']
    assert gate['performance_margin_pass']==performance and gate['both_separate_PR_point_rules_pass']==pr
    assert gate['fusion_point_rule_pass']==fusion and gate['pre_reference_point_rule_pass']==pre
    assert gate['qualified_for_final_study_planning']==bool(performance and pr and fusion and pre)
    assert gate['status']==('PASS' if performance and pr and fusion and pre else 'FAIL')
    assert not gate['new_Outer_authorized'] and not gate['active_goal_achieved']
    own=Path(__file__).resolve().parent
    admitted=json.loads((root/'ADMISSION.json').read_text())
    assert admitted['status']=='PASS' and admitted['reuse_cells']==15 and len(admitted['source'])==12
    assert all(hashlib.sha256((own/n).read_bytes()).hexdigest()==h for n,h in admitted['source'].items())
    completed=read_json(root/'COMPLETION_STATUS.json')
    assert completed['status']=='PRUNED_FUSION_INTERNAL_COMPLETE_VERIFIED' and completed['fresh_fits']==90 and completed['exact_reused_fits']==15
    assert not completed['new_Outer'] and not completed['active_goal_achieved']
    print(json.dumps(dict(status='PASS',public_package_members=31,source_files=12,arms=7,seeds=3,nominal_intervals=135,additional_intervals=10,
                          private_data_used=False,full_patient_bootstrap_recomputed=False,private_independent_verifier_required=True)))

if __name__=='__main__':main(Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent/'results')
