"""Private-data-free release checks; not a substitute for server patient audit."""
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
ARMS=['B0','B1','B2','B3'];SEEDS=[42,43,44];PRIMARY=['auroc','auprc','macro_f1','f1']
def main(root):
    def jr(n):return json.loads((root/n).read_text(encoding='utf-8'))
    def csvread(n):
        with (root/n).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
    def equal(a,b):assert math.isfinite(float(a)) and abs(float(a)-float(b))<1e-12,(a,b)
    manifest=jr('PACKAGE_MANIFEST.json');assert not manifest['private_artifacts_exported']
    for n,h in manifest['files'].items():
        p=Path(n);assert not p.is_absolute() and '..' not in p.parts and p.suffix in ['.json','.csv','.md'] and 'PRIVATE' not in n
        assert hashlib.sha256((root/p).read_bytes()).hexdigest()==h,n
    status=jr('COMPLETION_STATUS.json');stages=status['stages'];assert stages in [['internal'],['internal','final']]
    points={};seedpoints={};stages_checked=[]
    for stage in stages:
        prefix=stage+'/'
        verification=jr(prefix+'INDEPENDENT_VERIFICATION.json');assert verification['status']=='PASS'
        assert verification['patient_fold_seed_rows']==(480 if stage=='internal' else 960)
        assert verification['patient_metric_checks']==verification['patient_fold_seed_rows']*9
        assert verification['nominal_interval_checks']==63 and verification['primary_family_interval_checks']==12
        cells=csvread(prefix+'TRAIN_CELLS.csv');assert len(cells)==60
        expected={(a,s,f) for a in ARMS for s in SEEDS for f in range(1,6)}
        assert {(r['arm'],int(r['seed']),int(r['fold'])) for r in cells}==expected
        assert sum(r['reused']=='True' for r in cells)==(0 if stage=='internal' else 5)
        for r in cells:
            assert int(r['parameters'])==(11425 if r['arm']=='B0' else 13201)
            assert 1<=int(r['epoch'])<=30 and 0<=float(r['threshold'])<=1
            assert int(r['epochs'])==0 if r['reused']=='True' else 6<=int(r['epochs'])<=30
        history=csvread(prefix+'TRAIN_HISTORY.csv');assert len(history)==sum(int(r['epochs']) for r in cells)==jr(prefix+'TRAIN_STATUS.json')['epochs']
        roles=csvread(prefix+'ROLE_AUDIT.csv');assert len(roles)==5
        for r in roles:
            f=int(r['fold']);n=[51,51,50,52,51][f-1]
            assert int(r['train_patients'])==(n-16 if stage=='internal' else n)
            assert int(r['cal_patients'])==(8 if stage=='internal' else 13)
            assert int(r['assessment_patients'])==(8 if stage=='internal' else 0)
        main=csvread(prefix+'MAIN_RESULTS.csv');seeds=csvread(prefix+'SEED_RESULTS.csv');sd=csvread(prefix+'SEED_SD.csv')
        assert len(main)==len(sd)==4 and len(seeds)==12
        pp={r['arm']:r for r in main};ss={(r['arm'],int(r['seed'])):r for r in seeds};points[stage]=pp;seedpoints[stage]=ss
        for arm in ARMS:
            r=pp[arm];assert int(r['seeds'])==3 and int(r['unique_patients'])==verification['unique_patients']
            assert int(r['patient_fold_seed_rows'])==(120 if stage=='internal' else 240)
            equal(r['macro_f1'],(float(r['f1'])+float(r['nez_f1']))/2)
            for m in METRICS:
                equal(r[m],statistics.mean(float(ss[(arm,s)][m]) for s in SEEDS))
                equal(next(z for z in sd if z['arm']==arm)[m],statistics.stdev(float(ss[(arm,s)][m]) for s in SEEDS))
                assert -1<=float(r[m])<=1 if m=='ez_fraction_bias' else 0<=float(r[m])<=1
        ci=csvread(prefix+'INTERVALS.csv');family=csvread(prefix+'PRIMARY_FAMILY_INTERVALS.csv');assert len(ci)==63 and len(family)==12
        for r in ci:
            expected=float(pp[r['contrast']][r['metric']]) if r['kind']=='absolute' else float(pp['B1'][r['metric']])-float(pp[r['contrast'].removeprefix('B1_minus_')][r['metric']])
            equal(r['mean'],expected);assert float(r['low'])<=float(r['high']) and int(r['draws'])==10000
        for r in family:
            equal(r['mean'],float(pp['B1'][r['metric']])-float(pp[r['contrast'].removeprefix('B1_minus_')][r['metric']]))
            assert r['metric'] in PRIMARY and int(r['comparisons'])==12 and int(r['draws'])==10000 and float(r['low'])<=float(r['high'])
        assert jr(prefix+'FREEZE_STATUS.json')['status']=='ALL60_FROZEN_BEFORE_SCORES' and jr(prefix+'PREDICTION_STATUS.json')['status']=='ALL60_CELLS_SEALED'
        stages_checked.append(dict(stage=stage,cells=60,epochs=len(history),patient_endpoint_checks=verification['patient_metric_checks'],nominal_intervals=63,family_intervals=12))
    p=points['internal'];s=seedpoints['internal'];g=jr('internal/GATE.json')
    d0={m:float(p['B1'][m])-float(p['B0'][m]) for m in METRICS};d2={m:float(p['B1'][m])-float(p['B2'][m]) for m in METRICS}
    ns0=sum(all(float(s[('B1',seed)][m])-float(s[('B0',seed)][m])>=-.01 for m in PRIMARY) for seed in SEEDS)
    ns2=sum(all(float(s[('B1',seed)][m])-float(s[('B2',seed)][m])>0 for m in ['auroc','auprc']) for seed in SEEDS)
    margin=all(d0[m]>=-.01 for m in PRIMARY) and ns0>=2
    context=all(d2[m]>0 for m in ['auroc','auprc']) and all(d2[m]>=-.01 for m in ['macro_f1','f1']) and ns2>=2
    assert margin==g['performance_margin_pass'] and context==g['global_reference_comparison_pass'] and (margin and context)==g['continue_final']==status['new_outer_completed']
    assert ns0==g['seeds_meeting_all_four_margins'] and ns2==g['seeds_B1_exceeding_B2_both_ranking_metrics']
    for m in METRICS:equal(d0[m],g['B1_minus_B0'][m]);equal(d2[m],g['B1_minus_B2'][m])
    print(json.dumps(dict(status='PUBLIC_AGGREGATES_PASS',manifest_files=len(manifest['files']),stages=stages_checked,
        fixed_gates_independently_recomputed=True,does_not_replace_private_patient_verification=True),indent=2),flush=True)

if __name__=='__main__':main(Path(sys.argv[1]))
