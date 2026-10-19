"""Portable public-only independent aggregate/schema/report integrity checks."""
import csv
import hashlib
import json
import math
from pathlib import Path
import re

HERE=Path(__file__).resolve().parent;ROOT=HERE/'results'
METRICS=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
def table(name):
    with (ROOT/(name+'.csv')).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    arms=json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))['arms'];status=json.loads((ROOT/'OUTER_STATUS.json').read_text(encoding='utf-8'));v=json.loads((ROOT/'INDEPENDENT_OUTER_VERIFICATION.json').read_text(encoding='utf-8'))
    assert len(arms)==18 and status['status']=='COMPLETE_VERIFIED_F0_18_ARM_FIVE_FOLD_OUTER' and v['status']=='PASS'
    assert status['patient_rows']==v['patient_rows']==1440 and status['metric_checks']==v['patient_metric_checks']==12960
    assert status['new_fits']==80 and status['reused_cells']==10 and status['logical_model_cells']==90 and status['selected_arm'] is None and not status['independent_confirmation']
    assert v['nominal_interval_checks']==324 and v['macro_family_interval_checks']==18 and v['maximum_interval_difference']<1e-12
    rows=table('MAIN_RESULTS');assert len(rows)==18;points={r['arm']:r for r in rows};assert set(points)==set(arms)
    for r in rows:
        assert int(r['patients'])==80 and all(math.isfinite(float(r[m])) for m in METRICS)
        assert abs(float(r['macro_f1'])-.5*(float(r['f1'])+float(r['nez_f1'])))<1e-12
    assert abs(float(points['FULL_F0']['macro_f1'])-.6479765335907922)<1e-12
    assert abs(float(points['WO_EARLY_FILM']['macro_f1'])-.6396579900303192)<1e-12
    folds=table('FOLD_RESULTS');sd=table('FOLD_SD');assert len(folds)==90 and len(sd)==18
    for arm in arms:
        z=[r for r in folds if r['arm']==arm];assert {int(r['fold']) for r in z}==set(range(1,6)) and sum(int(r['patients']) for r in z)==80
        for m in METRICS:
            assert abs(math.fsum(float(r[m])*int(r['patients']) for r in z)/80-float(points[arm][m]))<1e-12
            mu=math.fsum(float(r[m]) for r in z)/5;expected=math.sqrt(math.fsum((float(r[m])-mu)**2 for r in z)/4)
            assert abs(float(next(r for r in sd if r['arm']==arm)[m])-expected)<1e-12
    ci=table('INTERVALS');family=table('MACRO_FAMILY_INTERVALS');assert len(ci)==324 and len(family)==18
    expected={a:{a:1} for a in arms}
    expected.update({'FULL_minus_'+a:{'FULL_F0':1,a:-1} for a in arms[1:]})
    expected['Early_by_late_guidance']={'FULL_F0':1,'WO_BOTH_GUIDANCE':1,'WO_EARLY_FILM':-1,'WO_LATE_GATE':-1}
    assert {r['contrast'] for r in ci}==set(expected)
    for r in ci:
        point=math.fsum(float(points[a][r['metric']])*w for a,w in expected[r['contrast']].items())
        assert abs(float(r['mean'])-point)<1e-12 and int(r['patients'])==80 and int(r['draws'])==10000
    for r in family:
        q=next(q for q in ci if q['contrast']==r['contrast'] and q['metric']=='macro_f1')
        assert abs(float(r['mean'])-float(q['mean']))<1e-12 and float(r['low'])<=float(q['low']) and float(r['high'])>=float(q['high']) and int(r['comparisons'])==18
    cells=table('TRAIN_CELLS');history=table('TRAIN_HISTORY');assert len(cells)==90
    assert {(r['arm'],int(r['fold'])) for r in cells}=={(a,f) for a in arms for f in range(1,6)}
    assert sum(r['reused']=='True' for r in cells)==10 and sum(int(r['epochs']) for r in cells)==len(history)
    for r in cells:
        assert len(r['checkpoint_sha256'])==64 and len(r['validation_sha256'])==64 and 1<=int(r['epoch'])<=30 and 0<=float(r['threshold'])<=1
        if r['reused']=='True':assert int(r['epochs'])==0 and r['arm'] in arms[:2]
        else:assert 6<=int(r['epochs'])<=30 and int(r['epoch'])<=int(r['epochs'])
    raw=table('RAW_COVERAGE');assert len(raw)==80
    for arm in arms[2:]:
        z=[r for r in raw if r['arm']==arm];assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635
        assert sum(int(r['valid_windows']) for r in z)==(0 if arm=='PHYSIOLOGY_ONLY' else 1471965)
        if arm=='EARLY_FILM_MISMATCH':assert all(int(r['eligible_valid_windows'])==int(r['changed_valid_windows']) and int(r['eligible_valid_windows'])+int(r['singleton_valid_windows'])==int(r['valid_windows']) for r in z)
    cov=table('METRIC_COVERAGE');assert len(cov)==162 and all(int(r['finite_patients'])==int(r['patients'])==80 for r in cov)
    centers=table('SOURCE_RESULTS');assert len(centers)==72
    for arm in arms:
        z=[r for r in centers if r['arm']==arm];assert {r['center']:int(r['patients']) for r in z}=={'hup':36,'lzu':21,'multicenter':15,'pediatric':8}
        for m in METRICS:assert abs(math.fsum(float(r[m])*int(r['patients']) for r in z)/80-float(points[arm][m]))<1e-12
    feature=table('FEATURE_CONTRIBUTION');assert len(feature)==7 and [r['arm'] for r in feature]==arms[11:]
    for r in feature:
        for m in METRICS:
            q=next(q for q in ci if q['contrast']=='FULL_minus_'+r['arm'] and q['metric']==m)
            assert all(abs(float(r[m+'_'+k])-float(q[k]))<1e-12 for k in ['mean','low','high'])
    for name,selected in [('CORE_ABLATION_TABLE',arms[:11]),('RAW_COMPLEMENTARITY',['FULL_F0','RAW_ONLY','PHYSIOLOGY_ONLY'])]:
        z=table(name);assert [r['arm'] for r in z]==selected
        assert all(r==points[r['arm']] for r in z)
    md=(ROOT/'REPORT.md').read_text(encoding='utf-8');lines=md.splitlines();start=next(i for i,s in enumerate(lines) if s.startswith('| Configuration |'))+2
    for a,line in zip(arms,lines[start:start+18]):
        values=[x.strip() for x in line.split('|')[2:-1]];assert len(values)==9
        assert all(abs(float(x)-float(points[a][m]))<=.50001e-6 for x,m in zip(values,METRICS))
    pairs=[s for s in lines if s.startswith('| Full minus ')];assert len(pairs)==17
    for a,line in zip(arms[1:],pairs):
        fields=[x.strip() for x in line.split('|')[1:-1]];q=next(q for q in ci if q['contrast']=='FULL_minus_'+a and q['metric']=='macro_f1');f=next(f for f in family if f['contrast']==q['contrast'])
        nums=[float(x) for x in re.findall(r'[+-]?\d+\.\d+',fields[1]+' '+fields[2]+' '+fields[3])]
        assert len(nums)==5 and all(abs(x-float(y))<=.50001e-6 for x,y in zip(nums,[q['mean'],q['low'],q['high'],f['low'],f['high']]))
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'));assert not manifest['private_artifacts_exported']
    assert all(sha(ROOT/n)==h for n,h in manifest['files'].items())
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    for path in ROOT.glob('*.csv'):
        with path.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden),path.name
    print('F0_18_ARM_PUBLIC_RESULTS_REPORT_AND_MANIFEST_PASS')

if __name__=='__main__':main()
