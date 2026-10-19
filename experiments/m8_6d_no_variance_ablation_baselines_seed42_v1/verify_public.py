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
    assert len(arms)==30 and status['status']=='COMPLETE_VERIFIED_F0_6D_30_GROUP_FIVE_FOLD_OUTER' and v['status']=='PASS'
    assert status['patient_rows']==v['patient_rows']==2400 and status['metric_checks']==v['patient_metric_checks']==21600
    assert status['new_neural_fits']==70 and status['new_tabular_fits']==70 and status['reused_cells']==10 and status['logical_model_cells']==150 and status['selected_arm'] is None and not status['independent_confirmation']
    assert v['nominal_interval_checks']==603 and v['macro_family_interval_checks']==37 and v['maximum_interval_difference']<1e-12
    rows=table('MAIN_RESULTS');assert len(rows)==30;points={r['arm']:r for r in rows};assert set(points)==set(arms)
    for r in rows:
        assert int(r['patients'])==80 and all(math.isfinite(float(r[m])) for m in METRICS)
        assert abs(float(r['macro_f1'])-.5*(float(r['f1'])+float(r['nez_f1'])))<1e-12
    assert abs(float(points['FULL_F0']['macro_f1'])-.6502248746169609)<1e-12
    assert abs(float(points['RAW_ONLY']['macro_f1'])-.6212881144113119)<1e-12
    folds=table('FOLD_RESULTS');sd=table('FOLD_SD');assert len(folds)==150 and len(sd)==30
    for arm in arms:
        z=[r for r in folds if r['arm']==arm];assert {int(r['fold']) for r in z}==set(range(1,6)) and sum(int(r['patients']) for r in z)==80
        for m in METRICS:
            assert abs(math.fsum(float(r[m])*int(r['patients']) for r in z)/80-float(points[arm][m]))<1e-12
            mu=math.fsum(float(r[m]) for r in z)/5;expected=math.sqrt(math.fsum((float(r[m])-mu)**2 for r in z)/4)
            assert abs(float(next(r for r in sd if r['arm']==arm)[m])-expected)<1e-12
    ci=table('INTERVALS');family=table('MACRO_FAMILY_INTERVALS');assert len(ci)==603 and len(family)==37
    expected={a:{a:1} for a in arms}
    expected.update({'FULL_minus_'+a:{'FULL_F0':1,a:-1} for a in arms[1:]})
    expected.update({c+'_PR_minus_NO_PR':{c+'_PR':1,c+'_NO_PR':-1} for c in json.loads((HERE/'PROTOCOL_LOCK.json').read_text(encoding='utf-8'))['classifiers']})
    expected['Early_by_late_guidance']={'FULL_F0':1,'WO_BOTH_GUIDANCE':1,'WO_EARLY_FILM':-1,'WO_LATE_GATE':-1}
    assert {r['contrast'] for r in ci}==set(expected)
    for r in ci:
        point=math.fsum(float(points[a][r['metric']])*w for a,w in expected[r['contrast']].items())
        assert abs(float(r['mean'])-point)<1e-12 and int(r['patients'])==80 and int(r['draws'])==10000
    for r in family:
        q=next(q for q in ci if q['contrast']==r['contrast'] and q['metric']=='macro_f1')
        assert abs(float(r['mean'])-float(q['mean']))<1e-12 and float(r['low'])<=float(q['low']) and float(r['high'])>=float(q['high']) and int(r['comparisons'])==37
    cells=table('TRAIN_CELLS');history=table('TRAIN_HISTORY');assert len(cells)==80
    assert {(r['arm'],int(r['fold'])) for r in cells}=={(a,f) for a in arms[:16] for f in range(1,6)}
    assert sum(r['reused']=='True' for r in cells)==10 and sum(int(r['epochs']) for r in cells)==len(history)
    for r in cells:
        assert len(r['checkpoint_sha256'])==64 and len(r['validation_sha256'])==64 and 1<=int(r['epoch'])<=30 and 0<=float(r['threshold'])<=1
        if r['reused']=='True':assert int(r['epochs'])==0 and r['arm'] in ['FULL_F0','RAW_ONLY']
        else:assert 6<=int(r['epochs'])<=30 and int(r['epoch'])<=int(r['epochs'])
    raw=table('RAW_COVERAGE');assert len(raw)==70
    for arm in [a for a in arms[:16] if a not in ['FULL_F0','RAW_ONLY']]:
        z=[r for r in raw if r['arm']==arm];assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635
        assert sum(int(r['valid_windows']) for r in z)==(0 if arm=='PHYSIOLOGY_ONLY' else 1471965)
        if arm=='EARLY_FILM_MISMATCH':assert all(int(r['eligible_valid_windows'])==int(r['changed_valid_windows']) and int(r['eligible_valid_windows'])+int(r['singleton_valid_windows'])==int(r['valid_windows']) for r in z)
    cov=table('METRIC_COVERAGE');assert len(cov)==270 and all(int(r['finite_patients'])==int(r['patients'])==80 for r in cov)
    centers=table('SOURCE_RESULTS');assert len(centers)==120
    for arm in arms:
        z=[r for r in centers if r['arm']==arm];assert {r['center']:int(r['patients']) for r in z}=={'hup':36,'lzu':21,'multicenter':15,'pediatric':8}
        for m in METRICS:assert abs(math.fsum(float(r[m])*int(r['patients']) for r in z)/80-float(points[arm][m]))<1e-12
    for name,selected in [('CORE_ABLATION_TABLE',arms[:10]),('FEATURE_ABLATION_TABLE',arms[10:16]),('CLASSIFIER_TABLE',arms[16:])]:
        z=table(name);assert [r['arm'] for r in z]==selected and all(r==points[r['arm']] for r in z)
    pri=table('PR_PAIRED_INTERVALS');assert len(pri)==63 and all(r['kind']=='PR_pair' for r in pri)
    tab=table('TABULAR_TRAIN_CELLS');assert len(tab)==70 and all(int(r['active_dimension'])==12 for r in tab)
    cfg=json.loads((HERE/'BASELINE_CONFIGS.json').read_text(encoding='utf-8'))['configs']
    for c,folds in cfg.items():
        for f,config in folds.items():
            z=[r for r in tab if r['arm'] in [c+'_PR',c+'_NO_PR'] and r['fold']==f]
            assert len(z)==2 and all(json.loads(r['config_json'])==config for r in z)
    tcov=table('TABULAR_COVERAGE');assert len(tcov)==70
    for arm in arms[16:]:
        z=[r for r in tcov if r['arm']==arm]
        assert len(z)==5 and sum(int(r['patients']) for r in z)==80 and sum(int(r['channels']) for r in z)==7635
        assert all(r['raw_consumed']=='False' and int(r['active_dimension'])==12 for r in z)
    md=(ROOT/'REPORT.md').read_text(encoding='utf-8');lines=md.splitlines()
    headers=[i for i,s in enumerate(lines) if s.startswith('| Configuration |')]
    assert len(headers)==2
    for selected,start in [(arms[:16],headers[0]+2),(arms[16:],headers[1]+2)]:
        for arm,line in zip(selected,lines[start:start+len(selected)]):
            values=[x.strip() for x in line.split('|')[2:-1]];assert len(values)==9
            assert all(abs(float(x)-float(points[arm][m]))<=.50001e-6 for x,m in zip(values,METRICS))
    start=next(i for i,s in enumerate(lines) if s.startswith('| Contrast |'))+2
    pairs=lines[start:start+37];assert len(pairs)==37
    for line in pairs:
        fields=[x.strip() for x in line.split('|')[1:-1]];contrast=fields[0]
        q=next(r for r in ci if r['contrast']==contrast and r['metric']=='macro_f1');f=next(r for r in family if r['contrast']==contrast)
        nums=[float(x) for x in re.findall(r'[+-]?\d+\.\d+',' '.join(fields[1:]))]
        assert len(nums)==5 and all(abs(x-float(y))<=.50001e-6 for x,y in zip(nums,[q['mean'],q['low'],q['high'],f['low'],f['high']]))
    manifest=json.loads((ROOT/'PACKAGE_MANIFEST.json').read_text(encoding='utf-8'));assert not manifest['private_artifacts_exported']
    assert all(sha(ROOT/n)==h for n,h in manifest['files'].items())
    source=json.loads((ROOT/'SOURCE_PUSH_BEFORE_TRAIN.json').read_text(encoding='utf-8'))
    assert source['source_pushed_before_formal_fits'] and source['commit']=='9c84578a83cbae402f28e308de05602931111019'
    assert all(sha(HERE/n)==h for n,h in source['source'].items())
    render=json.loads((ROOT/'RENDER_MANIFEST.json').read_text(encoding='utf-8'))
    assert sha(HERE/'render_report.py')==render['renderer_sha256']
    assert all(sha(ROOT/n)==h for n,h in render['files'].items())
    for name in ['FEATURE_CONTRIBUTION','CORE_PAIRED_INTERVALS','CLASSIFIER_PR_INTERVALS']:
        assert '<text' in (ROOT/'figures'/(name+'.svg')).read_text(encoding='utf-8')
        assert (ROOT/'figures'/(name+'.pdf')).read_bytes().startswith(b'%PDF-')
    if (ROOT/'NATIVE_RESUME_AUDIT.json').exists():
        q=json.loads((ROOT/'NATIVE_RESUME_AUDIT.json').read_text(encoding='utf-8-sig'))
        assert q['backup_hash_verified'] and q['all_completed_hashes_verified'] and not q['scientific_source_changed'] and not q['model_data_loss_selection_hyperparameters_changed']
    forbidden={'patient','patient_id','channel','channel_id','archive_indices','logit','score_nez','y','prediction'}
    for path in ROOT.glob('*.csv'):
        with path.open(encoding='utf-8',newline='') as f:assert not(set(next(csv.reader(f)))&forbidden),path.name
    print('SIXD_30_GROUP_PUBLIC_RESULTS_REPORT_AND_MANIFEST_PASS')

if __name__=='__main__':main()
