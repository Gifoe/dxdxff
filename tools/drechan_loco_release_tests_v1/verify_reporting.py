"""Release-only synthetic test of the frozen LOCO report; never clinical data."""
import ast
import importlib.util
from pathlib import Path
import numpy as np
import pandas as pd

REPO=Path(__file__).resolve().parents[2]
EXP=REPO/'experiments/drechan_loco_five_models_seed42_v1'
spec=importlib.util.spec_from_file_location('loco_release_metrics',REPO/'experiments/m8_raw_backbones_pr_seed42_v1/evaluate.py')
ev=importlib.util.module_from_spec(spec);spec.loader.exec_module(ev)
sources=['hup','lzu','multicenter','pediatric']
arms=['OURS_RAW_FEATURE','OURS_RAW','OURS_FEATURE','LINEAR_SVM_FEATURE_PR','TIMECONVCNN_RAW']
metrics=['macro_f1','auroc','auprc','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
scope=dict(np=np,pd=pd,EV=ev,SOURCES=sources,ARMS=arms,METRICS=metrics)
tree=ast.parse((EXP/'pipeline.py').read_text())
names=['patient_metrics','bootstrap_draws','boot','tables']
nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
assert len(nodes)==4
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(EXP/'pipeline.py'),'exec'),scope)

rng=np.random.default_rng(42);rows=[];patients=np.repeat(sources,[36,21,15,8])
for i,source in enumerate(patients):
    y=np.array([1,0,0,1]);p=rng.uniform(.01,.99,4)
    if source=='hup':p=.75*y+.1+rng.uniform(0,.1,4)
    for arm in arms:
        for j in range(4):
            rows.append(dict(arm=arm,held_source=source,patient=f'SYNTHETIC_{i:03d}',channel=f'C{j}',y=int(y[j]),probability=float(p[j]),prediction=bool(p[j]>=.5)))
frame=pd.DataFrame(rows);p=scope['patient_metrics'](frame);assert len(p)==400
tables=scope['tables'](frame,p)
assert len(tables['MAIN_RESULTS'])==5 and len(tables['SOURCE_RESULTS'])==20
assert len(tables['POOLED_RESULTS'])==25 and len(tables['INTERVALS'])==405 and len(tables['PRIMARY_FAMILY_INTERVALS'])==8
for row in tables['INTERVALS']:
    if row['comparison'].startswith('FULL_MINUS_'):
        assert row['estimate']==row['ci_low']==row['ci_high']==0.
for row in tables['PRIMARY_FAMILY_INTERVALS']:assert row['ci_low']==row['ci_high']==0.
for a in arms:
    g=p[p.arm==a];mean=next(r for r in tables['MAIN_RESULTS'] if r['arm']==a)
    equal=next(r for r in tables['SOURCE_EQUAL_RESULTS'] if r['arm']==a)
    sd=next(r for r in tables['SOURCE_SD'] if r['arm']==a)
    for m in metrics:
        group=[g[g.held_source==s][m].to_numpy().mean() for s in sources]
        assert abs(mean[m]-g[m].to_numpy().mean())<1e-14
        assert abs(equal[m]-np.mean(group))<1e-14 and abs(sd[m]-np.std(group,ddof=1))<1e-14
    assert abs(mean['auroc']-equal['auroc'])>.001

# Independent multinomial-count calculation of ALL absolute intervals.
first=p[p.arm==arms[0]].sort_values('patient');ids=first.patient.tolist();src=first.held_source.to_numpy()
rng=np.random.default_rng(42);counts=np.zeros((10000,80))
for s in sources:
    ix=np.flatnonzero(src==s);draw=ix[rng.integers(0,len(ix),size=(10000,len(ix)))]
    np.add.at(counts,(np.arange(10000)[:,None],draw),1.)
v=p[p.arm==arms[0]].set_index('patient').loc[ids,metrics].to_numpy()
ci=np.quantile(counts@v/80,[.025,.975],axis=0)
for j,m in enumerate(metrics):
    row=next(r for r in tables['INTERVALS'] if r['scope']=='ALL' and r['comparison']==arms[0] and r['metric']==m)
    assert np.max(np.abs([row['ci_low'],row['ci_high']]-ci[:,j]))<1e-12
print('SYNTHETIC_REPORT_PASS:400rows,3600metrics,405nominal,8family;independent_count_bootstrap;patient/source weights distinct;zero paired contrasts exact;NO_CLINICAL_DATA')
