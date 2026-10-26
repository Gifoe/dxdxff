"""Predeclared aggregation and conditional paired patient inference."""
import csv
import numpy as np
from math_core import ARMS,COORDS,SOURCES,bootstrap_indices
import contract as io

CONTRASTS=[('GLOBAL-FULL','GLOBAL','FULL',0),('GLOBAL-CENTER','GLOBAL','CENTER',1),('GLOBAL-SCALE','GLOBAL','SCALE',1),('CENTER-FULL','CENTER','FULL',1),('SCALE-FULL','SCALE','FULL',1),('SOURCE-FULL','SOURCE','FULL',1)]

def arrays(rows):
    return dict(crps=np.array([r['crps'] for r in rows]),EZ=np.array([r['EZ'] for r in rows]),NEZ=np.array([r['NEZ'] for r in rows]),
        gap=np.array([r['EZ']-r['NEZ'] for r in rows]),unsupported=np.array([r['unsupported'] for r in rows]),
        unsupported_query_nonconstant=np.array([r['unsupported_query_nonconstant'] for r in rows]),permutation=np.array([r['permutation'] for r in rows]))

def interval(v,draw,family=False):
    samples=v[draw].mean(1);lo,hi=np.quantile(samples,[.025,.975])
    q=dict(mean=float(v.mean()),median=float(np.median(v)),ci_low=float(lo),ci_high=float(hi),patients=len(v),positive_fraction=float((v>0).mean()))
    if family:q.update(zip(['family_low','family_high'],map(float,np.quantile(samples,[.05/12,1-.05/12]))))
    return q

def summarize(rows):
    a=arrays(rows);src=np.array([r['source'] for r in rows]);draw=bootstrap_indices(src);result={k:[] for k in ['BACKGROUND','CONTRASTS','ANOMALY','SUPPORT','PERMUTATION','PERMUTATION_SUMMARY']}
    for scope in ['ALL']+SOURCES:
        ix=np.arange(len(rows)) if scope=='ALL' else np.flatnonzero(src==scope)
        local={int(j):i for i,j in enumerate(ix)}
        d=draw if scope=='ALL' else np.array([[local[int(j)] for j in q if src[j]==scope] for q in draw])
        for k,coordinate in [(-1,'ALL12')]+list(enumerate(COORDS)):
            scores=a['crps'][ix].mean(-1) if k==-1 else a['crps'][ix,:,k]
            gap=a['gap'][ix].mean(-1) if k==-1 else a['gap'][ix,k]
            for j,arm in enumerate(ARMS):result['BACKGROUND'].append(dict(scope=scope,coordinate=coordinate,arm=arm,**interval(scores[:,j],d)))
            for name,left,right,fam in CONTRASTS:
                is_family=bool(fam and scope=='ALL' and k==-1)
                result['CONTRASTS'].append(dict(scope=scope,coordinate=coordinate,contrast=name,primary=bool(not fam and scope=='ALL' and k==-1),**interval(scores[:,ARMS.index(left)]-scores[:,ARMS.index(right)],d,is_family)))
            for endpoint in ['EZ','NEZ','gap']:
                v=a[endpoint][ix].mean(-1) if k==-1 else a[endpoint][ix,k]
                result['ANOMALY'].append(dict(scope=scope,coordinate=coordinate,endpoint=endpoint,**interval(v,d,endpoint=='gap' and scope=='ALL' and k==-1)))
            for endpoint in ['unsupported','unsupported_query_nonconstant']:
                v=a[endpoint][ix].mean(-1) if k==-1 else a[endpoint][ix,k]
                result['SUPPORT'].append(dict(scope=scope,coordinate=coordinate,endpoint=endpoint,**interval(v,d)))
        null=a['permutation'][ix].mean(0);obs=float(a['gap'][ix].mean())
        result['PERMUTATION'].extend(dict(scope=scope,permutation=b,gap=float(v)) for b,v in enumerate(null))
        result['PERMUTATION_SUMMARY'].append(dict(scope=scope,observed=obs,null_mean=float(null.mean()),null_low=float(np.quantile(null,.025)),null_high=float(np.quantile(null,.975)),conditional_plus_one_upper=float((1+(null>=obs).sum())/1001),permutations=1000,patients=len(ix)))
    return result

def export(root,tables):
    for name,rows in tables.items():
        p=root/(name+'.csv');assert not p.exists();keys=list(dict.fromkeys(k for r in rows for k in r))
        with p.open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)

def report(root):
    assert io.read(root/'public/INDEPENDENT_VERIFICATION.json')['status']=='PASS'
    rows=io.load(root/'private/ENDPOINTS.pkl');tables=summarize(rows);export(root/'public',tables)
    io.write(root/'public/STATUS.json',dict(status='COMPLETE_VERIFIED_PR_REFERENCE_TRANSFER_80',patients=80,channels=7635,repetitions=100,permutations=1000,secondary_family=6,row_counts={k:len(v) for k,v in tables.items()},source_commit=io.commit()))
