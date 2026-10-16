"""Separate implementation of every paired/absolute interval, no model/inference."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from report import ROOT,PUB,read,jr,sha,write
import numpy as np
import pandas as pd

METRICS=['auroc','auprc','macro_f1','f1','nez_f1','accuracy','mrr','hit_at_1','ez_fraction_bias']
CONFIGS=[f'{i:03b}' for i in range(8)]

def signs(factors,condition=None):
    axes=['VHE'.index(x) for x in factors];fixed={} if condition is None else {'VHE'.index(condition[0]):int(condition[1])}
    return {c:float(np.prod([2*int(c[i])-1 for i in axes]))/2**(3-len(axes)-len(fixed)) for c in CONFIGS if all(int(c[i])==v for i,v in fixed.items())}

def coefficients(row):
    kind=row['kind'];head=row.get('head');c=row.get('config')
    if kind=='deletion':return {(head,c):1,(head,'000'):-1}
    if kind=='raw_increment':return {('M8',c):1,('D1',c):-1}
    if kind=='raw_compensation':return {('M8',c):1,('M8','000'):-1,('D1',c):-1,('D1','000'):1}
    if kind=='factorial':return {(head,c):w for c,w in signs(row['factors']).items()}
    if kind=='conditional_interaction':return {(head,c):w for c,w in signs(row['factors'],row['condition']).items()}
    assert kind=='raw_factorial_compensation'
    return {(h,c):w*(1 if h=='M8' else -1) for c,w in signs(row['factors']).items() for h in ['D1','M8']}

def replay(matrix,weights,ids,unique,counts):
    x=np.stack([matrix[k] for k in weights]);finite=np.isfinite(x).all(0)
    d=np.einsum('k,kij->ij',np.asarray(list(weights.values()),dtype=float),x)
    d[~finite]=np.nan
    sums=np.array([np.nansum(d[ids==pid],axis=0) for pid in unique])
    denoms=np.array([np.isfinite(d[ids==pid]).sum(0) for pid in unique])
    den=counts@denoms;dist=np.divide(counts@sums,den,out=np.full(den.shape,np.nan),where=den>0)
    return np.nanmean(d,axis=0),dist,finite.sum(0)

def main():
    checked=0;adjusted=0;absolute=0
    for stage in ['development','outer']:
        frame=read(ROOT/(stage.upper()+'_PATIENT_METRICS_PRIVATE.pkl'))
        base=frame[(frame.config=='000')&(frame['head']=='D1')].sort_values(['fold','patient'])
        ids=base.patient.to_numpy();unique=sorted(set(ids));draw=np.random.default_rng(42).integers(0,len(unique),(10000,len(unique)))
        counts=np.array([(draw==i).sum(axis=1) for i in range(len(unique))]).T
        matrix={(h,c):frame[(frame.config==c)&(frame['head']==h)].sort_values(['fold','patient'])[METRICS].to_numpy() for h in ['D1','M8'] for c in CONFIGS}
        table=pd.read_csv(PUB/stage/'PAIRED_CONTRASTS.csv',dtype={'config':str})
        for label,rows in table.groupby('contrast',sort=True):
            mean,dist,n=replay(matrix,coefficients(rows.iloc[0]),ids,unique,counts)
            intervals=np.nanquantile(dist,[.025,.975],axis=0)
            simultaneous=np.nanquantile(dist,[.05/84,1-.05/84],axis=0)
            for j,m in enumerate(METRICS):
                r=rows[rows.metric==m].iloc[0]
                assert np.allclose([mean[j],*intervals[:,j]],[r.delta,r.ci_low,r.ci_high],atol=1e-12,rtol=0,equal_nan=True),(stage,label,m)
                assert n[j]==r.valid_appearances
                if 'bonferroni_ci_low' in r and np.isfinite(r.bonferroni_ci_low):
                    assert np.allclose(simultaneous[:,j],[r.bonferroni_ci_low,r.bonferroni_ci_high],atol=1e-12,rtol=0);adjusted+=1
                checked+=1
        table=pd.read_csv(PUB/stage/'ABSOLUTE_INTERVALS.csv',dtype={'config':str})
        for (c,h),rows in table.groupby(['config','head']):
            mean,dist,n=replay(matrix,{(h,c):1},ids,unique,counts)
            intervals=np.nanquantile(dist,[.025,.975],axis=0)
            for j,m in enumerate(METRICS):
                r=rows[rows.metric==m].iloc[0]
                assert np.allclose([mean[j],*intervals[:,j]],[r['mean'],r.ci_low,r.ci_high],atol=1e-12,rtol=0,equal_nan=True)
                assert n[j]==r.valid_appearances;absolute+=1
    assert adjusted==84
    write(PUB/'INDEPENDENT_ALL_INTERVALS_AUDIT.json',dict(status='PASS',paired_metric_intervals=checked,adjusted_primary_intervals=adjusted,
        absolute_metric_intervals=absolute,all10000_draws_independently_recomputed=True,identical_patient_pairing=True,
        no_training_inference_or_threshold_selection=True,source_sha256=sha(__file__)))
    print('JOINT_ALL_10000_DRAW_INTERVALS_INDEPENDENTLY_VERIFIED',checked,adjusted,absolute,flush=True)

if __name__=='__main__':main()
