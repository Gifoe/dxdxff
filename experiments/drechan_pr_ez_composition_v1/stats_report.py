"""Patient inference; fixed two-primary and eleven-secondary families."""
import csv
import numpy as np
import core
PRIMARY=['E1_H_FULL','E2_D_FULL_25_MINUS_100']
SECONDARY=['E1_H_SOURCE','E1_H_CENTER','E1_H_SCALE','E2_CRPS_FULL_50_MINUS_100','E2_CRPS_FULL_75_MINUS_100','E2_ABSGAP_25_MINUS_100','E2_ABSGAP_50_MINUS_100','E2_ABSGAP_75_MINUS_100','COMPOSITION_SLOPE_25','COMPOSITION_SLOPE_50','COMPOSITION_SLOPE_75']

class Inference:
    def __init__(self,rows):
        self.sources=np.array([r['source'] for r in rows]);self.draw=core.bootstrap_indices(self.sources)
        self.counts=np.array([np.bincount(q,minlength=len(rows)) for q in self.draw],dtype=float)
    def summary(self,v,scope='ALL',family=0):
        ix=np.arange(len(v)) if scope=='ALL' else np.flatnonzero(self.sources==scope)
        v=np.asarray(v,float)[ix];valid=np.isfinite(v);n=int(valid.sum());w=self.counts[:,ix]
        den=w@valid.astype(float);empty=int((den==0).sum())
        out=dict(mean='',median='',ci_low='',ci_high='',family_low='',family_high='',patients=n,assessed_patients=len(ix),empty_bootstrap_draws=empty,status='not_estimable')
        if n==0:return out
        if n==1:out['status']='single_patient_numerics_withheld';return out
        z=v[valid]
        draws=np.full(int((den>0).sum()),z[0]) if np.ptp(z)==0 else (w@np.where(valid,v,0))[den>0]/den[den>0]
        out.update(mean=float(z.mean()),median=float(np.median(z)),ci_low=float(np.quantile(draws,.025)),ci_high=float(np.quantile(draws,.975)),status='estimable')
        if family:out.update(zip(['family_low','family_high'],map(float,np.quantile(draws,[.05/(2*family),1-.05/(2*family)]))))
        return out

def endpoint_arrays(rows):
    keys=['e1_class','e1_all','e1_h','e1_null','e2_all','e2_class','e2_h','e2_abs_class','e2_abs_all','support','unsupported_nonconstant','composition_slope','composition_denominator','composition_mean','composition_min','composition_max','reference_counts']
    return {k:np.array([r[k] for r in rows]) for k in keys}

def contrast_vectors(a):
    h=a['e1_h'].mean(-1);loss=a['e2_all'].mean(-1);gap=(a['e2_abs_class'][:,:,0]-a['e2_abs_class'][:,:,1]).mean(-1)
    out={'E1_H_FULL':h[:,4],'E2_D_FULL_25_MINUS_100':loss[:,0,4]-loss[:,3,4]}
    for arm in ['SOURCE','CENTER','SCALE']:out['E1_H_'+arm]=h[:,core.ARMS.index(arm)]
    for f,pct in [(1,50),(2,75)]:out[f'E2_CRPS_FULL_{pct}_MINUS_100']=loss[:,f,4]-loss[:,3,4]
    for f,pct in enumerate([25,50,75]):
        out[f'E2_ABSGAP_{pct}_MINUS_100']=gap[:,f]-gap[:,3];out[f'COMPOSITION_SLOPE_{pct}']=a['composition_slope'][:,f]
    return out

def summarize(rows):
    a=endpoint_arrays(rows);inf=Inference(rows)
    names=['E1_CLASS_LOSS','E1_GAIN','E1_H','E2_LOSS','E2_GAIN','E2_H','E2_ANOMALY','E2_SUPPORT','REFERENCE_COMPOSITION','COMPOSITION_SLOPE','PRIMARY','SECONDARY','H_PERMUTATION','H_PERMUTATION_SUMMARY']
    out={k:[] for k in names}
    def emit(name,v,scope,coordinate='ALL12',family=0,**tags):out[name].append(dict(scope=scope,coordinate=coordinate,**tags,**inf.summary(v,scope,family)))
    for scope in ['ALL']+core.SOURCES:
        for j,c in [(-1,'ALL12')]+list(enumerate(core.COORDS)):
            def coord(v):return v.mean(-1) if j==-1 else v[...,j]
            lc=coord(a['e1_class']);hh=coord(a['e1_h']);l2=coord(a['e2_all']);c2=coord(a['e2_class']);h2=coord(a['e2_h']);ac=coord(a['e2_abs_class']);aa=coord(a['e2_abs_all'])
            for ai,arm in enumerate(core.ARMS):
                for gi,group in enumerate(['EZ','NEZ']):
                    emit('E1_CLASS_LOSS',lc[:,gi,ai],scope,c,arm=arm,group=group);emit('E1_GAIN',lc[:,gi,0]-lc[:,gi,ai],scope,c,arm=arm,group=group)
                emit('E1_H',hh[:,ai],scope,c,arm=arm,identity=arm=='GLOBAL')
                for f,fraction in enumerate(core.FRACTIONS):
                    emit('E2_LOSS',l2[:,f,ai],scope,c,fraction=fraction,arm=arm,group='ALL')
                    for gi,group in enumerate(['EZ','NEZ']):emit('E2_LOSS',c2[:,f,gi,ai],scope,c,fraction=fraction,arm=arm,group=group)
                    emit('E2_H',h2[:,f,ai],scope,c,fraction=fraction,arm=arm,identity=arm=='GLOBAL')
                    if j==-1:
                        emit('E2_GAIN',l2[:,f,0]-l2[:,f,ai],scope,c,fraction=fraction,arm=arm,group='ALL')
                        for gi,group in enumerate(['EZ','NEZ']):emit('E2_GAIN',c2[:,f,gi,0]-c2[:,f,gi,ai],scope,c,fraction=fraction,arm=arm,group=group)
            for f,fraction in enumerate(core.FRACTIONS):
                for gi,group in enumerate(['EZ','NEZ']):emit('E2_ANOMALY',ac[:,f,gi],scope,c,fraction=fraction,endpoint=group)
                emit('E2_ANOMALY',aa[:,f],scope,c,fraction=fraction,endpoint='ALL');emit('E2_ANOMALY',ac[:,f,0]-ac[:,f,1],scope,c,fraction=fraction,endpoint='gap')
                for key in ['support','unsupported_nonconstant']:emit('E2_SUPPORT',coord(a[key])[:,f],scope,c,fraction=fraction,endpoint=key)
        for f,fraction in enumerate(core.FRACTIONS):
            for key in ['composition_mean','composition_min','composition_max','reference_counts']:emit('REFERENCE_COMPOSITION',a[key][:,f],scope,fraction=fraction,endpoint=key)
            emit('COMPOSITION_SLOPE',a['composition_slope'][:,f] if f<3 else np.full(len(rows),np.nan),scope,fraction=fraction,unit='CRPS_per10pp_EZ')
        ix=np.arange(len(rows)) if scope=='ALL' else np.flatnonzero(inf.sources==scope)
        for k,arm in enumerate(core.ARMS[1:],1):
            null=a['e1_null'][ix,:,k].mean(0);observed=float(a['e1_h'][ix,k].mean())
            out['H_PERMUTATION'].extend(dict(scope=scope,arm=arm,permutation=b,H=float(v)) for b,v in enumerate(null))
            tie=1e-10+1e-9*abs(observed)
            out['H_PERMUTATION_SUMMARY'].append(dict(scope=scope,arm=arm,observed_H=observed,null_mean=float(null.mean()),null_low=float(np.quantile(null,.025)),null_high=float(np.quantile(null,.975)),conditional_plus_one_upper=float((1+(null>=observed-tie).sum())/1001),conditional_plus_one_lower=float((1+(null<=observed+tie).sum())/1001),patients=len(ix),permutations=1000))
    v=contrast_vectors(a)
    for name in PRIMARY:emit('PRIMARY',v[name],'ALL',family=2,endpoint=name)
    for name in SECONDARY:emit('SECONDARY',v[name],'ALL',family=11,endpoint=name)
    return out

def export(directory,tables):
    for name,rows in tables.items():
        p=directory/(name+'.csv');assert not p.exists();keys=list(dict.fromkeys(k for r in rows for k in r))
        with p.open('w',encoding='utf-8',newline='') as f:
            w=csv.DictWriter(f,fieldnames=keys);w.writeheader();w.writerows(rows)
