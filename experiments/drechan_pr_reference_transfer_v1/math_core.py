"""Unlabeled independent-channel distribution forecasts; no classifiers."""
import hashlib
import numpy as np
from scipy.special import ndtr

ARMS=['GLOBAL','SOURCE','CENTER','SCALE','FULL']
COORDS=[f'{f}_{s}' for f in ['beta','low_gamma','partial_high_gamma','RMS','line_length','spectral_entropy'] for s in ['mean','SD']]
SOURCES=['hup','lzu','multicenter','pediatric']
REPEATS=100

def crps(x,mu,sd):
    x,mu,sd=np.broadcast_arrays(np.asarray(x,float),np.asarray(mu,float),np.asarray(sd,float))
    assert np.isfinite(x).all() and np.isfinite(mu).all() and np.isfinite(sd).all() and (sd>=0).all()
    a=np.abs(x-mu)
    with np.errstate(over='ignore',divide='ignore',invalid='ignore'):
        z=np.divide(a,sd,out=np.zeros_like(a),where=sd>0)
        density=np.exp(-.5*z*z)/np.sqrt(2*np.pi)
        v=a*(2*ndtr(z)-1)+sd*(2*density-1/np.sqrt(np.pi))
    v=np.where(sd==0,a,v)
    assert np.isfinite(v).all() and (v>=-1e-12).all()
    return v

def moments(x):
    x=np.asarray(x,float);assert x.ndim==2 and len(x)>0 and np.isfinite(x).all()
    mu=x.mean(0);sd=x.std(0,ddof=0);sd=np.where(np.ptp(x,axis=0)==0,0.,sd)
    return mu,sd

def split_masks(patient,channels):
    channels=np.asarray(channels).astype(str);n=len(channels)
    assert n>=4 and len(set(channels))==n
    out=np.zeros((REPEATS,n),bool)
    for r in range(REPEATS):
        order=sorted(range(n),key=lambda i:(hashlib.sha256(f'42|PRREFV1|{patient}|{r}|{channels[i]}'.encode()).digest(),channels[i]))
        out[r,order[:n//2]]=True
    return out

def reference_parameters(x,masks):
    return np.array([[moments(x[mask]),moments(x[~mask])] for mask in masks])

def relative_query(x,mu,sd):
    return np.divide(x-mu,sd,out=np.zeros_like(x),where=sd>1e-8)

def patient_endpoints(p):
    x=p['x'];n,d=x.shape
    scores=np.zeros((n,5,d));dev=np.zeros((n,d));unsupported=np.zeros((n,d));cross=np.zeros((n,d))
    for r,mask in enumerate(p['masks']):
        for side,ref in enumerate([mask,~mask]):
            query=~ref;mu,sd=p['references'][r,side]
            params=[p['global'],p['source_stats'],[mu,p['global'][1]],[p['global'][0],sd],[mu,sd]]
            for j,(loc,scale) in enumerate(params):scores[query,j]+=crps(x[query],loc,scale)/REPEATS
            dev[query]+=np.abs(relative_query(x[query],mu,sd))/REPEATS
            unsupported[query]+=(sd<=1e-8)/REPEATS
            cross[query]+=((sd<=1e-8)&(np.abs(x[query]-mu)>1e-8))/REPEATS
    return dict(crps=scores.mean(0),deviation=dev,unsupported=unsupported.mean(0),unsupported_query_nonconstant=cross.mean(0))

def permuted_gaps(dev,y,patient):
    rng=np.random.default_rng(int.from_bytes(hashlib.sha256(f'42|PRREFLABEL|{patient}'.encode()).digest()[:16],'little'))
    result=[];v=dev.mean(1)
    for _ in range(1000):
        z=rng.permutation(y);result.append(v[z==1].mean()-v[z==0].mean())
    return np.array(result)

def bootstrap_indices(sources):
    sources=np.asarray(sources);rng=np.random.default_rng(42)
    return np.concatenate([rng.choice(np.flatnonzero(sources==s),size=(10000,int((sources==s).sum())),replace=True) for s in SOURCES],axis=1)
