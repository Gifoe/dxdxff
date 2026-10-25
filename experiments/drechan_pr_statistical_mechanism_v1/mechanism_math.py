"""Threshold-free TRAIN-derived geometric probes; no neural or classifier runtime."""
import hashlib
import numpy as np

FACTOR=['G0_GLOBAL','G1_CENTER','G2_SCALE','G3_FULL']
ARMS=FACTOR+['DONOR_FULL','OP_GLOBAL','OP_PR']
ESTIMATORS=['DIRECT','FISHER']
FACTORS=[1e-8,1e-6,1e-10]
NORM_FLOOR=1e-12


def fit_global(raw):
    a=np.asarray(raw,float)
    assert a.ndim==2 and not np.isinf(a).any()
    valid=np.isfinite(a);count=valid.sum(0)
    fill=np.divide(np.where(valid,a,0).sum(0),count,out=np.zeros(a.shape[1]),where=count>0)
    x=np.where(valid,a,fill);mu=x.mean(0);sd=x.std(0,ddof=0)
    constant=(np.ptp(x,axis=0)==0)|(sd==0)
    return dict(fill=fill,mean=mu,scale=np.where(constant,1.,sd),constant=constant)


def global_query(raw,pre):
    a=np.asarray(raw,float);assert not np.isinf(a).any()
    return (np.where(np.isnan(a),pre['fill'],a)-pre['mean'])/pre['scale']


def patient_stats(x):
    mu=x.mean(0);sd=x.std(0,ddof=0);good=sd>1e-8
    return dict(mean=mu,sd=sd,support=good,safe=np.where(good,sd,1.))


def represent(x,own,donor=None):
    out=dict(zip(FACTOR,[x,x-own['mean'],x/own['safe'],(x-own['mean'])/own['safe']]))
    if donor is not None:out['DONOR_FULL']=(x-donor['mean'])/donor['safe']
    return out


def donor_map(patients,train,sources,counts,fold):
    result={}
    for p in sorted(patients):
        candidates=[q for q in sorted(train) if q!=p];assert candidates
        def key(q):
            return (sources[q]!=sources[p],int(np.floor(np.log2(counts[q])))!=int(np.floor(np.log2(counts[p]))),
                abs(np.log2(counts[q]/counts[p])),hashlib.sha256(f'42|{fold}|{p}|{q}'.encode()).digest())
        result[p]=min(candidates,key=key)
    return result


def moments(x,y):
    y=np.asarray(y,int);assert set(y)=={0,1}
    center=x.mean(0);res=x-center
    a=res[y==1];b=res[y==0];ma=a.mean(0);mb=b.mean(0)
    aa=a-ma;bb=b-mb
    return dict(delta=ma-mb,midpoint=center+(ma+mb)/2,
        covariance=.5*(aa.T@aa/len(aa)+bb.T@bb/len(bb)))


def direction(ms,estimator):
    delta=np.mean([q['delta'] for q in ms],axis=0);cov=np.mean([q['covariance'] for q in ms],axis=0)
    lam=max(.1*float(np.trace(cov))/len(delta),1e-12)
    w=delta if estimator=='DIRECT' else np.linalg.solve(cov+lam*np.eye(len(delta)),delta)
    norm=float(np.linalg.norm(w));v=w/norm if norm>NORM_FLOOR else np.zeros_like(w)
    noise=max(float(v@cov@v),0.)
    return dict(v=v,covariance=cov,delta=delta,norm=norm,degenerate=norm<=NORM_FLOOR,
        noise=noise,reference=float(np.mean([q['midpoint']@v for q in ms])),fisher_lambda=lam)


def endpoints(ms,model,factor=1e-8):
    v=model['v'];norm=float(np.linalg.norm(ms['delta']))
    alignment=float(ms['delta']@v/norm) if norm>NORM_FLOOR else 0.
    numerator=float(ms['delta']@v);variance=max(float(v@ms['covariance']@v),0.)
    eps=factor*max(model['noise'],1e-12);scale=np.sqrt(model['noise']+eps)
    midpoint=float(ms['midpoint']@v);z=(midpoint-model['reference'])/scale
    return dict(T=numerator/np.sqrt(variance+eps),A=alignment,z=z,MAE=abs(z),
        degenerate_delta=norm<=NORM_FLOOR,zero_direction=model['degenerate'],epsilon=eps)


def permute(y,ids,fold,b,assessment=False):
    rng=np.random.default_rng([42,20261026,fold,b,int(assessment)])
    out=np.asarray(y).copy()
    for p in sorted(set(ids)):
        ix=np.flatnonzero(ids==p);out[ix]=rng.permutation(y[ix])
        assert out[ix].sum()==y[ix].sum()
    return out


def aggregate(rows):
    import pandas as pd
    df=pd.DataFrame(rows)
    keys=['estimator','factor','permutation','diagnostic','arm','patient','source']
    return df.groupby(keys,sort=True,dropna=False)[['T','A','z','MAE']].mean().reset_index()


def boot_ids(n):return np.random.default_rng(42).integers(n,size=(10000,n))


def summary(a,ids=None):
    a=np.asarray(a,float);assert len(a)>0 and np.isfinite(a).all()
    ids=boot_ids(len(a)) if ids is None else ids
    samples=a[ids].mean(1);lo,hi=np.quantile(samples,[.025,.975])
    return dict(mean=float(a.mean()),median=float(np.median(a)),ci_low=float(lo),ci_high=float(hi),
        positive_fraction=float(np.mean(a>1e-10)),negative_fraction=float(np.mean(a< -1e-10)),patients=len(a))


def contrast(a,b,metric,ids,identity=False,family=False):
    a=np.asarray(a,float);b=np.asarray(b,float)
    if metric=='SD':
        val=float(b.std(ddof=1)-a.std(ddof=1));samples=b[ids].std(1,ddof=1)-a[ids].std(1,ddof=1)
        med=np.nan;positive=np.nan
    else:
        dif=(b-a) if metric=='MAE' else (a-b);val=float(dif.mean());samples=dif[ids].mean(1)
        med=float(np.median(dif));positive=float(np.mean(dif>1e-10))
    lo,hi=np.quantile(samples,[.025,.975]);al,ah=np.quantile(samples,[.025/25,1-.025/25])
    if identity:
        assert max(abs(val),abs(lo),abs(hi))<1e-8
    return dict(effect=val,median_paired=med,positive_fraction=positive,ci_low=float(lo),ci_high=float(hi),
        adjusted_ci_low=float(al) if family else np.nan,adjusted_ci_high=float(ah) if family else np.nan,
        family_size=25 if family else 0,mathematical_identity=identity,patients=len(a))
