"""Six record-level operators: censor-aware aggregation, not prediction."""
import warnings
import numpy as np

GROUPS=['early_partial_HG_80_125_slope','early_LL_slope','baseline_crossing_partial_HG_80_125',
        'baseline_crossing_LL','post_top20_partial_HG_80_125','post_top20_LL']
SOURCE=[0,1,2,3,6,7]


def crossing(series,t):
    pre=t<0;post=t>=0
    assert pre.sum()>=2 and post.sum()>=2 and np.isfinite(series).all()
    sd=series[pre].std(0);threshold=series[pre].mean(0)+2*sd
    eligible=sd>0
    over=series[post]>=threshold
    observed=over.any(0)&eligible
    fraction=np.full(series.shape[1],np.nan,float)
    first=np.argmax(over,axis=0)
    fraction[observed]=t[post][first[observed]]/max(float(np.max(np.abs(t))),1e-6)
    return eligible,observed,fraction


def mean_sd(a):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        return np.nanmean(a,axis=0),np.nanstd(a,axis=0)


def aggregate(scalar,present,eligible,observed,fraction,record_order=None):
    """S,C,6; absent channels never become zero-valued seizures."""
    if record_order is not None:
        order=np.argsort(record_order)
        scalar,present,eligible,observed,fraction=[a[order] for a in [scalar,present,eligible,observed,fraction]]
    scalar=np.asarray(scalar,float);present=np.asarray(present,bool)
    assert scalar.shape[:2]==present.shape and scalar.shape[2]==6
    out=np.full((scalar.shape[1],12),np.nan,float)
    for j in [0,1,4,5]:
        a=np.where(present,scalar[:,:,j],np.nan)
        out[:,2*j],out[:,2*j+1]=mean_sd(a)
    for k,j in enumerate([2,3]):
        valid=eligible[:,:,k]&present;obs=observed[:,:,k]&valid
        a=np.where(obs,fraction[:,:,k],np.nan)
        out[:,2*j]=mean_sd(a)[0]
        n=valid.sum(0)
        out[:,2*j+1]=np.divide(obs.sum(0),n,out=np.full(len(n),np.nan),where=n>0)
    return out


def patient_z(x,patients,legacy=False):
    out=np.full_like(x,np.nan,dtype=float)
    for pid in np.unique(patients):
        ix=patients==pid;a=np.asarray(x[ix],float);mu,sd=mean_sd(a)
        scale=np.maximum(sd,1e-8) if legacy else np.where(sd>0,sd,1.)
        out[ix]=(a-mu)/scale
    return out


def prepare(z,train):
    mu,_=mean_sd(z[train]);mu=np.where(np.isfinite(mu),mu,0.)
    filled=np.where(np.isnan(z),mu,z)
    center,sd=mean_sd(filled[train]);scale=np.where(sd>0,sd,1.)
    return (filled-center)/scale,dict(imputer=mu,mean=center,scale=scale)


def nonconstant_rate(a,patients):
    rates=[]
    for j in range(a.shape[1]):
        passed=[]
        for p in np.unique(patients):
            v=a[patients==p,j];v=v[np.isfinite(v)]
            passed.append(bool(len(v)>1 and np.ptp(v)>64*np.finfo(np.float32).eps*np.max(np.abs(v))))
        rates.append(float(np.mean(passed)))
    return np.array(rates)
