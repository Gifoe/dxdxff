"""Fixed, label-blind shape candidates on original valid250Hz/500sample windows.

No filters,resampling,subsampling,full80-150HzHFO or low-frequency recovery.
This restricted-bin spectral slope is deliberately NOT called FOOOF/aperiodic EI.
"""
import numpy as np
from scipy.signal import welch

NAMES=['log_variance_stable','hjorth_mobility_per_second','hjorth_complexity',
       'permutation_entropy_m3_tau1','spectral_centroid_13_95','log_gamma_beta_ratio',
       'partial_high_gamma_fraction_80_125','spectral_flatness_13_95',
       'spectral_slope_proxy_20_95_notchgap','normalized_line_length','clipped_excess_kurtosis']
FLOOR=1e-30


def permutation_entropy(x):
    n=len(x);triples=np.stack([x[:,:-2],x[:,1:-1],x[:,2:]],axis=-1)
    order=np.argsort(triples,axis=-1,kind='stable')
    code=order[...,0]*9+order[...,1]*3+order[...,2]
    count=np.bincount((code+27*np.arange(n)[:,None]).ravel(),minlength=27*n).reshape(n,27)
    p=count/count.sum(1,keepdims=True)
    return -(p*np.log(np.maximum(p,1e-30))).sum(1)/np.log(6.)


def extract(wave):
    x=np.asarray(wave,dtype=np.float64);assert x.ndim==2 and x.shape[1]==500 and np.isfinite(x).all()
    centered=x-x.mean(1,keepdims=True);v=np.mean(centered**2,axis=1)
    first=np.diff(x,axis=1);second=np.diff(first,axis=1)
    v1=first.var(1);v2=second.var(1)
    mob=250*np.sqrt(v1/np.maximum(v,FLOOR))
    comp=np.sqrt(v2/np.maximum(v1,FLOOR))/np.maximum(np.sqrt(v1/np.maximum(v,FLOOR)),1e-15)
    f,psd=welch(x,fs=250,window='hann',nperseg=500,noverlap=250,detrend=False,scaling='density',axis=-1)
    def power(lo,hi):
        take=(f>=lo)&(f<hi)
        return np.trapezoid(psd[:,take],f[take],axis=1)
    beta=power(13,30);gamma=power(30,80);high=power(80,125);total=power(13,125)
    take=(f>=13)&(f<=95);p=psd[:,take];ff=f[take];p=p/np.maximum(p.sum(1,keepdims=True),FLOOR)
    centroid=(p*ff).sum(1);flat=np.exp(np.log(np.maximum(p,1e-30)).mean(1))/np.maximum(p.mean(1),FLOOR)
    select=((f>=20)&(f<45))|((f>65)&(f<=95))
    logf=np.log(f[select]);xx=logf-logf.mean();yy=np.log(np.maximum(psd[:,select],FLOOR))
    slope=((yy-yy.mean(1,keepdims=True))*xx).sum(1)/(xx**2).sum()
    normalized_ll=np.abs(first).sum(1)/(500/250)/np.maximum(np.sqrt(v),1e-15)
    z=np.clip(centered/np.maximum(np.sqrt(v),1e-15)[:,None],-8,8)
    kurt=(z**4).mean(1)/np.maximum((z*z).mean(1)**2,FLOOR)-3
    values=np.stack([np.log(np.maximum(v,FLOOR)),mob,comp,permutation_entropy(x),centroid,
        np.log(np.maximum(gamma,FLOOR))-np.log(np.maximum(beta,FLOOR)),high/np.maximum(total,FLOOR),flat,
        slope,normalized_ll,kurt],axis=1)
    constant=v<=FLOOR
    values[constant,1:]=0  # Constant waveforms have no supported shape, not infinities.
    assert values.shape==(len(x),len(NAMES)) and np.isfinite(values).all()
    quality=dict(variance=v,mean_square=(x*x).mean(1),rms_floor_fraction=1e-8/((x*x).mean(1)+1e-8),constant=constant)
    return values.astype(np.float32),quality


def delta_summary(window_values,mask,times):
    """Original float32 pre-reference -> all-valid mean -> available-seizure stats."""
    values=np.asarray(window_values,dtype=np.float32);mask=np.asarray(mask,bool);times=np.asarray(times)
    S,C,W,F=values.shape;delta=np.zeros_like(values);seizure=np.full((S,C,F),np.nan,np.float32)
    for s in range(S):
        for c in range(C):
            valid=mask[s,c]
            if not valid.any():continue
            pre=valid&(times[s]<0);assert pre.any()
            base=values[s,c,pre].mean(0)
            delta[s,c,valid]=values[s,c,valid]-base
            seizure[s,c]=delta[s,c,valid].mean(0)
    mu=np.empty((C,F),np.float32);sd=mu.copy()
    for c in range(C):
        for k in range(F):
            a=seizure[:,c,k];a=a[np.isfinite(a)];assert len(a)>0
            mu[c,k]=a.mean();sd[c,k]=a.std(ddof=0)
    return delta,np.stack([mu,sd],axis=-1).reshape(C,2*F)
