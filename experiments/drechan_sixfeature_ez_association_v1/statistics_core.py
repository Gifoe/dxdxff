"""Patient-unit signed association; no learned model or feature selection."""
import numpy as np
from scipy.stats import rankdata, wilcoxon


def effect(values, ez):
    x=np.asarray(values,dtype=float);y=np.asarray(ez,dtype=bool)
    good=np.isfinite(x);x=x[good];y=y[good]
    n1=int(y.sum());n0=int((~y).sum())
    if not n1 or not n0:return np.nan
    ranks=rankdata(x,method='average')
    u=float(ranks[y].sum()-n1*(n1+1)/2)
    return float((2*u-n1*n0)/(n1*n0))


def bh(pvalues):
    p=np.asarray(pvalues,float);valid=np.isfinite(p)
    conservative=np.where(valid,p,1.)
    order=np.argsort(conservative,kind='stable')
    ranked=conservative[order]*len(p)/np.arange(1,len(p)+1)
    adjusted=np.minimum(1,np.minimum.accumulate(ranked[::-1])[::-1])
    out=np.empty_like(p);out[order]=adjusted;out[~valid]=np.nan
    return out


def summarize(values,draws=10000):
    a=np.asarray(values,float);a=a[np.isfinite(a)];n=len(a)
    if not n:
        return dict(valid_patients=0,mean_effect=np.nan,median_effect=np.nan,ci_low=np.nan,ci_high=np.nan,
                    pvalue=np.nan,positive_percent=np.nan,negative_percent=np.nan,zero_percent=np.nan,
                    nonzero_patients=0,wilcoxon_statistic=np.nan)
    index=np.random.default_rng(42).integers(0,n,size=(draws,n))
    ci=np.quantile(a[index].mean(1),[.025,.975])
    if np.any(a!=0):
        test=wilcoxon(a,alternative='two-sided',zero_method='wilcox',correction=True,method='approx')
        p=float(test.pvalue);stat=float(test.statistic)
    else:p=1.;stat=0.
    return dict(valid_patients=n,mean_effect=float(a.mean()),median_effect=float(np.median(a)),
                ci_low=float(ci[0]),ci_high=float(ci[1]),pvalue=p,
                positive_percent=float(100*np.mean(a>0)),negative_percent=float(100*np.mean(a<0)),
                zero_percent=float(100*np.mean(a==0)),nonzero_patients=int(np.count_nonzero(a)),wilcoxon_statistic=stat)


def permuted_effects(x,ez,permuted_labels):
    """A common within-patient label permutation for all twelve coordinates."""
    x=np.asarray(x,float);labels=np.asarray(permuted_labels,bool)
    assert np.all(labels.sum(1)==np.sum(ez))
    out=np.full((len(labels),x.shape[1]),np.nan)
    for j in range(x.shape[1]):
        good=np.isfinite(x[:,j]);a=x[good,j];y=labels[:,good]
        n1=y.sum(1);n0=len(a)-n1;den=n1*n0;ok=den>0
        if len(a):
            ranks=rankdata(a,method='average')
            u=y.astype(float)@ranks-n1*(n1+1)/2
            out[ok,j]=(2*u[ok]-den[ok])/den[ok]
    return out
