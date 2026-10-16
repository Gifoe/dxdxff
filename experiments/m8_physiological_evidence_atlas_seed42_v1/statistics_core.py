"""Patient-unit effect and fixed descriptive inference; no classifier."""
import numpy as np
from scipy.stats import mannwhitneyu,wilcoxon


def effect(x,y):
    good=np.isfinite(x);a=np.asarray(x)[good];b=np.asarray(y)[good]
    positive=a[b==1];negative=a[b==0]
    if not len(positive) or not len(negative):return np.nan
    u=mannwhitneyu(positive,negative,method='asymptotic').statistic
    return float(2*u/(len(positive)*len(negative))-1)


def interval(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    if not len(a):return np.nan,np.nan,np.nan
    draws=np.random.default_rng(42).integers(0,len(a),(10000,len(a)))
    low,high=np.percentile(a[draws].mean(1),[2.5,97.5])
    return float(a.mean()),float(low),float(high)


def pvalue(a):
    a=np.asarray(a,float);a=a[np.isfinite(a)]
    return 1. if not len(a) or np.all(a==0) else float(wilcoxon(a,zero_method='wilcox',alternative='two-sided',method='auto').pvalue)


def bh(p):
    p=np.asarray(p,float);order=np.argsort(p);adjusted=p[order]*len(p)/np.arange(1,len(p)+1)
    adjusted=np.minimum.accumulate(adjusted[::-1])[::-1];out=np.empty_like(p);out[order]=np.minimum(adjusted,1.)
    return out


def old_summary(g):
    a=g.effect.to_numpy();a=a[np.isfinite(a)];mu,lo,hi=interval(a);sources=[]
    for center,h in g.groupby('center'):
        values=h.effect.to_numpy();values=values[np.isfinite(values)]
        if len(values)>=5:sources.append(bool(np.sign(values.mean())==np.sign(mu)))
    agree=g.cross_seizure_sign_agreement.to_numpy();agree=agree[np.isfinite(agree)]
    return dict(patients=len(a),mean_effect=mu,ci_low=lo,ci_high=hi,pvalue=pvalue(a),
        patient_direction_fraction=float((np.sign(a)==np.sign(mu)).mean()) if len(a) else np.nan,
        eligible_source_groups=len(sources),same_direction_source_groups=sum(sources),repeat_seizure_patients=len(agree),
        median_seizure_sign_agreement=float(np.median(agree)) if len(agree) else np.nan)
