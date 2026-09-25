from __future__ import annotations
import numpy as np
from scipy.signal import resample_poly
from fractions import Fraction
TARGET_FS=250.; TARGET_SAMPLES=500
def preprocess_window(signal,fs):
    x=np.asarray(signal,float); finite=np.isfinite(x); valid=finite.mean()>=.95 and finite.sum()>=int(fs)
    if not valid:return np.zeros(TARGET_SAMPLES,np.float32),False
    x=np.where(finite,x,np.nanmedian(x[finite])); med=np.median(x); mad=np.median(np.abs(x-med)); scale=max(1.4826*mad,1e-6); x=(x-med)/scale; x=np.clip(x,-8,8)
    if (np.abs(x)>=8).mean()>.20 or np.std(x)<1e-6:return np.zeros(TARGET_SAMPLES,np.float32),False
    if fs!=TARGET_FS:
        ratio=Fraction(TARGET_FS/float(fs)).limit_denominator(1000); x=resample_poly(x,ratio.numerator,ratio.denominator)
    if len(x)>=TARGET_SAMPLES:
        start=(len(x)-TARGET_SAMPLES)//2; x=x[start:start+TARGET_SAMPLES]
    else:
        pad=TARGET_SAMPLES-len(x); x=np.pad(x,(pad//2,pad-pad//2))
    return x.astype(np.float32),True
