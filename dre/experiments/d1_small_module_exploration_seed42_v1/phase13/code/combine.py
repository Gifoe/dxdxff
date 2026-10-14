import numpy as np
from scipy.special import expit

def combine_logits(logits):
    x=np.asarray(logits,dtype=np.float64);assert x.ndim==2 and 2<=len(x)<=3 and np.isfinite(x).all()
    z=x.mean(0);return z,expit(z)
