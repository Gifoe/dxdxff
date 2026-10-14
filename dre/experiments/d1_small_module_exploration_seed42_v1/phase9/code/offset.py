"""Observable patient offset; no label argument in inference helpers."""
import numpy as np
from scipy.special import expit,logit

def transform(score,patient,coefficient):
    p=np.asarray(score,dtype=float);z=logit(p.clip(1e-7,1-1e-7));m=np.empty(len(p))
    for pid in np.unique(patient):m[patient==pid]=np.median(z[patient==pid])
    return expit(z-coefficient*m),m

def effective(threshold,median,coefficient):return expit(logit(np.asarray(threshold,dtype=float))+coefficient*median)
