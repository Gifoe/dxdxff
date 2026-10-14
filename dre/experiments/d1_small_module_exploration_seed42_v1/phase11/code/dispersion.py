import numpy as np
from scipy.special import logit,expit

def transform_scaled(score,patient,coefficient,exponent):
    z=logit(np.asarray(score,dtype=float).clip(1e-7,1-1e-7));m=np.empty(len(z));s=np.empty(len(z))
    for pid in np.unique(patient):
        ix=patient==pid;m[ix]=np.median(z[ix]);sd=np.std(z[ix],ddof=0);s[ix]=(sd if sd>1e-8 else 1.)**exponent
    return expit((z-coefficient*m)/s),m,s

def cutoff(threshold,median,scale,coefficient):return expit(logit(np.asarray(threshold,dtype=float))*scale+coefficient*median)
