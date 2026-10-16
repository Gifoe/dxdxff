"""Exact-zero mathematical SD guard, never a variance floor or tolerance change."""
import numpy as np
from measurement import patient_z as original


def patient_z(x,patients,legacy=False):
    out=original(x,patients,legacy)
    if legacy:return out
    for pid in np.unique(patients):
        ix=np.flatnonzero(patients==pid)
        for c in range(x.shape[1]):
            finite=ix[np.isfinite(x[ix,c])]
            if len(finite) and np.ptp(x[finite,c])==0:out[finite,c]=0.
    return out
