"""A0/A1 global FIT-only transform, deliberately no patient-local operation."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def prepare(x,patient,train):
    x=np.asarray(x,dtype=np.float64)
    imp=SimpleImputer();sc=StandardScaler()
    imp.fit(x[train]);sc.fit(imp.transform(x[train]))
    out=sc.transform(imp.transform(x)).astype(np.float32)
    pre=dict(imputer_statistics=imp.statistics_,mean=sc.mean_,scale=sc.scale_,var=sc.var_,fit_patient_ids=sorted(set(patient[train])),patient_z=False)
    assert np.isfinite(out).all() and out.shape==x.shape
    assert np.array_equal(out,transform(x,pre))
    return out,pre

def transform(x,pre):
    assert pre['patient_z'] is False
    x=np.asarray(x,dtype=np.float64)
    out=((np.where(np.isnan(x),pre['imputer_statistics'],x)-pre['mean'])/pre['scale']).astype(np.float32)
    assert np.isfinite(out).all()
    return out
