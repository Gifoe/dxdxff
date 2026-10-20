"""TRAIN-unit-consistent patient contrast; no label input,clipping or feature loss."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

EPS=1e-8


def fit_units(raw,train):
    x=np.asarray(raw,np.float64)
    imputer=SimpleImputer();imputer.fit(x[train]);scaler=StandardScaler();scaler.fit(imputer.transform(x[train]))
    assert scaler.scale_.shape==(14,) and np.isfinite(scaler.scale_).all() and (scaler.scale_>0).all()
    return scaler.scale_.copy(),scaler.var_.copy()


def relative(raw,patient,units):
    x=np.asarray(raw,np.float64);patient=np.asarray(patient).astype(str);units=np.asarray(units,np.float64)
    assert x.shape==(len(patient),14) and units.shape==(14,) and np.isfinite(units).all() and (units>0).all()
    assert not np.isinf(x).any();out=np.full_like(x,np.nan)
    for pid in np.unique(patient):
        ix=patient==pid;a=x[ix];mu=np.nanmean(a,axis=0);sd=np.nanstd(a,axis=0,ddof=0)
        good=np.isfinite(sd)&(sd>EPS*units)
        q=np.divide(a-mu,sd,out=np.zeros_like(a),where=good[None,:])
        q[np.isnan(a)]=np.nan;out[ix]=q
    return out


def prepare(raw,patient,train):
    units,unitvar=fit_units(raw,train);z=relative(raw,patient,units)
    imputer=SimpleImputer();imputer.fit(z[train]);scaler=StandardScaler();scaler.fit(imputer.transform(z[train]))
    result=scaler.transform(imputer.transform(z)).astype(np.float32)
    assert result.shape==np.asarray(raw).shape and result.shape[1]==14 and np.isfinite(result).all()
    pre=dict(imputer_statistics=imputer.statistics_,mean=scaler.mean_,scale=scaler.scale_,var=scaler.var_,
             TRAIN_unit_scale=units,TRAIN_unit_variance=unitvar,fit_patient_ids=sorted(set(np.asarray(patient)[train])))
    return result,pre
