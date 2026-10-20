"""Unit-consistent direct physiology PR, fitted only on original fold FIT."""
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

EPSILON = 1e-8


def relative(raw, patient, units):
    raw = np.asarray(raw, np.float64)
    patient = np.asarray(patient).astype(str)
    units = np.asarray(units, np.float64)
    assert raw.shape == (len(patient), 14) and units.shape == (14,)
    assert np.isfinite(units).all() and (units > 0).all() and not np.isinf(raw).any()
    result = np.full_like(raw, np.nan)
    for pid in np.unique(patient):
        ix = patient == pid
        value = raw[ix]
        mean = np.nanmean(value, axis=0)
        sd = np.nanstd(value, axis=0, ddof=0)
        supported = np.isfinite(sd) & (sd > EPSILON * units)
        z = np.divide(value-mean, sd, out=np.zeros_like(value), where=supported[None, :])
        z[np.isnan(value)] = np.nan
        result[ix] = z
    return result


def prepare(raw, patient, fit):
    raw = np.asarray(raw, np.float64)
    original_imputer = SimpleImputer().fit(raw[fit])
    unit_scaler = StandardScaler().fit(original_imputer.transform(raw[fit]))
    units = unit_scaler.scale_
    assert units.shape == (14,) and np.isfinite(units).all() and (units > 0).all()
    z = relative(raw, patient, units)
    imputer = SimpleImputer().fit(z[fit])
    scaler = StandardScaler().fit(imputer.transform(z[fit]))
    result = scaler.transform(imputer.transform(z)).astype(np.float32)
    assert result.shape == raw.shape and np.isfinite(result).all()
    pre = dict(imputer_statistics=imputer.statistics_, mean=scaler.mean_, scale=scaler.scale_,
               var=scaler.var_, FIT_unit_scale=units.copy(), FIT_unit_variance=unit_scaler.var_.copy(),
               fit_patient_ids=sorted(set(np.asarray(patient)[fit])))
    return result, pre


def query(raw, patient, pre):
    z = relative(raw, patient, pre['FIT_unit_scale'])
    return ((np.where(np.isnan(z), pre['imputer_statistics'], z)-pre['mean'])/pre['scale']).astype(np.float32)
