from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.signal import welch

ALGORITHM_VERSION = "relative_log_welch_rank_v1"
BANDS = {"delta": (1,4), "theta": (4,8), "alpha": (8,13), "beta": (13,30),
         "gamma": (30,70), "high_gamma": (70,100)}


@dataclass
class SpectralState:
    times_sec: np.ndarray
    maps: dict[str, np.ndarray]
    valid_mask: np.ndarray
    effective_upper: float


def _rank_rows(values):
    output = np.full(values.shape, np.nan)
    for i,row in enumerate(values):
        valid = np.isfinite(row); n=valid.sum()
        if not n: continue
        order=np.argsort(row[valid],kind="stable"); ranks=np.empty(n); ranks[order]=np.arange(1,n+1); output[i,valid]=(ranks-.5)/n
    return output


def compute_spectral_state(record, window_sec: float=.5, step_sec: float=.25) -> tuple[SpectralState, dict]:
    fs=float(record.sampling_rate); upper=min(100.,.40*fs); width=max(16,round(window_sec*fs)); step=max(1,round(step_sec*fs))
    starts=np.arange(0,max(record.signal.shape[1]-width+1,0),step,dtype=int); c=len(record.channel_names)
    raw={name:np.full((len(starts),c),np.nan) for name in (*BANDS,"low_entropy")}
    for wi,start in enumerate(starts):
        f,p=welch(np.nan_to_num(record.signal[:,start:start+width]),fs=fs,nperseg=width,axis=1)
        total_mask=(f>=1)&(f<=upper); total=np.trapezoid(p[:,total_mask],f[total_mask],axis=1)
        spectrum=p[:,total_mask]; denom=spectrum.sum(axis=1,keepdims=True); prob=np.divide(spectrum,denom,out=np.zeros_like(spectrum),where=denom>0)
        entropy=-(prob*np.log(prob+1e-12)).sum(axis=1)/max(np.log(max(spectrum.shape[1],2)),1e-12)
        raw["low_entropy"][wi]=1-_rank_rows(entropy[None,:])[0]
        for name,(lo,hi0) in BANDS.items():
            hi=min(hi0,upper); mask=(f>=lo)&(f<hi)
            if hi<=lo or mask.sum()<2: continue
            power=np.trapezoid(p[:,mask],f[mask],axis=1); relative=np.divide(power,total,out=np.full(c,np.nan),where=total>0)
            raw[name][wi]=_rank_rows(np.log(np.maximum(relative,1e-12))[None,:])[0]
    valid=np.broadcast_to(record.valid_channel_mask,(len(starts),c)).copy()
    for name in raw: raw[name][~valid]=np.nan
    times=(starts+width/2-record.onset_sample)/fs
    return SpectralState(times,raw,valid,upper), {"patient_key":record.patient_key,"seizure_id":record.seizure_id,"n_windows":len(starts),"effective_upper":upper,"algorithm_version":ALGORITHM_VERSION}


__all__=["ALGORITHM_VERSION","BANDS","SpectralState","compute_spectral_state"]
