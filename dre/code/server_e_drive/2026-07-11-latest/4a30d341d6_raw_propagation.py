from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd
from scipy.signal import butter, hilbert, sosfiltfilt

from .schema import RawRunRecord


RAW_FEATURES=("raw__target_to_outside_delay_mean","raw__target_to_outside_delay_std","raw__target_to_outside_delay_min","raw__earliest_outside_recruitment_mean","raw__earliest_outside_recruitment_min","raw__outside_recruited_1s_mean","raw__outside_recruited_3s_mean","raw__outside_recruited_5s_mean","raw__outside_recruited_3s_std","raw__early_outside_minus_target_fraction_3s_mean","raw__early_outside_channel_recurrence","raw__outside_unrecruited_fraction_mean")


def _rolling_mean(values:np.ndarray,width:int,step:int)->tuple[np.ndarray,np.ndarray]:
    width=max(2,int(width));step=max(1,int(step));starts=np.arange(0,max(values.shape[1]-width+1,1),step,dtype=int)
    if not starts.size:return np.empty((values.shape[0],0)),starts
    cumulative=np.pad(np.cumsum(values,axis=1),((0,0),(1,0)));return (cumulative[:,starts+width]-cumulative[:,starts])/width,starts


def _robust_scale(signal:np.ndarray)->np.ndarray:
    median=np.nanmedian(signal,axis=1,keepdims=True);mad=np.nanmedian(np.abs(signal-median),axis=1,keepdims=True);return (signal-median)/(1.4826*mad+1e-6)


def recruitment_times(record:RawRunRecord)->tuple[np.ndarray,dict[str,Any]]:
    fs=record.sampling_rate;signal=_robust_scale(record.signal.astype(float));win=max(2,round(.250*fs));step=max(1,round(.050*fs))
    ll,_= _rolling_mean(np.abs(np.diff(signal,axis=1)),max(2,win-1),step);starts=np.arange(ll.shape[1])*step
    gamma_available=fs>=200
    if gamma_available:
        sos=butter(4,[30,80],btype="bandpass",fs=fs,output="sos");filtered=sosfiltfilt(sos,signal,axis=1);envelope=np.log1p(np.abs(hilbert(filtered,axis=1)));gamma,_=_rolling_mean(envelope,win,step)
        n=min(ll.shape[1],gamma.shape[1]);ll,gamma,starts=ll[:,:n],gamma[:,:n],starts[:n]
    times=(starts+win/2-record.onset_sample)/fs;baseline=times<0
    if baseline.sum()<5:raise ValueError(f"MISSING_REQUIRED_FIELD: {record.seizure_id} insufficient preictal raw baseline")
    def zscore(x:np.ndarray)->np.ndarray:
        base=x[:,baseline];median=np.nanmedian(base,axis=1,keepdims=True);mad=np.nanmedian(np.abs(base-median),axis=1,keepdims=True);return (x-median)/(1.4826*mad+1e-6)
    score=zscore(ll)
    if gamma_available:score=np.maximum(score,zscore(gamma))
    persist=max(1,math.ceil(.250/(step/fs)));above=(score>3.)&(times[None,:]>=0);output=np.full(signal.shape[0],np.nan)
    for channel in range(signal.shape[0]):
        conv=np.convolve(above[channel].astype(int),np.ones(persist,dtype=int),mode="valid");indices=np.where(conv>=persist)[0]
        if indices.size:output[channel]=times[indices[0]]
    output[~record.valid_channel_mask]=np.nan
    return output,{"gamma_available":gamma_available,"status":"OK" if gamma_available else "LOW_SAMPLING_RATE_GAMMA_SKIPPED","trajectory_step_sec":step/fs,"threshold":3.,"persistence_sec":.25}


def _fraction(times:np.ndarray,mask:np.ndarray,limit:float)->float:
    if not mask.any():return np.nan
    return float((np.isfinite(times[mask])&(times[mask]<=limit)).mean())


def _jaccard(sets:Sequence[set[str]])->float:
    if len(sets)<2:return np.nan
    values=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            union=sets[i]|sets[j];values.append(len(sets[i]&sets[j])/len(union) if union else 1.)
    return float(np.mean(values))


def _nan_stat(values:Sequence[float],kind:str)->float:
    x=np.asarray(values,dtype=float)
    if not np.isfinite(x).any():return np.nan
    return float(getattr(np,"nan"+kind)(x))


def build_raw_propagation(records:Sequence[RawRunRecord],*,permutation:bool=False,seed:int=42)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rng=np.random.default_rng(seed);by_patient:dict[str,list[dict[str,Any]]]=defaultdict(list);channel_rows=[];seizure_rows=[]
    for record in records:
        times,quality=recruitment_times(record)
        if permutation:
            idx=np.where(record.valid_channel_mask)[0];times=times.copy();times[idx]=times[rng.permutation(idx)]
        target=record.clinical_target_mask&record.valid_channel_mask;outside=~record.clinical_target_mask&record.valid_channel_mask
        ti=times[target];oi=times[outside];tm=float(np.nanmedian(ti)) if np.isfinite(ti).any() else np.nan;om=float(np.nanmedian(oi)) if np.isfinite(oi).any() else np.nan
        seizure={"patient_key":record.patient_key,"center":record.center,"seizure_id":record.seizure_id,"target_recruitment_median":tm,"outside_recruitment_median":om,
            "target_recruitment_min":float(np.nanmin(ti)) if np.isfinite(ti).any() else np.nan,"outside_recruitment_min":float(np.nanmin(oi)) if np.isfinite(oi).any() else np.nan,
            "target_to_outside_delay":om-tm if np.isfinite([om,tm]).all() else np.nan,"earliest_outside_recruitment_time":float(np.nanmin(oi)) if np.isfinite(oi).any() else np.nan,
            "outside_recruited_1s_fraction":_fraction(times,outside,1),"outside_recruited_3s_fraction":_fraction(times,outside,3),"outside_recruited_5s_fraction":_fraction(times,outside,5),
            "target_recruited_1s_fraction":_fraction(times,target,1),"target_recruited_3s_fraction":_fraction(times,target,3),"target_recruited_5s_fraction":_fraction(times,target,5),
            "outside_unrecruited_fraction":float((~np.isfinite(oi)).mean()) if oi.size else np.nan,**quality}
        seizure["early_outside_minus_target_fraction_1s"]=seizure["outside_recruited_1s_fraction"]-seizure["target_recruited_1s_fraction"]
        seizure["early_outside_minus_target_fraction_3s"]=seizure["outside_recruited_3s_fraction"]-seizure["target_recruited_3s_fraction"]
        seizure["early_outside_channels"]={record.channel_names[i] for i in np.where(outside&np.isfinite(times)&(times<=3))[0]};by_patient[record.patient_key].append(seizure)
        seizure_rows.append({k:(";".join(sorted(v)) if isinstance(v,set) else v) for k,v in seizure.items()})
        for i,name in enumerate(record.channel_names):channel_rows.append({"patient_key":record.patient_key,"seizure_id":record.seizure_id,"channel":name,"clinical_target":int(record.clinical_target_mask[i]),"valid":int(record.valid_channel_mask[i]),"recruited":int(np.isfinite(times[i])),"recruitment_time_sec":times[i],**quality})
    patient_rows=[]
    for patient,values in sorted(by_patient.items()):
        get=lambda name:[float(v[name]) for v in values]
        row={"patient_key":patient,"center":values[0]["center"],"raw_valid_seizure_count":len(values),
            "raw__target_to_outside_delay_mean":_nan_stat(get("target_to_outside_delay"),"mean"),"raw__target_to_outside_delay_std":_nan_stat(get("target_to_outside_delay"),"std"),"raw__target_to_outside_delay_min":_nan_stat(get("target_to_outside_delay"),"min"),
            "raw__earliest_outside_recruitment_mean":_nan_stat(get("earliest_outside_recruitment_time"),"mean"),"raw__earliest_outside_recruitment_min":_nan_stat(get("earliest_outside_recruitment_time"),"min"),
            "raw__outside_recruited_1s_mean":_nan_stat(get("outside_recruited_1s_fraction"),"mean"),"raw__outside_recruited_3s_mean":_nan_stat(get("outside_recruited_3s_fraction"),"mean"),"raw__outside_recruited_5s_mean":_nan_stat(get("outside_recruited_5s_fraction"),"mean"),
            "raw__outside_recruited_3s_std":_nan_stat(get("outside_recruited_3s_fraction"),"std"),"raw__early_outside_minus_target_fraction_3s_mean":_nan_stat(get("early_outside_minus_target_fraction_3s"),"mean"),
            "raw__early_outside_channel_recurrence":_jaccard([v["early_outside_channels"] for v in values]),"raw__outside_unrecruited_fraction_mean":_nan_stat(get("outside_unrecruited_fraction"),"mean")}
        patient_rows.append(row)
    return pd.DataFrame(patient_rows),pd.DataFrame(channel_rows),pd.DataFrame(seizure_rows)


__all__=["RAW_FEATURES","build_raw_propagation","recruitment_times"]
