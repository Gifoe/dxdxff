from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .feature_groups import FAMILY_PRIORITY
from .schema import FeatureRunRecord


PHASES={"preictal":lambda t:t<0,"onset":lambda t:(t>=0)&(t<10),"spread":lambda t:(t>=10)&(t<30),"late":lambda t:t>=30}


def _top_mean(values: np.ndarray, fraction: float=.10) -> float:
    value=np.asarray(values,dtype=float);value=value[np.isfinite(value)]
    if not value.size:return np.nan
    k=min(value.size,max(1,math.ceil(value.size*fraction)));return float(np.partition(value,value.size-k)[-k:].mean())


def _gini(values: np.ndarray) -> float:
    x=np.sort(np.maximum(np.asarray(values,dtype=float),0));x=x[np.isfinite(x)];n=x.size
    if n<=1 or x.sum()<=1e-12:return 0.
    return float(((2*np.arange(1,n+1)-n-1)*x).sum()/(n*x.sum()))


def _entropy(values: np.ndarray) -> float:
    x=np.maximum(np.asarray(values,dtype=float),0);x=x[np.isfinite(x)]
    if x.size<=1 or x.sum()<=1e-12:return 0.
    p=x/x.sum();return float(-(p*np.log(np.maximum(p,1e-12))).sum()/np.log(x.size))


def _mean(values: Sequence[float]) -> float:
    x=np.asarray(values,dtype=float);return float(np.nanmean(x)) if np.isfinite(x).any() else np.nan


def _std(values: Sequence[float]) -> float:
    x=np.asarray(values,dtype=float);return float(np.nanstd(x)) if np.isfinite(x).any() else np.nan


def _max(values: Sequence[float]) -> float:
    x=np.asarray(values,dtype=float);return float(np.nanmax(x)) if np.isfinite(x).any() else np.nan


def _family_values(record: FeatureRunRecord, indices: Sequence[int]) -> np.ndarray:
    raw=record.feature_values[:,:,list(indices)].astype(float); standardized=np.full_like(raw,np.nan)
    for local in range(raw.shape[-1]):
        values=raw[:,:,local];finite=np.isfinite(values);median=np.nanmedian(values);q25,q75=np.nanquantile(values,[.25,.75]);standardized[:,:,local]=(values-median)/(q75-q25+1e-6);standardized[:,:,local][~finite]=np.nan
    return np.nanmedian(standardized,axis=-1)


def _phase_channel(value: np.ndarray, window_mask: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rows=value[window_mask];mask=valid[window_mask];output=np.full(value.shape[1],np.nan)
    for channel in range(value.shape[1]):
        local=rows[:,channel][mask[:,channel]&np.isfinite(rows[:,channel])]
        if local.size:output[channel]=_top_mean(local,.10)
    return output


def _relative(channel: np.ndarray) -> np.ndarray:
    finite=np.isfinite(channel);output=np.full_like(channel,np.nan,dtype=float)
    if finite.any():
        median=np.nanmedian(channel);q25,q75=np.nanquantile(channel,[.25,.75]);output[finite]=(channel[finite]-median)/(q75-q25+1e-6)
    return output


def rank_percentile(channel: np.ndarray) -> np.ndarray:
    value=np.asarray(channel,dtype=float);finite=np.where(np.isfinite(value))[0];output=np.full_like(value,np.nan)
    if finite.size:
        order=finite[np.argsort(value[finite],kind="stable")];output[order]=np.arange(finite.size,dtype=float)/max(finite.size-1,1)
    return output


def _phase_stats(relative: np.ndarray,target: np.ndarray) -> dict[str,float]:
    valid=np.isfinite(relative);inside=valid&target;outside=valid&~target
    if not inside.any() or not outside.any():return {key:np.nan for key in ("global_top10","target_top10","outside_top10","target_outside_gap","entropy","gini","abnormal_fraction","rank_top10")}
    target_top=_top_mean(relative[inside]);outside_top=_top_mean(relative[outside]);positive=np.maximum(relative[valid],0)
    return {"global_top10":_top_mean(relative[valid]),"target_top10":target_top,"outside_top10":outside_top,"target_outside_gap":target_top-outside_top,"entropy":_entropy(positive),"gini":_gini(positive),"abnormal_fraction":float((relative[valid]>2).mean()),"rank_top10":_top_mean(rank_percentile(relative)[valid])}


def _jaccard_mean(sets: Sequence[set[str]]) -> float:
    if len(sets)<2:return np.nan
    values=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            union=sets[i]|sets[j];values.append(len(sets[i]&sets[j])/len(union) if union else 1.)
    return float(np.mean(values))


def build_feature_phenotype(records: Sequence[FeatureRunRecord], groups: dict[str,list[int]]) -> tuple[pd.DataFrame,pd.DataFrame,list[dict[str,Any]]]:
    by_patient:dict[str,list[FeatureRunRecord]]=defaultdict(list)
    for record in records:by_patient[record.patient_key].append(record)
    rows=[];seizure_audit=[];definitions=[]
    active=[family for family in FAMILY_PRIORITY if groups.get(family)][:4]
    for family in active:
        for feature,category in (("spread_rel_global_top10_mean","relative_burden"),("spread_rel_entropy_mean","focality"),("spread_rel_target_outside_gap_mean","target_concordance"),("spread_rel_target_outside_gap_std","cross_seizure_stability"),("spread_rel_target_outside_gap_max","cross_seizure_stability"),("spread_rel_target_outside_gap_max_minus_mean","cross_seizure_stability"),("spread_minus_onset_outside_top10_mean","phase_dynamics"),("spread_minus_onset_outside_top10_std","cross_seizure_stability"),("spread_minus_onset_outside_top10_max","cross_seizure_stability"),("spread_minus_onset_outside_top10_max_minus_mean","cross_seizure_stability"),("spread_top_channel_recurrence","cross_seizure_stability")):
            definitions.append({"feature":f"feature__{family}__{feature}","family":family,"category":category,"definition":"fixed within-patient relative phenotype","model_candidate":True})
    for patient,patient_records in sorted(by_patient.items()):
        row={"patient_key":patient,"center":patient_records[0].center,"n_valid_seizures":len(patient_records)}
        family_seizures={family:[] for family in active};recurrence={family:[] for family in active}
        for record in patient_records:
            centers=(record.window_start_sec+record.window_end_sec)/2
            for family in active:
                value=_family_values(record,groups[family]);phase={}
                for phase_name,selector in PHASES.items():
                    wm=selector(centers);channel=_phase_channel(value,wm,record.valid_mask) if wm.any() else np.full(value.shape[1],np.nan);rel=_relative(channel);phase[phase_name]=_phase_stats(rel,record.clinical_target_mask)
                    if phase_name=="spread" and np.isfinite(rel).any():
                        valid=np.where(np.isfinite(rel))[0];k=min(len(valid),max(1,math.ceil(len(valid)*.10)));top=valid[np.argsort(rel[valid])[-k:]];recurrence[family].append({record.channel_names[i] for i in top})
                seizure={"spread_rel_global_top10":phase["spread"]["global_top10"],"spread_rel_target_outside_gap":phase["spread"]["target_outside_gap"],"spread_rel_outside_top10":phase["spread"]["outside_top10"],"spread_rel_entropy":phase["spread"]["entropy"],"spread_rel_gini":phase["spread"]["gini"],"spread_rank_top10":phase["spread"]["rank_top10"]}
                seizure["onset_minus_preictal_global_top10"]=phase["onset"]["global_top10"]-phase["preictal"]["global_top10"] if np.isfinite([phase["onset"]["global_top10"],phase["preictal"]["global_top10"]]).all() else np.nan
                seizure["spread_minus_onset_outside_top10"]=phase["spread"]["outside_top10"]-phase["onset"]["outside_top10"] if np.isfinite([phase["spread"]["outside_top10"],phase["onset"]["outside_top10"]]).all() else np.nan
                family_seizures[family].append(seizure);seizure_audit.append({"patient_key":patient,"seizure_id":record.seizure_id,"family":family,**seizure})
        for family in active:
            values=family_seizures[family];prefix=f"feature__{family}__"
            row[prefix+"spread_rel_global_top10_mean"]=_mean([item["spread_rel_global_top10"] for item in values]);row[prefix+"spread_rel_entropy_mean"]=_mean([item["spread_rel_entropy"] for item in values])
            for name in ("spread_rel_target_outside_gap","spread_minus_onset_outside_top10"):
                local=[item[name] for item in values];mean=_mean(local);maximum=_max(local);row[prefix+name+"_mean"]=mean;row[prefix+name+"_std"]=_std(local);row[prefix+name+"_max"]=maximum;row[prefix+name+"_max_minus_mean"]=maximum-mean if np.isfinite([maximum,mean]).all() else np.nan
            row[prefix+"spread_top_channel_recurrence"]=_jaccard_mean(recurrence[family])
        rows.append(row)
    return pd.DataFrame(rows),pd.DataFrame(definitions),seizure_audit


__all__=["PHASES","build_feature_phenotype","rank_percentile"]
