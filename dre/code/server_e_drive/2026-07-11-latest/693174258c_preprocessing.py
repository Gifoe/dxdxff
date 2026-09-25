from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def feature_priority(name:str)->int:
    if name.startswith("raw__"):return 0
    if name.startswith("feature__") and ("recurrence" in name or name.endswith("_std")):return 1
    if name.startswith("feature__") and "minus" in name:return 2
    if name.startswith("feature__") and any(x in name for x in ("entropy","gini","fraction")):return 3
    if name.startswith("feature__"):return 4
    if name.startswith("p2__"):return 5
    return 6


@dataclass
class COPPreprocessor:
    input_features:list[str]
    medians:dict[str,float]
    indicator_sources:list[str]
    retained_features:list[str]
    scaler:StandardScaler
    audit:list[dict]

    @classmethod
    def fit(cls,frame:pd.DataFrame,features:Sequence[str],outer_fold:int)->"COPPreprocessor":
        input_features=list(features);medians={};indicator=[];work=pd.DataFrame(index=frame.index);audit=[]
        for name in input_features:
            value=pd.to_numeric(frame[name],errors="coerce").astype(float);finite=value[np.isfinite(value)]
            if finite.empty:
                audit.append({"outer_fold":outer_fold,"feature":name,"status":"removed","reason":"all_missing_outer_train","correlated_with":"","correlation":np.nan,"train_only_verified":True});continue
            medians[name]=float(finite.median());work[name]=value.fillna(medians[name])
            if value.isna().any():indicator.append(name);work[name+"__missing"]=value.isna().astype(float)
        candidates=[]
        for name in work:
            if float(work[name].std(ddof=0))<1e-12:audit.append({"outer_fold":outer_fold,"feature":name,"status":"removed","reason":"constant_outer_train","correlated_with":"","correlation":np.nan,"train_only_verified":True})
            else:candidates.append(name)
        ordered=sorted(candidates,key=lambda name:(feature_priority(name),input_features.index(name.replace("__missing","")) if name.replace("__missing","") in input_features else 10**9,name))
        retained=[]
        for name in ordered:
            conflict="";correlation=np.nan
            for keep in retained:
                value=float(np.corrcoef(work[name].to_numpy(),work[keep].to_numpy())[0,1])
                if np.isfinite(value) and abs(value)>.95:conflict=keep;correlation=value;break
            if conflict:audit.append({"outer_fold":outer_fold,"feature":name,"status":"removed","reason":"correlation_gt_0p95","correlated_with":conflict,"correlation":correlation,"train_only_verified":True})
            else:retained.append(name);audit.append({"outer_fold":outer_fold,"feature":name,"status":"retained","reason":"","correlated_with":"","correlation":np.nan,"train_only_verified":True})
        if not retained:raise ValueError("COP preprocessing removed every feature")
        scaler=StandardScaler().fit(work[retained].to_numpy(dtype=float));return cls(input_features,medians,indicator,retained,scaler,audit)

    def transform(self,frame:pd.DataFrame)->np.ndarray:
        work=pd.DataFrame(index=frame.index)
        for name in self.input_features:
            value=pd.to_numeric(frame[name],errors="coerce").astype(float)
            if name in self.medians:work[name]=value.fillna(self.medians[name])
            if name in self.indicator_sources:work[name+"__missing"]=value.isna().astype(float)
        return self.scaler.transform(work[self.retained_features].to_numpy(dtype=float))


__all__=["COPPreprocessor","feature_priority"]
