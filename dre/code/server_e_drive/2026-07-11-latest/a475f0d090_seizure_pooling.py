from __future__ import annotations

import numpy as np
from .quantile_features import pool_named_rows


def top10_set(values):
    x=np.asarray(values,float); valid=np.where(np.isfinite(x))[0]
    if not valid.size:return set()
    n=max(1,int(np.ceil(.10*valid.size))); return set(valid[np.argsort(x[valid],kind="stable")[-n:]].tolist())


def mean_pairwise_jaccard(channel_maps):
    sets=[top10_set(x) for x in channel_maps]
    if len(sets)<2:return np.nan
    scores=[]
    for i in range(len(sets)):
        for j in range(i+1,len(sets)):
            union=sets[i]|sets[j]; scores.append(len(sets[i]&sets[j])/len(union) if union else np.nan)
    return float(np.nanmean(scores)) if np.isfinite(scores).any() else np.nan


def pool_seizures(rows, channel_maps=None):
    result=pool_named_rows(rows)
    if channel_maps is not None: result["top10_recurrence_mean_jaccard"]=mean_pairwise_jaccard(channel_maps)
    return result


__all__=["top10_set","mean_pairwise_jaccard","pool_seizures"]
