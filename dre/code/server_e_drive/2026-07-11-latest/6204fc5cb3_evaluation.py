from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

from ..metrics import bootstrap_metrics,compute_metrics


def metric_row(frame:pd.DataFrame,prediction_column:str,threshold_column:str|float,outer_fold:int)->dict:
    threshold=float(threshold_column) if isinstance(threshold_column,(float,int)) else float(frame[threshold_column].iloc[0]);prediction=(frame[prediction_column].to_numpy() if prediction_column in frame else (frame.probability_success.to_numpy()>=threshold).astype(int))
    return {"outer_fold":outer_fold,"threshold":threshold,**compute_metrics(frame.outcome_true,frame.probability_success,prediction_success=prediction)}


def center_metrics(predictions:pd.DataFrame,prediction_column:str)->pd.DataFrame:
    rows=[]
    for center,group in predictions.groupby("center"):rows.append({"center":center,**compute_metrics(group.outcome_true,group.probability_success,prediction_success=group[prediction_column])})
    return pd.DataFrame(rows)


def bootstrap_ci(predictions:pd.DataFrame,prediction_column:str,repeats:int,seed:int)->pd.DataFrame:
    samples,skipped=bootstrap_metrics(predictions.outcome_true,predictions.probability_success,prediction_success=predictions[prediction_column],repeats=repeats,seed=seed);rows=[]
    for name in ("accuracy","balanced_accuracy","macro_f1","weighted_f1","auroc","success_auprc","failure_auprc","success_precision","success_recall","failure_precision","failure_recall","brier","ece"):
        value=samples[name].dropna().to_numpy() if name in samples else np.asarray([]);rows.append({"metric":name,"mean":float(value.mean()) if value.size else np.nan,"ci_low":float(np.quantile(value,.025)) if value.size else np.nan,"ci_high":float(np.quantile(value,.975)) if value.size else np.nan,"valid_repeats":len(value),"skipped_single_class":skipped})
    return pd.DataFrame(rows)


def summarize_fold_metrics(folds:pd.DataFrame,predictions:pd.DataFrame,prediction_column:str)->pd.DataFrame:
    pooled={"scope":"pooled_oof",**compute_metrics(predictions.outcome_true,predictions.probability_success,prediction_success=predictions[prediction_column])};numeric=[c for c in folds if c not in ("outer_fold","threshold")]
    mean={"scope":"fold_mean",**{c:pd.to_numeric(folds[c],errors="coerce").mean() for c in numeric}};std={"scope":"fold_std",**{c:pd.to_numeric(folds[c],errors="coerce").std(ddof=1) for c in numeric}};return pd.DataFrame([pooled,mean,std])


__all__=["bootstrap_ci","center_metrics","metric_row","summarize_fold_metrics"]
