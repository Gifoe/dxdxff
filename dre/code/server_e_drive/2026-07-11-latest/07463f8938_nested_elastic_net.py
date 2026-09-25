from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score,brier_score_loss,f1_score,roc_auc_score
from sklearn.model_selection import StratifiedKFold

from .preprocessing import COPPreprocessor


C_GRID=(.01,.03,.1,.3,1.,3.)
L1_GRID=(.1,.5,.9)


def deterministic_inner_splits(y:np.ndarray,centers:Sequence[str],folds:int,seed:int)->list[tuple[np.ndarray,np.ndarray]]:
    joint=np.asarray([f"{int(label)}|{center}" for label,center in zip(y,centers)]);counts=pd.Series(joint).value_counts();strata=joint if len(counts) and counts.min()>=folds else y
    splitter=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed);return list(splitter.split(np.zeros(len(y)),strata))


def _metrics(y:np.ndarray,p:np.ndarray,threshold:float=.5)->dict[str,float]:
    pred=(p>=threshold).astype(int);return {"auroc":float(roc_auc_score(y,p)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),"brier":float(brier_score_loss(y,p))}


def select_inner_threshold(y:np.ndarray,p:np.ndarray)->float:
    candidates=[]
    for threshold in np.round(np.arange(.20,.801,.01),2):
        pred=(p>=threshold).astype(int);candidates.append((float(balanced_accuracy_score(y,pred)),float(f1_score(y,pred,average="macro",zero_division=0)),-abs(float(threshold)-.5),-float(threshold),float(threshold)))
    return max(candidates)[-1]


@dataclass
class InnerSearchResult:
    C:float
    l1_ratio:float
    threshold:float
    probability:np.ndarray
    metrics:dict[str,float]
    rows:list[dict[str,Any]]


def inner_search(frame:pd.DataFrame,features:Sequence[str],y:np.ndarray,centers:Sequence[str],*,outer_fold:int,inner_folds:int,seed:int,quick:bool=False)->InnerSearchResult:
    splits=deterministic_inner_splits(y,centers,inner_folds,seed);prepared=[]
    for train_idx,test_idx in splits:
        prep=COPPreprocessor.fit(frame.iloc[train_idx],features,outer_fold);prepared.append((train_idx,test_idx,prep,prep.transform(frame.iloc[train_idx]),prep.transform(frame.iloc[test_idx])))
    grid=((.1,.5),(.3,.5), (1.,.5)) if quick else tuple((c,l1) for c in C_GRID for l1 in L1_GRID);results=[]
    for C,l1 in grid:
        probability=np.full(len(y),np.nan)
        for train_idx,test_idx,_,x_train,x_test in prepared:
            model=LogisticRegression(penalty="elasticnet",solver="saga",C=C,l1_ratio=l1,class_weight=None,max_iter=20000,random_state=seed,tol=1e-5)
            model.fit(x_train,y[train_idx]);probability[test_idx]=model.predict_proba(x_test)[:,1]
        metric=_metrics(y,probability);results.append({"C":C,"l1_ratio":l1,"probability":probability,**metric})
    best=max(results,key=lambda x:(x["auroc"],x["balanced_accuracy"],-x["brier"],-x["C"],x["l1_ratio"]));threshold=select_inner_threshold(y,best["probability"])
    rows=[{"outer_fold":outer_fold,"C":x["C"],"l1_ratio":x["l1_ratio"],"inner_auroc":x["auroc"],"inner_balanced_accuracy":x["balanced_accuracy"],"inner_macro_f1":x["macro_f1"],"inner_brier":x["brier"],"selected":bool(x is best)} for x in results]
    return InnerSearchResult(float(best["C"]),float(best["l1_ratio"]),threshold,best["probability"],{k:float(best[k]) for k in ("auroc","balanced_accuracy","macro_f1","brier")},rows)


@dataclass
class FittedCOPModel:
    preprocessor:COPPreprocessor
    model:LogisticRegression
    inner:InnerSearchResult
    groups:tuple[str,...]

    def predict_probability(self,frame:pd.DataFrame)->np.ndarray:return self.model.predict_proba(self.preprocessor.transform(frame))[:,1]


def fit_outer_model(frame:pd.DataFrame,features:Sequence[str],y:np.ndarray,centers:Sequence[str],groups:Sequence[str],*,outer_fold:int,inner_folds:int,seed:int,quick:bool=False)->FittedCOPModel:
    inner=inner_search(frame,features,y,centers,outer_fold=outer_fold,inner_folds=inner_folds,seed=seed,quick=quick);prep=COPPreprocessor.fit(frame,features,outer_fold);x=prep.transform(frame)
    model=LogisticRegression(penalty="elasticnet",solver="saga",C=inner.C,l1_ratio=inner.l1_ratio,class_weight=None,max_iter=20000,random_state=seed,tol=1e-5).fit(x,y)
    return FittedCOPModel(prep,model,inner,tuple(groups))


def choose_groups(frame:pd.DataFrame,group_features:dict[str,list[str]],y:np.ndarray,centers:Sequence[str],*,outer_fold:int,inner_folds:int,seed:int,quick:bool=False)->tuple[tuple[str,...],list[dict[str,Any]],list[dict[str,Any]]]:
    selected=["feature"];base=inner_search(frame,group_features["feature"],y,centers,outer_fold=outer_fold,inner_folds=inner_folds,seed=seed,quick=quick);all_rows=[dict(row,candidate_groups="feature") for row in base.rows];audit=[]
    for group,threshold in (("raw",.015),("p2",.010)):
        candidate=selected+[group];features=[name for value in candidate for name in group_features[value]];result=inner_search(frame,features,y,centers,outer_fold=outer_fold,inner_folds=inner_folds,seed=seed,quick=quick);all_rows.extend(dict(row,candidate_groups="+".join(candidate)) for row in result.rows)
        gain_auc=result.metrics["auroc"]-base.metrics["auroc"];gain_bal=result.metrics["balanced_accuracy"]-base.metrics["balanced_accuracy"];accepted=gain_auc>=threshold or gain_bal>=threshold
        audit.append({"outer_fold":outer_fold,"base_groups":"+".join(selected),"candidate_group":group,"inner_auroc_before":base.metrics["auroc"],"inner_auroc_after":result.metrics["auroc"],"inner_balanced_accuracy_before":base.metrics["balanced_accuracy"],"inner_balanced_accuracy_after":result.metrics["balanced_accuracy"],"selected":accepted,"selection_rule":f"AUROC gain >= {threshold:.3f} OR balanced accuracy gain >= {threshold:.3f}"})
        if accepted:selected.append(group);base=result
    return tuple(selected),audit,all_rows


__all__=["C_GRID","L1_GRID","FittedCOPModel","choose_groups","deterministic_inner_splits","fit_outer_model","inner_search","select_inner_threshold"]
