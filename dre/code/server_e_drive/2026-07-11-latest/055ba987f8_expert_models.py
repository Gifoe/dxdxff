from __future__ import annotations

import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

EXPERTS=("fragility","cii","recruitment","spectral","nez")


def make_expert_model(expert,n_train,seed=42,n_jobs=-1):
    if expert=="fragility":
        return RandomForestClassifier(n_estimators=500,max_depth=3,min_samples_leaf=max(2,math.ceil(.05*n_train)),max_features="sqrt",class_weight="balanced_subsample",bootstrap=True,random_state=seed,n_jobs=n_jobs)
    return LogisticRegression(penalty="l2",C=.1,solver="liblinear",class_weight="balanced",max_iter=5000,random_state=seed)


def expert_uses_pca(expert): return expert in {"cii","spectral"}


__all__=["EXPERTS","make_expert_model","expert_uses_pca"]
