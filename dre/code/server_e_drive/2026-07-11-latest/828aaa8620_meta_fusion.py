from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

META_FEATURES=["logit_fragility","logit_cii","logit_recruitment","logit_spectral","logit_nez","capture_consensus","outside_residual_consensus","causal_spread_coupling"]


def stable_logit(p):
    p=np.clip(np.asarray(p,float),1e-4,1-1e-4); return np.log(p/(1-p))


def evidence_bottleneck(expert_probabilities: pd.DataFrame,evidence: pd.DataFrame,fit_reference: pd.DataFrame|None=None):
    wide=expert_probabilities.pivot(index="patient_key",columns="expert",values="probability_success")
    out=pd.DataFrame(index=wide.index)
    for expert in ("fragility","cii","recruitment","spectral","nez"):
        probability=wide[expert] if expert in wide else pd.Series(.5,index=wide.index)
        out[f"logit_{expert}"]=stable_logit(probability)
    ev=evidence.set_index("patient_key").loc[out.index]
    capture=[c for c in ev if c.startswith("capture_")][:4]; residual=[c for c in ev if c.startswith("outside_residual_")][:4]
    if len(capture)!=4 or len(residual)!=4: raise ValueError("evidence table requires four capture and four outside residual summaries")
    out["capture_consensus"]=np.nanmedian(ev[capture],axis=1)
    residual_values=ev[residual].to_numpy(float)
    out["outside_residual_consensus"]=[np.sort(row[np.isfinite(row)])[-2] if np.isfinite(row).sum()>=2 else (row[np.isfinite(row)][0] if np.isfinite(row).sum()==1 else np.nan) for row in residual_values]
    ref=fit_reference.set_index("patient_key") if fit_reference is not None else ev
    coupling=[]
    for name in ("positive_net_escape","outside_recruited_3s_fraction"):
        med=np.nanmedian(ref[name]); sd=np.nanstd(ref[name]); coupling.append((ev[name]-med)/(sd if sd>1e-12 else 1.0))
    out["causal_spread_coupling"]=(coupling[0]+coupling[1])/2
    return out.reset_index()


@dataclass
class MetaFusion:
    medians: np.ndarray
    scaler: object
    model: object
    fit_patient_keys: tuple[str,...]
    def predict(self,frame):
        x=frame[META_FEATURES].to_numpy(float); x=np.where(np.isfinite(x),x,self.medians); x=self.scaler.transform(x); return self.model.predict_proba(x)[:,1]


def fit_meta(frame: pd.DataFrame,seed=42,forbidden_patients=()):
    if set(frame.patient_key.astype(str))&set(map(str,forbidden_patients)): raise ValueError("LEAKAGE: outer-test patient in meta training")
    x=frame[META_FEATURES].to_numpy(float); medians=np.asarray([np.nanmedian(x[:,i]) if np.isfinite(x[:,i]).any() else 0.0 for i in range(x.shape[1])]); x=np.where(np.isfinite(x),x,medians); scaler=StandardScaler().fit(x); x=scaler.transform(x)
    model=LogisticRegression(penalty="l2",C=.1,solver="liblinear",class_weight="balanced",max_iter=5000,random_state=seed).fit(x,frame.outcome_true.astype(int))
    return MetaFusion(medians,scaler,model,tuple(frame.patient_key.astype(str)))


__all__=["META_FEATURES","stable_logit","evidence_bottleneck","MetaFusion","fit_meta"]
