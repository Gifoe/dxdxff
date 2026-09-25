from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd

from .expert_models import EXPERTS, make_expert_model, expert_uses_pca
from .expert_preprocessing import ExpertPreprocessor


@dataclass
class FittedExpert:
    expert: str
    features: list[str]
    preprocessor: ExpertPreprocessor
    model: object

    @property
    def fit_patient_keys(self): return self.preprocessor.fit_patient_keys

    def predict(self,frame): return self.model.predict_proba(self.preprocessor.transform(frame[self.features]))[:,1]


def _feature_names(frame): return [x for x in frame.columns if x not in {"patient_key","center","outcome_true","outer_fold"}]


def fit_expert(expert,frame,seed=42,n_jobs=-1,forbidden_patients=()):
    if set(map(str,frame.patient_key)) & set(map(str,forbidden_patients)): raise ValueError("LEAKAGE: forbidden patient in expert fit partition")
    features=_feature_names(frame); prep=ExpertPreprocessor(use_pca=expert_uses_pca(expert)).fit(frame[features],frame.patient_key)
    model=make_expert_model(expert,len(frame),seed,n_jobs).fit(prep.transform(frame[features]),frame.outcome_true.astype(int))
    return FittedExpert(expert,features,prep,model)


def lopo_crossfit_experts(expert_frames: dict[str,pd.DataFrame], train_patients, outer_fold:int, seed=42,n_jobs=-1, experts=EXPERTS):
    patients=sorted(map(str,train_patients)); rows=[]
    for expert in experts:
        frame=expert_frames[expert].set_index("patient_key",drop=False)
        if set(patients)-set(frame.index.astype(str)): raise ValueError(f"{expert}: missing training patients")
        for patient in patients:
            inner=[p for p in patients if p!=patient]; fit=fit_expert(expert,frame.loc[inner].reset_index(drop=True),seed+outer_fold,n_jobs,forbidden_patients=[patient])
            probability=float(fit.predict(frame.loc[[patient]])[0])
            rows.append({"patient_key":patient,"outer_fold":outer_fold,"expert":expert,"probability_success":probability,"true_label":int(frame.loc[patient,"outcome_true"]),"fit_patient_keys":";".join(inner),"held_out_verified":True})
    return pd.DataFrame(rows)


def fit_outer_experts(expert_frames,train_patients,test_patients,outer_fold,seed=42,n_jobs=-1,experts=EXPERTS):
    predictions=[]; fitted={}
    for expert in experts:
        frame=expert_frames[expert].set_index("patient_key",drop=False); train=frame.loc[sorted(train_patients)].reset_index(drop=True); test=frame.loc[sorted(test_patients)]
        fit=fit_expert(expert,train,seed+outer_fold,n_jobs,forbidden_patients=test_patients); fitted[expert]=fit
        for patient,prob in zip(test.patient_key,fit.predict(test)):
            predictions.append({"patient_key":patient,"outer_fold":outer_fold,"expert":expert,"probability_success":float(prob),"true_label":int(test.loc[patient,"outcome_true"]),"fit_patient_keys":";".join(sorted(train_patients)),"outer_test_excluded":True})
    return pd.DataFrame(predictions),fitted


__all__=["FittedExpert","fit_expert","lopo_crossfit_experts","fit_outer_experts"]
