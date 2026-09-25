from __future__ import annotations
import pandas as pd,numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
def development_split(keys,meta,fold,seed,fraction=.2,existing=None):
    existing=pd.DataFrame() if existing is None else existing;old=existing[(existing.outer_fold==fold)&existing.partition.isin(['development_train','development_validation'])] if not existing.empty else pd.DataFrame()
    if not old.empty and set(old.patient_key)==set(keys):return old.loc[old.partition=='development_train','patient_key'].tolist(),old.loc[old.partition=='development_validation','patient_key'].tolist()
    labels=np.array([meta[k]['outcome_success'] for k in keys]);joint=np.array([f"{meta[k]['center']}|{meta[k]['outcome_success']}" for k in keys]);strata=joint if min((joint==x).sum() for x in set(joint))>=2 else labels
    try:train,val=next(StratifiedShuffleSplit(1,test_size=fraction,random_state=seed+int(fold)).split(np.zeros(len(keys)),strata))
    except ValueError:train,val=next(StratifiedShuffleSplit(1,test_size=fraction,random_state=seed+int(fold)).split(np.zeros(len(keys)),labels))
    return [keys[i] for i in train],[keys[i] for i in val]
def assert_isolated(train,val,test):
    if set(train)&set(val) or set(train)&set(test) or set(val)&set(test):raise RuntimeError('patient split leakage')
