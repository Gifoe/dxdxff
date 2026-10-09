"""Post-training engineering regression; no real model or data is trained."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'code'))
from base import patient_metrics
import importlib.util
spec=importlib.util.spec_from_file_location('tabm_audit_finalizer',Path(__file__).resolve().parents[1]/'code'/'finalize.py')
module=importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
members=module.members


def test_member_metadata_concat_preserves_metrics(tmp_path):
    rng=np.random.default_rng(42)
    pid=np.repeat(['synthetic_a','synthetic_b'],8)
    y=np.tile([0,1],8); center=np.repeat(['synthetic_center'],16)
    probabilities=rng.uniform(.1,.9,size=(16,4)); score=probabilities.mean(1)
    c=pd.DataFrame({'patient':pid,'y_nez':y,'center':center,
        'score_nez':score,'threshold':.5,'fold':1,'arm':'W2'})
    for k in range(4):c[f'member{k}_p_nez']=probabilities[:,k]
    pm=patient_metrics(y,score,pid,center,.5)
    pm['fold']=1; pm['arm']='W2'
    members(c,pm,tmp_path)
    actual=pd.read_csv(tmp_path/'MEMBER_DIVERSITY_SUMMARY.csv')
    actual=actual[(actual.scope=='overall')&(actual.threshold_mode=='parent_ensemble_threshold')].set_index('member')
    assert len(actual)==5
    for k in range(4):
        expected=patient_metrics(y,probabilities[:,k],pid,center,.5)
        for metric in ['macro_f1','ez_auprc']:
            assert abs(actual.loc[str(k),metric]-expected[metric].mean())<1e-12
    assert abs(actual.loc['ensemble','macro_f1']-pm.macro_f1.mean())<1e-12
    diversity=pd.read_csv(tmp_path/'MEMBER_PAIR_DIVERSITY.csv')
    row=diversity[diversity.scope=='overall'].iloc[0]
    assert abs(row.member_probability_variance-probabilities.var(axis=1).mean())<1e-12
    comparison=pd.read_csv(tmp_path/'ENSEMBLE_VS_BEST_MEMBER_DIAGNOSTIC.csv')
    assert not comparison.deployed_member_selection.any()
