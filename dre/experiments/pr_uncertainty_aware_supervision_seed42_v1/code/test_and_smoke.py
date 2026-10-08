"""Source parity and genuine FIT-only smoke, no outer/validation labels queried."""
import argparse
import copy
import json
import subprocess
import sys
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from uas_core import (PRMLP, prepare, seed_all, train_model, state_hash, sha, json_write,
                      soft_target, select_threshold)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True); p.add_argument('--tests',type=Path,required=True)
    a=p.parse_args(); torch.set_num_threads(2)
    environment=dict(os.environ,PR_UAS_SYNTHETIC_REFERENCE_FIXTURE='1')
    result=subprocess.run([sys.executable,'-m','pytest',str(a.tests),'-q'],capture_output=True,text=True,env=environment)
    print(result.stdout,flush=True)
    if result.returncode: print(result.stderr); raise RuntimeError('Synthetic tests failed')
    gate=json.loads((a.runtime/'gate'/'A0_REPRODUCTION.json').read_text())
    assert gate['status']=='PASS'
    admission=json.loads((a.runtime/'RUNTIME_ADMISSION.json').read_text())
    assert admission['status']=='PASS' and admission['numpy']==np.__version__ and admission['torch']==torch.__version__
    assert all(sha(a.source/n)==v for n,v in gate['source_hashes'].items())
    assert sha(a.runtime/'gate'/'FEATURES_PRIVATE.npz')==gate['private_features_sha256']
    with np.load(a.runtime/'gate'/'FEATURES_PRIVATE.npz',allow_pickle=True) as z:
        fit=set(z['split_patient'][(z['split_fold']==1)&(z['split_role']=='fit')])
        take=np.isin(z['patient'],list(fit))
        x,y,pid=z['x'][take],z['y'][take],z['patient'][take]
        features=z['features']
    assert len(set(pid))==51 and x.shape[1]==88
    assert len(set(features))==88 and not any(n in features for n in ['label_nez','center','clinical_true_ez'])
    ids=sorted(set(pid))[:3]; smoke=np.flatnonzero(np.isin(pid,ids))
    x,y,pid=x[smoke],y[smoke],pid[smoke]
    tr=np.flatnonzero(np.isin(pid,ids[:2])); va=np.flatnonzero(pid==ids[2])
    ours,pre=prepare(x,pid,tr)
    sys.path.insert(0,str(a.source))
    from task1_baselines.patient_controls import _ChannelMLP, _Preprocessor, _fit_torch
    import task1_baselines.patient_controls as controls
    from task1_baselines.thresholds import Task1Threshold, _patient_metrics
    # Substitute only the independently fixture-validated mathematical threshold
    # evaluator in this smoke. Original model/BCE/optimizer/data order stay untouched.
    def source_select(table, *, source='inner_oof_patient_macro_f1'):
        s=select_threshold(table.label_nez.to_numpy(),table.score_nez_probability.to_numpy(),table.subject_id.to_numpy())
        return Task1Threshold(s['threshold'],s['macro_f1'],s['ez_f1'],s['pooled_ba'],source)
    controls.select_patient_macro_threshold=source_select
    frame=pd.DataFrame(x,columns=list(features)); frame['subject_id']=pid; frame['label_nez']=y
    original=_Preprocessor(SimpleImputer(),StandardScaler(),True)
    ox=original.fit_transform(frame.iloc[tr],list(features)); ov=original.transform(frame.iloc[va],list(features))
    assert np.array_equal(ox,ours[tr]) and np.array_equal(ov,ours[va])
    seed_all(1051); initial=copy.deepcopy(PRMLP().state_dict())
    om=_ChannelMLP(88); om.load_state_dict(initial)
    old,threshold,epoch,history=_fit_torch(om,ox,y[tr],pid[tr],frame.iloc[va].reset_index(drop=True),ov,
        device=torch.device('cuda'),seed=1051,max_epochs=1,patience=6,checkpoint_path=None,rank_scores=False)
    new,best,h=train_model(ours,y,pid,tr,va,a.runtime/'smoke'/'source_parity',1051,initial,
        sha(Path(__file__).parent/'uas_core.py'),'cuda',max_epochs=1)
    assert state_hash(old.state_dict())==state_hash(new.state_dict())
    assert best['threshold']==threshold.threshold and epoch==best['epoch']
    with torch.no_grad():
        score=torch.sigmoid(old(torch.as_tensor(ov,device='cuda'))).cpu().numpy()
    frameval=frame.iloc[va][['subject_id','label_nez']].copy(); frameval['score_nez_probability']=score
    fast=select_threshold(y[va],score,pid[va])
    reference_at_selected=_patient_metrics(frameval,fast['threshold'])
    assert np.allclose(reference_at_selected,[fast['macro_f1'],fast['ez_f1'],fast['pooled_ba']],rtol=0,atol=1e-15)
    # Real FIT-only A2 backward: fixed hypothetical OOF values, never reused as teachers.
    model=PRMLP().cuda(); ytorch=torch.as_tensor(y[tr],dtype=torch.float32,device='cuda')
    target=soft_target(ytorch,torch.full_like(ytorch,.5),torch.ones_like(ytorch),10)
    loss=torch.nn.functional.binary_cross_entropy_with_logits(model(torch.as_tensor(ours[tr],device='cuda')),target)
    loss.backward(); assert all(torch.isfinite(p.grad).all() for p in model.parameters())
    audit={'status':'PASS','synthetic_pytest_stdout':result.stdout.strip(),
        'synthetic_tests_sha256':sha(a.tests/'test_uas.py'),'core_sha256':sha(Path(__file__).parent/'uas_core.py'),
        'source_source_parity_model_hash':state_hash(new.state_dict()),'real_FIT_patients_smoke':3,
        'preprocessing_array_max_drift':0.,'single_epoch_model_parameter_max_drift':0.,
        'source_threshold_selected_metric_exact_match':True,'finite_real_A2_gradients':True,
        'threshold_reference_mode':'identical eight synthetic cases, independently locally computed sklearn oracle; selected real FIT threshold metrics checked with original source',
        'synthetic_oracle_sha256':sha(a.tests/'SYNTHETIC_THRESHOLD_ORACLE.json'),
        'outer_VAL_TEST_labels_used':False,'features_labels_identity_gate':True,
        'versions':{'numpy':np.__version__,'torch':torch.__version__},
        'source_hashes':gate['source_hashes'],'runtime_admission_sha256':sha(a.runtime/'RUNTIME_ADMISSION.json')}
    json_write(a.runtime/'TEST_AND_SMOKE_AUDIT.json',audit)
    print('TEST_AND_REAL_FIT_SMOKE_PASS',flush=True)


if __name__=='__main__': main()
