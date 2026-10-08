"""Admit replacement runtime using this task's freshly hash-verified 88D cache."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from uas_core import sha, json_write, patient_metrics, METRICS


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True); a=p.parse_args()
    torch.set_num_threads(2); root=a.runtime
    gate=json.loads((root/'gate'/'A0_REPRODUCTION.json').read_text())
    assert gate['status']=='PASS' and sha(root/'gate'/'FEATURES_PRIVATE.npz')==gate['private_features_sha256']
    assert all(sha(a.source/n)==v for n,v in gate['source_hashes'].items())
    sys.path.insert(0,str(a.source))
    from task1_baselines.patient_controls import _ChannelMLP, _Preprocessor
    # Authentic private export contains pandas object-string identities; hash gate first.
    with np.load(root/'gate'/'FEATURES_PRIVATE.npz',allow_pickle=True) as z: arrays={k:z[k] for k in z.files}
    frame=pd.DataFrame(arrays['x'],columns=list(arrays['features']))
    frame['subject_id']=arrays['patient']; frame['channel_name']=arrays['channel']
    frame['label_nez']=arrays['y']; frame['center']=arrays['center']
    base=Path(r'D:\nips-temp\task1_patient_relative_controls_v2')
    ledger=base/'oof_ledgers'/'patient_z_mlp'/'seed_42_channel_oof.csv'
    assert sha(ledger)=='3bd2039cc7d01ecdb375256354978066b4a1b4fc5b8adda1a58b84b2abb9f896'
    old=pd.read_csv(ledger); drifts=[]; results=[]
    for fold in range(1,6):
        sf=arrays['split_fold']==fold
        fit=set(arrays['split_patient'][sf&(arrays['split_role']=='fit')])
        test=set(arrays['split_patient'][sf&(arrays['split_role']=='test')])
        tr=frame[frame.subject_id.isin(fit)]; te=frame[frame.subject_id.isin(test)]
        pre=_Preprocessor(SimpleImputer(),StandardScaler(),True)
        pre.fit_transform(tr,list(arrays['features'])); x=pre.transform(te,list(arrays['features']))
        path=base/'checkpoints'/'patient_z_mlp'/'seed_42'/f'fold_{fold}'/'best_model.pt'
        assert sha(path)==gate['folds'][fold-1]['checkpoint_sha256']
        saved=torch.load(path,map_location='cpu',weights_only=True)
        model=_ChannelMLP(88).eval(); model.load_state_dict(saved['state_dict'])
        with torch.no_grad(): score=torch.sigmoid(model(torch.from_numpy(x))).numpy()
        replay=te[['subject_id','channel_name']].copy(); replay['score']=score
        joined=replay.merge(old[old.outer_fold==fold],on=['subject_id','channel_name'],validate='one_to_one')
        drift=float(np.abs(joined.score-joined.score_nez_probability).max()); assert drift<2e-6
        assert np.array_equal((joined.score>=saved['selected_threshold']).astype(int),joined.predicted_nez)
        results.append(patient_metrics(te.label_nez.to_numpy(),score,te.subject_id.to_numpy(),te.center.to_numpy(),saved['selected_threshold']))
        drifts.append(drift)
    metrics=pd.concat(results)[METRICS].mean().to_dict()
    assert all(abs(metrics[k]-v)<1e-8 for k,v in gate['metrics'].items())
    json_write(root/'RUNTIME_ADMISSION.json',{'status':'PASS','source':'fresh_this_task_hash_verified_88D_features',
        'feature_sha256':gate['private_features_sha256'],'five_checkpoint_max_drifts':drifts,
        'historical_metrics_exact':True,'metrics':metrics,'numpy':np.__version__,'torch':torch.__version__,
        'source_code_sha256':sha(Path(__file__))})
    print('REPLACEMENT_RUNTIME_FROZEN_B0_REPLAY_PASS',flush=True)


if __name__=='__main__': main()
