"""Independent completed-run verifier; no test-model replay or threshold selection."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (f1_score,roc_auc_score,average_precision_score,
                             recall_score,accuracy_score)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path(r'C:\d2_no_patient_z_outer_seed42_runtime')
    baseline=Path(r'C:\d2_ictal_outer_seed42_runtime')
    freeze=json.loads((root/'FROZEN_BEFORE_OUTER.json').read_text())
    seal=json.loads((root/'OUTER_PREDICTION_SEAL.json').read_text())
    assert seal['binding']==sha(root/'FROZEN_BEFORE_OUTER.json')
    assert json.loads((root/'public/OUTER_STATUS.json').read_text())['status']=='ONE_FROZEN_NO_PATIENT_Z_OUTER_PASS_COMPLETE'
    with np.load(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz',allow_pickle=True) as f:
        y,patient,channel=f['y'],f['patient'],f['channel']
    table=pd.DataFrame(torch.load(root/'PATIENT_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False))
    baseline_seal=json.loads((baseline/'OUTER_PREDICTIONS_SEAL.json').read_text())
    max_drift=0.; checks=0; prechecks=0
    for fold in range(1,6):
        d=torch.load(root/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        original=torch.load(Path(r'C:\a0_source_posterior_decoder_seed42_runtime')/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        im=SimpleImputer().fit(np.asarray(original['raw_x'],dtype=np.float64)[original['train']])
        sc=StandardScaler().fit(im.transform(np.asarray(original['raw_x'],dtype=np.float64)[original['train']]))
        for key,value in [('imputer_statistics',im.statistics_),('mean',sc.mean_),('scale',sc.scale_),('var',sc.var_)]:
            assert np.array_equal(d['pre'][key],value)
        assert np.array_equal(d['x'],sc.transform(im.transform(original['raw_x'])).astype(np.float32))
        ckpath=root/f'fold{fold}/NO_PR/BEST_PRIVATE.pt'
        assert sha(ckpath)==freeze['folds'][fold-1]['checkpoint_sha256']
        ck=torch.load(ckpath,map_location='cpu',weights_only=False)
        assert ck['binding']['device']=='cuda' and ck['binding']['penalty']==.001
        assert ck['binding']['max_epochs']==30 and ck['binding']['seed']==42+1009*fold
        assert ck['binding']['initial_hash']==d['initial_hash']
        assert sum(t.numel() for t in ck['model'].values())==9221
        prechecks+=1
        for arm,path,h in [('D2_NO_PATIENT_Z',root/f'fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt',seal['hashes'][str(fold)]),
                           ('D2_PR',baseline/f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt',baseline_seal['files'][str(fold)])]:
            assert sha(path)==h
            cell=torch.load(path,map_location='cpu',weights_only=False)
            ix=cell['indices']; score=cell['scores'].astype(float)
            for pat in sorted(set(patient[ix])):
                k=np.flatnonzero(patient[ix]==pat)
                k=k[np.argsort(channel[ix][k].astype(str),kind='stable')]
                a=y[ix[k]];s=score[k];pred=(s>=cell['threshold']).astype(int)
                assert len(set(a))==2
                r0=recall_score(a,pred,pos_label=0);r1=recall_score(a,pred,pos_label=1)
                ranked=np.argsort(s,kind='stable'); ez_positions=np.flatnonzero(a[ranked]==0)
                values={'macro_f1':f1_score(a,pred,labels=[0,1],average='macro',zero_division=0),
                    'ez_f1':f1_score(a,pred,pos_label=0,zero_division=0),
                    'nez_f1':f1_score(a,pred,pos_label=1,zero_division=0),
                    'balanced_accuracy':(r0+r1)/2,'sensitivity':r0,'specificity':r1,
                    'accuracy':accuracy_score(a,pred),'ez_auroc':roc_auc_score(1-a,1-s),
                    'ez_auprc':average_precision_score(1-a,1-s),
                    'ez_mrr':1/(ez_positions[0]+1),'top1_is_ez':float(a[ranked[0]]==0)}
                recorded=table[(table.patient==pat)&(table.arm==arm)].iloc[0]
                for key,value in values.items():
                    drift=abs(value-recorded[key]);max_drift=max(max_drift,drift)
                    assert drift<1e-12
                checks+=1
    assert checks==160 and prechecks==5
    aggregate=pd.read_csv(root/'public/OUTER_METRICS.csv')
    comparisons=pd.read_csv(root/'public/PAIRED_BOOTSTRAP.csv')
    keys=comparisons.metric.tolist()
    a=table[table.arm=='D2_NO_PATIENT_Z'].set_index('patient').sort_index()[keys].to_numpy()
    b=table[table.arm=='D2_PR'].set_index('patient').sort_index()[keys].to_numpy()
    ix=np.random.default_rng(42).integers(0,80,size=(10000,80))
    delta=a-b;draws=delta[ix].mean(1)
    for j,row in comparisons.iterrows():
        assert abs(delta[:,j].mean()-row.delta)<1e-12
        assert abs(np.percentile(draws[:,j],2.5)-row.ci_low)<1e-12
        assert abs(np.percentile(draws[:,j],97.5)-row.ci_high)<1e-12
    for arm,values in [('D2_NO_PATIENT_Z',a),('D2_PR',b)]:
        boot=values[ix].mean(1)
        for j,key in enumerate(keys):
            row=aggregate[(aggregate.arm==arm)&(aggregate.metric==key)].iloc[0]
            assert abs(values[:,j].mean()-row.estimate)<1e-12
            assert abs(np.percentile(boot[:,j],2.5)-row.ci_low)<1e-12
            assert abs(np.percentile(boot[:,j],97.5)-row.ci_high)<1e-12
    output=root/'public/INDEPENDENT_VERIFICATION.json';assert not output.exists()
    output.write_text(json.dumps({'status':'PASS','model_patient_cells':checks,'metrics_per_cell':11,
        'maximum_metric_absolute_drift':max_drift,'FIT_only_scaler_parity_folds':prechecks,
        'CUDA_training_contract_preserved':True,'all_initial_state_hashes_match_original':True,
        'parameter_count':9221,'bootstrap_draws':10000,'bootstrap_seed':42,
        'paired_and_absolute_intervals_independently_verified':True,
        'no_test_model_replay':True,'no_threshold_reoptimization':True,
        'source_sha256':sha(__file__)},indent=2),encoding='utf-8')
    print('INDEPENDENT_NO_PATIENT_Z_VERIFICATION_PASS',max_drift)


if __name__=='__main__':main()
