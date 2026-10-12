"""Independent sklearn score audit; no model replay, threshold tuning or training."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score,accuracy_score,recall_score,average_precision_score,roc_auc_score
from ablation_core import ARMS, NEW_ARMS, KEYS, mask_for


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime');pub=root/'public'
    full=Path(r'C:\d2_ictal_outer_seed42_runtime');noz=Path(r'C:\d2_no_patient_z_outer_seed42_runtime')
    original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime')
    assert json.loads((pub/'RUN_STATUS.json').read_text())['status']=='SEVEN_ARM_FROZEN_OUTER_COMPLETE'
    freeze=json.loads((root/'FROZEN_BEFORE_OUTER.json').read_text());seal=json.loads((root/'OUTER_PREDICTIONS_SEAL.json').read_text())
    assert seal['binding']==sha(root/'FROZEN_BEFORE_OUTER.json') and len(seal['files'])==35
    manifest=json.loads((pub/'FEATURE_GROUP_MANIFEST.json').read_text());columns=manifest['columns']
    assert sha(pub/'FEATURE_GROUP_MANIFEST.json')==freeze['feature_manifest_sha256']
    with np.load(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz',allow_pickle=True) as f:
        y,patient,channel=f['y'],f['patient'],f['channel']
    saved=pd.DataFrame(torch.load(root/'PATIENT_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False))
    independent=[];max_drift=0.;mask_checks=0;epochs=0
    for rec in freeze['selections']:
        arm,fold=rec['arm'],rec['fold'];mask=mask_for(columns,arm)
        assert np.array_equal(mask,np.asarray(rec['mask'],dtype=bool))
        if arm=='Full D2':bp=original/f'fold{fold}/BANK_PRIVATE.pt';cp=original/f'fold{fold}/D2/BEST_PRIVATE.pt';pp=full/f'fold{fold}_OUTER_PREDICTIONS_PRIVATE.pt'
        elif arm=='w/o Patient Z':bp=noz/f'fold{fold}/BANK_PRIVATE.pt';cp=noz/f'fold{fold}/NO_PR/BEST_PRIVATE.pt';pp=noz/f'fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt'
        else:bp=root/KEYS[arm]/f'fold{fold}/BANK_PRIVATE.pt';cp=bp.parent/'TRAIN/BEST_PRIVATE.pt';pp=bp.parent/'OUTER_PREDICTIONS_PRIVATE.pt'
        assert sha(bp)==rec['bank_sha256'] and sha(cp)==rec['checkpoint_sha256'] and sha(pp)==seal['files'][KEYS[arm]+'_'+str(fold)]
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        assert ck['epoch']==rec['epoch'] and ck['threshold']==rec['nez_threshold']
        assert ck['binding']['seed']==42+1009*fold and ck['binding']['max_epochs']==30 and ck['binding']['device']=='cuda'
        assert sum(v.numel() for v in ck['model'].values())==rec['parameters']
        assert ck['binding']['penalty']==(0. if arm=='w/o Source' else .001)
        if arm=='w/o Source':assert all(k.startswith('network.') for k in ck['model']) and len(ck['model'])==6
        if arm in NEW_ARMS:
            d=torch.load(bp,map_location='cpu',weights_only=False)
            old=torch.load(original/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
            assert np.all(d['x'][:,~mask]==0) and np.array_equal(d['x'][:,mask],old['x'][:,mask])
            for key in ['mean','scale','var','imputer_statistics']:assert np.array_equal(d['pre'][key],old['pre'][key])
            assert ck['binding']['initial_hash']==d['initial_hash']
            done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text())
            lastpath=cp.parent/'LAST_PRIVATE.pt';assert sha(lastpath)==done['last_sha256'] and sha(cp)==done['best_sha256']
            last=torch.load(lastpath,map_location='cpu',weights_only=False);epochs+=last['epoch'];mask_checks+=1
        cell=torch.load(pp,map_location='cpu',weights_only=False);ix=cell['indices'];scores=cell['scores'].astype(float)
        assert cell['threshold']==ck['threshold']
        for pat in sorted(set(patient[ix])):
            k=np.flatnonzero(patient[ix]==pat);k=k[np.argsort(channel[ix][k].astype(str),kind='stable')]
            a,s=y[ix[k]],scores[k];assert len(set(a))==2;pred=(s>=ck['threshold']).astype(int)
            r0=recall_score(a,pred,pos_label=0);r1=recall_score(a,pred,pos_label=1)
            order=np.argsort(s,kind='stable');ez=np.flatnonzero(a[order]==0)
            values={'macro_f1':f1_score(a,pred,labels=[0,1],average='macro',zero_division=0),
                'ez_f1':f1_score(a,pred,pos_label=0,zero_division=0),'nez_f1':f1_score(a,pred,pos_label=1,zero_division=0),
                'balanced_accuracy':(r0+r1)/2,'sensitivity':r0,'specificity':r1,'accuracy':accuracy_score(a,pred),
                'ez_auprc':average_precision_score(1-a,1-s),'ez_auroc':roc_auc_score(1-a,1-s),
                'ez_mrr':float(1/(ez[0]+1)),'top1_is_ez':float(a[order[0]]==0)}
            row=saved[(saved.arm==arm)&(saved.patient==pat)].iloc[0]
            for key,value in values.items():
                drift=abs(value-row[key]);max_drift=max(max_drift,drift);assert drift<1e-12
            independent.append({'arm':arm,'patient':pat,**values})
    df=pd.DataFrame(independent);assert len(df)==560 and df.patient.nunique()==80
    table=pd.read_csv(pub/'ABLATION_SUMMARY.csv');boot=pd.read_csv(pub/'PAIRED_BOOTSTRAP.csv')
    keys=[k for k in table.columns if k in independent[0] and k not in ['arm','patient']]
    f=df[df.arm=='Full D2'].set_index('patient').sort_index()[keys].to_numpy()
    ix=np.random.default_rng(42).integers(0,80,size=(10000,80))
    for arm in ARMS:
        part=df[df.arm==arm].set_index('patient').sort_index()[keys].to_numpy();assert part.shape==(80,11)
        result=table[table.arm==arm].iloc[0]
        for j,key in enumerate(keys):assert abs(part[:,j].mean()-result[key])<1e-12
        if arm=='Full D2':continue
        delta=f-part;draws=delta[ix].mean(1)
        for j,key in enumerate(keys):
            row=boot[(boot.ablated_arm==arm)&(boot.metric==key)].iloc[0]
            assert abs(delta[:,j].mean()-row.delta)<1e-12
            assert abs(np.percentile(draws[:,j],2.5)-row.ci_low)<1e-12 and abs(np.percentile(draws[:,j],97.5)-row.ci_high)<1e-12
    output=pub/'INDEPENDENT_VERIFICATION.json';assert not output.exists()
    output.write_text(json.dumps({'status':'PASS','model_patient_cells':560,'metrics_per_cell':11,
        'maximum_metric_absolute_drift':max_drift,'checkpoint_bank_prediction_bindings_checked':35,
        'actual_FIT_VAL_masks_checked':mask_checks,'new_training_epochs':epochs,
        'all_seven_aggregate_rows_independently_verified':True,'paired_bootstrap_metrics':66,'bootstrap_draws':10000,
        'bootstrap_direction':'Full minus ablated','no_model_replay':True,'no_threshold_reoptimization':True,
        'source_sha256':sha(__file__)},indent=2),encoding='utf-8')
    print('INDEPENDENT_SEVEN_ARM_VERIFICATION_PASS',max_drift,epochs,flush=True)


if __name__=='__main__':main()
