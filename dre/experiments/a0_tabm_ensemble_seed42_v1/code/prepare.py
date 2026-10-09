"""Verify source/splits and frozen A0; seal original FIT/VAL 88D tensors only."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from base import sha,json_write,torch_write,prepare,PRMLP,patient_metrics,METRICS


def binding(protocol):
    names=['base.py','tabm_layers.py','model.py','source_parity.py','prepare.py','train.py','run.py','smoke.py']
    return {**{n:sha(Path(__file__).parent/n) for n in names},'protocol':sha(protocol)}


def main():
    p=argparse.ArgumentParser()
    for n in ['runtime','protocol','a0-runtime','split']:p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args(); root=a.runtime; root.mkdir(exist_ok=True,parents=True); pub=root/'public'
    torch.set_num_threads(2)
    lock=json.loads(a.protocol.read_text()); bind=binding(a.protocol)
    assert sha(a.split)==lock['split_sha256']
    export=a.a0_runtime/'gate/FEATURES_PRIVATE.npz'
    assert sha(export)==lock['private_feature_sha256']
    with np.load(export,allow_pickle=True) as f:d={k:f[k] for k in f.files}
    assert d['x'].shape==(7635,88) and len(set(d['patient']))==80
    assert len(set(zip(d['patient'],d['channel'])))==7635 and len(set(d['features']))==88
    selections=json.loads((a.a0_runtime/'public/SELECTION_AUDIT.json').read_text())['selections']
    original_manifest=pd.read_csv(a.split)
    assert len(original_manifest)>0
    records=[]; checks=[]; patients=[]
    for fold in range(1,6):
        s=d['split_fold']==fold
        ids={r:set(d['split_patient'][s&(d['split_role']==r)].astype(str)) for r in ['fit','validation','test']}
        assert set.union(*ids.values())==set(d['patient'].astype(str))
        assert all(not ids[r]&ids[t] for r,t in [('fit','validation'),('fit','test'),('validation','test')])
        ix=np.flatnonzero(np.isin(d['patient'],list(ids['fit']|ids['validation'])))
        x,y,pid,ch,center=[d[k][ix] for k in ['x','y','patient','channel','center']]
        pid=pid.astype(str); ch=ch.astype(str); center=center.astype(str)
        tr=np.flatnonzero(np.isin(pid,list(ids['fit']))); va=np.flatnonzero(np.isin(pid,list(ids['validation'])))
        xx,pre=prepare(x,pid,tr)
        old=torch.load(a.a0_runtime/f'fold{fold}/DEVELOPMENT_PRIVATE.pt',weights_only=False)
        assert all(np.array_equal(old[k],value) for k,value in [('x',xx),('y',y),('patient',pid),('channel',ch),('features',d['features']),('train',tr),('val',va)])
        assert all(np.array_equal(old['pre'][k],pre[k]) for k in ['imputer_statistics','mean','scale','var'])
        assert set(pre['fit_patient_ids'])==ids['fit']
        sel=next(r for r in selections if r['fold']==fold and r['arm']=='A0')
        checkpoint=a.a0_runtime/f'fold{fold}/A0/BEST_PRIVATE.pt'
        assert sha(checkpoint)==sel['checkpoint_sha256']
        saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
        assert saved['epoch']==sel['epoch'] and saved['threshold']==sel['threshold']
        m=PRMLP().eval().requires_grad_(False); m.load_state_dict(saved['model'])
        with torch.no_grad():score=torch.sigmoid(m(torch.from_numpy(xx[va]))).numpy()
        frame=pd.DataFrame({'patient':pid[va],'channel':ch[va],'center':center[va],'y_nez':y[va],'score_nez':score})
        reference=pd.read_csv(a.a0_runtime/f'fold{fold}/A0/VALIDATION_CHANNEL_PRIVATE.csv')
        join=frame.merge(reference,on=['patient','channel','center','y_nez'],suffixes=('_new','_old'),validate='one_to_one')
        assert len(join)==len(va)==len(reference)
        drift=float(np.abs(join.score_nez_new-join.score_nez_old).max()); assert drift<=lock['A0_replay_probability_tolerance']
        pm=patient_metrics(y[va],score,pid[va],center[va],sel['threshold'])
        oldpm=pd.read_csv(a.a0_runtime/f'fold{fold}/A0/VALIDATION_PATIENT_PRIVATE.csv')
        joined=pm.merge(oldpm,on=['patient','center'],suffixes=('_new','_old'),validate='one_to_one')
        assert all(np.allclose(joined[k+'_new'],joined[k+'_old'],rtol=0,atol=1e-12,equal_nan=True) for k in METRICS)
        foldroot=root/f'fold{fold}'; foldroot.mkdir(exist_ok=True)
        pm['arm']='A0'; pm['fold']=fold; patients.append(pm); pm.to_csv(foldroot/'A0_PATIENT_PRIVATE.csv',index=False)
        frame['threshold']=sel['threshold']; frame['arm']='A0'; frame['fold']=fold
        frame.to_csv(foldroot/'A0_CHANNEL_PRIVATE.csv',index=False)
        path=foldroot/'DEVELOPMENT_PRIVATE.pt'
        record={'binding':bind,'x':xx,'y':y,'patient':pid,'channel':ch,'center':center,'features':d['features'],
            'train':tr,'val':va,'pre':pre,'A0_threshold':sel['threshold'],'A0_checkpoint':str(checkpoint),'A0_checkpoint_sha256':sha(checkpoint)}
        if path.exists():
            sealed=torch.load(path,weights_only=False); assert sealed['binding']==bind
            assert all(np.array_equal(sealed[k],record[k]) for k in ['x','y','patient','channel','center','features','train','val'])
        else:torch_write(path,record)
        records.append({'fold':fold,'input_sha256':sha(path),'fit_patients':len(ids['fit']),'validation_patients':len(ids['validation']),
            'fit_channels':len(tr),'validation_channels':len(va),'outer_rows_in_bank':0,'exact_prior_A0_tensors':True,
            'FIT_preprocessing_exact':True,'feature_order_sha256':hashlib.sha256(json.dumps(d['features'].tolist(),separators=(',',':')).encode()).hexdigest()})
        checks.append({'fold':fold,'checkpoint_sha256':sha(checkpoint),'epoch':sel['epoch'],'threshold':sel['threshold'],
            'probability_drift':drift,'metrics':pm[METRICS].mean().to_dict()})
        print('A0_REPLAY_AND_SEAL_PASS',fold,flush=True)
    allpm=pd.concat(patients,ignore_index=True); assert len(allpm)==65 and allpm.patient.nunique()==47
    assert abs(float(allpm.macro_f1.mean())-.6380797828499001)<1e-12
    json_write(pub/'A0_REPRODUCTION.json',{'status':'PASS','folds':checks,'retrained':False,'development_macro_f1':float(allpm.macro_f1.mean()),
        'patient_fold_cells':65,'unique_patients':47,'new_outer_predictions':False,'binding':bind})
    json_write(pub/'DATA_PREPROCESSING_AUDIT.json',{'status':'PASS','folds':records,'patients':80,'channels':7635,'features':88,
        'private_feature_sha256':sha(export),'split_sha256':sha(a.split),'same_input_all_arms':True,
        'model_input_no_patient_center_label':True,'outer_labels_source_materialized_before_sealing':True,'outer_evaluation':False,'binding':bind})
    json_write(pub/'RUN_STATUS.json',{'status':'A0_REPLAY_AND_SEAL_PASS','binding':bind,'outer_evaluation':False})


if __name__=='__main__':main()
