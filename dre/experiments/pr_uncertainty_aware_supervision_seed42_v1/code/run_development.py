"""Run five matched PR-UAS folds, guarded by frozen B0 replay and source tests."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from uas_core import (PRMLP, sha, json_write, torch_write, seed_all, state_hash,
                      prepare, oof_plan, mc_predict, train_model, patient_metrics)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True)
    p.add_argument('--device',default='cuda')
    p.add_argument('--source',type=Path,required=True)
    a=p.parse_args(); root=a.runtime
    torch.set_num_threads(2)
    gate=json.loads((root/'gate'/'A0_REPRODUCTION.json').read_text())
    tests=json.loads((root/'TEST_AND_SMOKE_AUDIT.json').read_text())
    assert gate['status']=='PASS' and tests['status']=='PASS'
    data=root/'gate'/'FEATURES_PRIVATE.npz'
    assert sha(data)==gate['private_features_sha256']
    binding={**{n:sha(Path(__file__).parent/n) for n in ['uas_core.py',Path(__file__).name,'test_and_smoke.py']},
             'protocol':sha(a.protocol),'gate':sha(root/'gate'/'A0_REPRODUCTION.json'),
             'data':sha(data),'tests':sha(root/'TEST_AND_SMOKE_AUDIT.json')}
    bind=json.dumps(binding,sort_keys=True)
    # Hash-verified export made by this task, not an arbitrary untrusted NPZ.
    with np.load(data,allow_pickle=True) as z:
        arrays={k:z[k] for k in z.files}
    x,y,pid,channel,center=[arrays[k] for k in ['x','y','patient','channel','center']]
    assert x.shape==(7635,88) and len(set(pid))==80
    plan_audit=[]
    for fold in range(1,6):
        take=arrays['split_fold']==fold
        ids={r:set(arrays['split_patient'][take & (arrays['split_role']==r)]) for r in ['fit','validation','test']}
        assert ids['fit']|ids['validation']|ids['test']==set(pid)
        assert all(not ids[r]&ids[s] for r,s in [('fit','validation'),('fit','test'),('validation','test')])
        fi=np.flatnonzero(np.isin(pid,list(ids['fit'])))
        # Teachers receive exclusively FIT patient arrays and labels.
        fx,fy,fp,fc=x[fi],y[fi],pid[fi],channel[fi]
        foldroot=root/f'fold{fold}'; foldroot.mkdir(exist_ok=True)
        plans=oof_plan(ids['fit'],fold)
        q=np.full(len(fi),np.nan); u=q.copy(); variance=q.copy(); ee=q.copy(); mi=q.copy(); seen=np.zeros(len(fi),int)
        for plan in plans:
            tr=np.flatnonzero(np.isin(fp,plan['train'])); va=np.flatnonzero(np.isin(fp,plan['validation']))
            query=np.flatnonzero(np.isin(fp,plan['query']))
            assert set(fp[tr])<=ids['fit'] and not set(fp[tr])&ids['validation']
            tx,pre=prepare(fx,fp,tr)
            assert pre['fit_patient_ids']==plan['train']
            cell=foldroot/f'teacher{plan["group"]}'
            seed_all(plan['seed']); initial=PRMLP().state_dict()
            teacher,best,history=train_model(tx,fy,fp,tr,va,cell,plan['seed'],initial,
                bind+json.dumps(plan,sort_keys=True),a.device)
            prediction=cell/'OOF_PRIVATE.pt'
            source={'binding':bind,'plan':plan,'checkpoint':sha(cell/'BEST_PRIVATE.pt')}
            if prediction.exists():
                saved=torch.load(prediction,weights_only=False)
                assert saved['source']==source
                assert np.array_equal(saved['patient'],fp[query]) and np.array_equal(saved['channel'],fc[query])
                mc=saved['mc']
            else:
                mc=mc_predict(teacher,torch.as_tensor(tx[query],device=a.device),plan['seed']+9000000)
                torch_write(prediction,{'source':source,'patient':fp[query],'channel':fc[query],
                                        'mc':mc,'preprocessor':pre})
            for target,key in [(q,'q'),(u,'u'),(variance,'variance'),(ee,'expected_entropy'),(mi,'mutual_information')]:
                target[query]=mc[key]
            seen[query]+=1
            plan_audit.append({'fold':fold,'teacher_group':plan['group'],'train_patients':len(plan['train']),
                'selection_patients':len(plan['validation']),'query_patients':len(plan['query']),
                'query_channels':len(query),'query_train_overlap':0,'query_selection_overlap':0,
                'outer_validation_test_overlap':0,'preprocessing_fit_only':True,
                'query_identity_sha256':__import__('hashlib').sha256(json.dumps(list(zip(fp[query],fc[query]))).encode()).hexdigest(),
                'OOF_file_sha256':sha(prediction),'checkpoint_sha256':source['checkpoint'],
                'selected_epoch':best['epoch'],'threshold_nez':best['threshold']})
        assert (seen==1).all() and all(np.isfinite(v).all() for v in [q,u,variance,ee,mi])
        assert ((q>0)&(q<1)).all() and ((u>=0)&(u<=1)).all()
        oof=foldroot/'OOF_FROZEN_PRIVATE.pt'
        frozen={'binding':bind,'patient':fp,'channel':fc,'q':q,'u':u,'variance':variance,
                'expected_entropy':ee,'mutual_information':mi,'y_nez':fy,
                'teacher_hashes':[r['OOF_file_sha256'] for r in plan_audit if r['fold']==fold]}
        if oof.exists():
            old=torch.load(oof,weights_only=False)
            assert old['binding']==bind and old['teacher_hashes']==frozen['teacher_hashes']
            assert np.array_equal(old['q'],q) and np.array_equal(old['u'],u)
        else: torch_write(oof,frozen)
        # No outer labels or features are presented to any student during development.
        di=np.flatnonzero(np.isin(pid,list(ids['fit']|ids['validation'])))
        dx,dy,dp,dc,dcenter=x[di],y[di],pid[di],channel[di],center[di]
        tr=np.flatnonzero(np.isin(dp,list(ids['fit']))); va=np.flatnonzero(np.isin(dp,list(ids['validation'])))
        prepared,pre=prepare(dx,dp,tr)
        assert np.array_equal(dp[tr],fp) and np.array_equal(dc[tr],fc)
        dq=np.full(len(di),np.nan); du=dq.copy(); dq[tr]=q; du[tr]=u
        seed=42+1009*fold; seed_all(seed); initial=PRMLP().state_dict(); initial_sha=state_hash(initial)
        for arm in ['A0','A1','A2']:
            json_write(root/'RUN_STATUS.json',{'status':'TRAINING_DEVELOPMENT','fold':fold,'arm':arm,
                'outer_evaluation_accessed':False,'historical_B0_replay_only':True,'binding':binding})
            cell=foldroot/arm
            model,best,history=train_model(prepared,dy,dp,tr,va,cell,seed,initial,
                bind+sha(oof),a.device,arm=arm,q=dq,u=du)
            with torch.no_grad(): score=torch.sigmoid(model(torch.as_tensor(prepared[va],device=a.device))).cpu().numpy()
            patients=patient_metrics(dy[va],score,dp[va],dcenter[va],best['threshold'])
            patients['fold']=fold; patients['arm']=arm
            patients.to_csv(cell/'VALIDATION_PATIENT_PRIVATE.csv',index=False)
            pred=pd.DataFrame({'patient':dp[va],'channel':dc[va],'center':dcenter[va],
                'y_nez':dy[va],'score_nez':score,'threshold':best['threshold'],'fold':fold,'arm':arm})
            pred.to_csv(cell/'VALIDATION_CHANNEL_PRIVATE.csv',index=False)
            torch_write(cell/'PREPROCESSOR_PRIVATE.pt',pre)
            json_write(cell/'SELECTION.json',{'arm':arm,'fold':fold,'epoch':best['epoch'],
                'threshold':best['threshold'],'initial_hash':initial_sha,'parameters':8817,
                'checkpoint_sha256':sha(cell/'BEST_PRIVATE.pt'),'OOF_sha256':sha(oof),
                'metrics':{k:float(v) for k,v in patients.select_dtypes('number').mean().items()},
                'completed_epochs':len(history)})
            print('STUDENT_COMPLETE',fold,arm,'selected_epoch',best['epoch'],flush=True)
    json_write(root/'OOF_TEACHER_AUDIT.json',{'status':'PASS','teachers':20,'folds':plan_audit,
        'complete_once_per_FIT_channel':True,'MC_passes':10,'no_outer_labels_used':True,
        'preprocess_teacher_TRAIN_only':True,'features_dimension':88})
    json_write(root/'RUN_STATUS.json',{'status':'ALL_DEVELOPMENT_TRAINING_COMPLETE',
        'student_cells':15,'teacher_cells':20,'outer_evaluation_accessed':False,'binding':binding})
    print('ALL_DEVELOPMENT_TRAINING_COMPLETE',flush=True)


if __name__=='__main__': main()
