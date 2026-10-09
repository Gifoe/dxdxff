"""Only ten scratch W1/W2 development students; no outer data or new variants."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from base import seed_all,state_hash,sha,json_write,torch_write,patient_metrics,PRMLP
from model import make
from prepare import binding
from train import train,predict,inference_benchmark


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--protocol',type=Path,required=True); p.add_argument('--device',default='cuda')
    a=p.parse_args(); root=a.runtime; pub=root/'public'; torch.set_num_threads(2); bind=binding(a.protocol)
    for name in ['A0_REPRODUCTION','DATA_PREPROCESSING_AUDIT','TEST_AND_SMOKE_AUDIT']:
        r=json.loads((pub/(name+'.json')).read_text()); assert r['status']=='PASS' and r['binding']==bind
    smoke=json.loads((pub/'TEST_AND_SMOKE_AUDIT.json').read_text()); assert smoke['torch']==torch.__version__ and smoke['numpy']==np.__version__
    data=json.loads((pub/'DATA_PREPROCESSING_AUDIT.json').read_text())['folds']
    selections=[]; histories=[]; efficiencies=[]
    for fold in range(1,6):
        path=root/f'fold{fold}/DEVELOPMENT_PRIVATE.pt'; assert sha(path)==next(r for r in data if r['fold']==fold)['input_sha256']
        d=torch.load(path,weights_only=False); assert d['binding']==bind
        va=d['val']; vx=torch.as_tensor(d['x'][va],device=a.device)
        assert sha(d['A0_checkpoint'])==d['A0_checkpoint_sha256']
        am=PRMLP().to(a.device); am.load_state_dict(torch.load(d['A0_checkpoint'],map_location=a.device,weights_only=False)['model'])
        efficiencies.append({'fold':fold,'arm':'A0','parameters':8817,'train_seconds':None,'completed_epochs':None,
            'peak_gpu_bytes':None,'inference_seconds_per_validation_batch':inference_benchmark(am,vx,a.device),
            'validation_channels':len(va),'member_predictions_per_channel':1,'device':str(a.device),'training_reused':True})
        del am
    for arm in ['W1','W2']:
        for fold in range(1,6):
            path=root/f'fold{fold}/DEVELOPMENT_PRIVATE.pt'
            assert sha(path)==next(r for r in data if r['fold']==fold)['input_sha256']
            d=torch.load(path,weights_only=False); assert d['binding']==bind
            va=d['val']; vx=torch.as_tensor(d['x'][va],device=a.device)
            seed=42+1009*fold; seed_all(seed); model=make(arm); initial=copy.deepcopy(model.state_dict())
            parameters=sum(p.numel() for p in model.parameters()); assert parameters=={'W1':10167,'W2':10132}[arm]
            cell=root/f'fold{fold}'/arm; cell.mkdir(exist_ok=True)
            initpath=cell/'INITIAL_PRIVATE.pt'
            if initpath.exists():assert state_hash(torch.load(initpath,weights_only=False))==state_hash(initial)
            else:torch_write(initpath,initial)
            json_write(pub/'RUN_STATUS.json',{'status':'TRAINING','fold':fold,'arm':arm,'binding':bind,'outer_evaluation':False})
            model,best,history,meta=train(d,arm,cell,seed,initial,json.dumps({**bind,'sealed':sha(path)},sort_keys=True),a.device)
            score,members=predict(model,vx)
            if arm=='W2':assert members.shape==(len(va),4) and np.allclose(members.mean(1),score,rtol=0,atol=1e-7)
            pm=patient_metrics(d['y'][va],score,d['patient'][va],d['center'][va],best['threshold'])
            pm['arm']=arm; pm['fold']=fold; pm.to_csv(cell/'PATIENT_PRIVATE.csv',index=False)
            pred=pd.DataFrame({'patient':d['patient'][va],'channel':d['channel'][va],'center':d['center'][va],
                'y_nez':d['y'][va],'score_nez':score,'threshold':best['threshold'],'A0_threshold':d['A0_threshold'],'arm':arm,'fold':fold})
            if arm=='W2':
                for k in range(4):pred[f'member{k}_p_nez']=members[:,k]
            pred.to_csv(cell/'PREDICTIONS_PRIVATE.csv',index=False)
            histories.extend({'fold':fold,'arm':arm,**r} for r in history)
            selections.append({'arm':arm,'fold':fold,'epoch':best['epoch'],'threshold':best['threshold'],
                'parameters':parameters,'initial_sha256':state_hash(initial),'checkpoint_sha256':meta['best_sha256'],
                'input_sha256':sha(path),'completed_epochs':len(history)})
            efficiencies.append({'fold':fold,'arm':arm,'parameters':parameters,'train_seconds':meta['train_seconds'],
                'completed_epochs':len(history),'peak_gpu_bytes':meta['peak_gpu_bytes'],
                'inference_seconds_per_validation_batch':inference_benchmark(model,vx,a.device),
                'validation_channels':len(va),'member_predictions_per_channel':4 if arm=='W2' else 1,'device':str(a.device),'training_reused':False})
            print('SCRATCH_STUDENT_COMPLETE',fold,arm,flush=True)
            del model
    pd.DataFrame(histories).to_csv(pub/'TRAINING_HISTORY.csv',index=False)
    pd.DataFrame(efficiencies).to_csv(pub/'EFFICIENCY_AUDIT.csv',index=False)
    json_write(pub/'SELECTION_AUDIT.json',{'status':'PASS','models':selections,'outer_evaluation':False,'no_member_selection':True})
    json_write(pub/'RUN_STATUS.json',{'status':'ALL_10_STUDENTS_COMPLETE','binding':bind,'A0_retrained':False,'outer_evaluation':False})


if __name__=='__main__':main()
