"""Ten requested temporal students; T0 frozen and no new outer evaluation."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from base import seed_all,state_hash,sha,json_write,torch_write,patient_metrics,METRICS
from model import Temporal
from prepare import binding
from train import train,predict,tensors


def main():
    p=argparse.ArgumentParser(); p.add_argument('--runtime',type=Path,required=True)
    p.add_argument('--protocol',type=Path,required=True); p.add_argument('--device',default='cuda')
    a=p.parse_args(); root=a.runtime; pub=root/'public'; torch.set_num_threads(2)
    bind=binding(a.protocol)
    for name in ['A0_REPRODUCTION','TEMPORAL_NORMALIZATION_AUDIT','OOF_A0_AUDIT','TEST_AND_SMOKE_AUDIT']:
        report=json.loads((pub/(name+'.json')).read_text()); assert report['status']=='PASS'
    assert json.loads((pub/'TEST_AND_SMOKE_AUDIT.json').read_text())['binding']==bind
    norms=json.loads((pub/'TEMPORAL_NORMALIZATION_AUDIT.json').read_text())['folds']
    selections=[]; historyrows=[]; epoch0=[]
    for fold in range(1,6):
        path=root/f'fold{fold}/DEVELOPMENT_PRIVATE.pt'
        assert sha(path)==next(r for r in norms if r['fold']==fold)['input_sha256']
        d=torch.load(path,weights_only=False); assert d['binding']==bind
        seed=42+1009*fold; seed_all(seed); initial=copy.deepcopy(Temporal('T1').state_dict())
        for arm in ['T1','T2']:
            json_write(pub/'RUN_STATUS.json',{'status':'TRAINING','fold':fold,'arm':arm,'outer_evaluation':False,'binding':bind})
            cell=root/f'fold{fold}'/arm
            model,best,history=train(d,arm,cell,seed,initial,json.dumps({**bind,'sealed':sha(path)},sort_keys=True),a.device)
            va=d['val']; bank=tensors(d['banks'],a.device)
            score,delta=predict(model,d,bank,va,a.device)
            pm=patient_metrics(d['y'][va],score,d['patient'][va],d['center'][va],best['threshold'])
            pm['arm']=arm; pm['fold']=fold; pm.to_csv(cell/'PATIENT_PRIVATE.csv',index=False)
            pred=pd.DataFrame({'patient':d['patient'][va],'channel':d['channel'][va],'center':d['center'][va],
                'y_nez':d['y'][va],'score_nez':score,'delta_nez':delta[va],'base_logit':d['base_logits'][va],
                'threshold':best['threshold'],'T0_threshold':d['T0_threshold'],'fold':fold,'arm':arm})
            pred.to_csv(cell/'PREDICTIONS_PRIVATE.csv',index=False)
            for row in history:historyrows.append({'fold':fold,'arm':arm,**row})
            sel={'fold':fold,'arm':arm,'epoch':best['epoch'],'threshold':best['threshold'],
                'initial_sha256':state_hash(initial),'checkpoint_sha256':sha(cell/'BEST_PRIVATE.pt'),
                'input_sha256':sha(path),'parameters':sum(p.numel() for p in model.parameters()),'completed_epochs':len(history)-1}
            selections.append(sel); json_write(cell/'SELECTION.json',sel)
            epoch0.append({'fold':fold,'arm':arm,'max_delta':0.,'initial_probability_exact_T0':True,
                'fallback_selected':best['epoch']==0,'A0_parameters_trained':0})
            print('STUDENT_COMPLETE',fold,arm,'BEST',best['epoch'],flush=True)
    pd.DataFrame(historyrows).to_csv(pub/'TRAINING_HISTORY.csv',index=False)
    json_write(pub/'SELECTION_AUDIT.json',{'status':'PASS','models':selections,'outer_evaluation':False})
    json_write(pub/'EPOCH0_IDENTITY_AUDIT.json',{'status':'PASS','folds':epoch0,'always_eligible_fallback':True})
    json_write(pub/'RUN_STATUS.json',{'status':'ALL_10_TEMPORAL_STUDENTS_COMPLETE','student_models':10,
        'T0_retrained':False,'outer_evaluation':False,'binding':bind})


if __name__=='__main__':main()
