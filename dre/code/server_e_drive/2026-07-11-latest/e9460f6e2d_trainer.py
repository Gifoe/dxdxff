from __future__ import annotations
import time,contextlib,torch,pandas as pd
from .sampling import patient_batches
from .losses import patient_bce
from .evaluation import metrics
from .progress import tqdm

def train_epoch(model,cohort,keys,labels,optimizer,device,epoch,args):
    model.train();plans=patient_batches(keys,labels,args.batch_size,args.seed+epoch);optimizer.zero_grad(set_to_none=True);rows=[];seen=0;started=time.time();progress=tqdm(plans,desc=f'Train epoch {epoch} batches',leave=False,disable=not args.show_progress)
    for index,batch in enumerate(progress):
        views=[cohort.view(k,epoch,True) for k in batch];dtype=torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16
        with torch.autocast(device_type='cuda',dtype=dtype,enabled=bool(args.amp and torch.device(device).type=='cuda')):outputs=model(views,args.encoder_window_batch_size_train)
        loss=patient_bce(outputs,[labels[k] for k in batch]);(loss/args.gradient_accumulation_steps).backward();seen+=len(batch)
        if (index+1)%args.gradient_accumulation_steps==0 or index+1==len(plans):torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step();optimizer.zero_grad(set_to_none=True)
        probs=torch.sigmoid(torch.stack([x['logit'] for x in outputs]).float()).detach().cpu();rows += [{'patient_key':k,'outcome_true':labels[k],'probability_success':float(p)} for k,p in zip(batch,probs)];progress.set_postfix(bce=f'{float(loss):.4f}',lr=optimizer.param_groups[0]['lr'],patients_seen=seen,raw_windows=sum(int(s['window_mask'].sum()) for v in views for s in v['seizures']),valid_channels=sum(int(s['window_mask'].any(1).sum()) for v in views for s in v['seizures']))
    frame=pd.DataFrame(rows);return {'train_bce':float(-(frame.outcome_true*frame.probability_success.clip(1e-7,1-1e-7).map(__import__('math').log)+(1-frame.outcome_true)*(1-frame.probability_success).clip(1e-7,1-1e-7).map(__import__('math').log)).mean()),'epoch_wall_time_seconds':time.time()-started,**{f'train_{k}':v for k,v in metrics(frame).items() if k in ('auroc','probability_mean','probability_std')},'sampled_patient_rows':len(rows),'unique_sampled_patients':frame.patient_key.nunique(),'duplicate_patients':int(frame.patient_key.duplicated().sum()),'actual_batches':len(plans)}
@torch.inference_mode()
def infer(model,cohort,keys,labels,fold,args,description):
    model.eval();rows=[];cores=[];seizures=[];patients=[]
    for key in tqdm(keys,desc=description,disable=not args.show_progress):
        view=cohort.view(key,0,False);dtype=torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16
        with torch.autocast(device_type='cuda',dtype=dtype,enabled=bool(args.amp and torch.device(args.device).type=='cuda')):output=model([view],args.encoder_window_batch_size_eval)[0]
        logit=float(output['logit'].float());prob=float(torch.sigmoid(torch.tensor(logit)));rows.append({'patient_key':key,'center':view['center'],'outcome_true':labels[key],'probability_success':prob,'probability_failure':1-prob,'success_logit':logit,'prediction_05':int(prob>=.5),'n_seizures':output['n_seizures'],'mean_valid_channels':output['mean_valid_channels']});patients.append({'patient_key':key,'outer_fold':fold,'embedding_norm':float(output['patient_embedding'].float().norm()),'finite':bool(torch.isfinite(output['patient_embedding']).all())})
        for seizure_index,u in enumerate(output['seizure_embeddings']):seizures.append({'patient_key':key,'outer_fold':fold,'seizure_index':seizure_index,'embedding_norm':float(u.float().norm()),'finite':bool(torch.isfinite(u).all())})
        for core in output['core_audits']:cores.append({'patient_key':key,'outer_fold':fold,**core})
    return pd.DataFrame(rows),pd.DataFrame(cores),pd.DataFrame(seizures),pd.DataFrame(patients)
