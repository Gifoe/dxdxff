from __future__ import annotations
import math,time,torch,pandas as pd,numpy as np
from tqdm.auto import tqdm
from neuroez_c.task2.tech_outcome_c1.sampling import patient_batches
from neuroez_c.task2.tech_outcome_c1.evaluation import metrics

def _autocast(args):return torch.autocast('cuda',dtype=torch.bfloat16 if args.amp_dtype=='bf16' else torch.float16,enabled=bool(args.amp and torch.cuda.is_available()))
def _bce(frame):
    y=frame.outcome_true.to_numpy(float);p=np.clip(frame.probability_success.to_numpy(float),1e-7,1-1e-7);return float(-np.mean(y*np.log(p)+(1-y)*np.log(1-p)))
def train_epoch(model,cohort,keys,labels,optimizer,fold,epoch,args):
    model.train();plans=patient_batches(keys,labels,args.batch_size,args.seed+fold*1000+epoch);rows=[];optimizer.zero_grad(set_to_none=True);seen=0;started=time.time();progress=tqdm(plans,desc=f'Train epoch {epoch} batches',leave=False,disable=not args.show_progress)
    for batch_index,batch in enumerate(progress):
        grouped=[cohort.train_views(key,fold,epoch,args.train_views_per_patient) for key in batch]
        with _autocast(args):outputs=model.forward_patient_views(grouped,args.encoder_window_batch_size_train)
        logits=torch.stack([x['logit'] for x in outputs]).float();targets=torch.tensor([labels[k] for k in batch],dtype=torch.float32,device=logits.device);loss=torch.nn.functional.binary_cross_entropy_with_logits(logits,targets);(loss/args.gradient_accumulation_steps).backward();seen+=len(batch)
        if (batch_index+1)%args.gradient_accumulation_steps==0 or batch_index+1==len(plans):torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip_norm);optimizer.step();optimizer.zero_grad(set_to_none=True)
        probs=torch.sigmoid(logits.detach()).cpu();rows += [{'patient_key':key,'outcome_true':labels[key],'probability_success':float(prob)} for key,prob in zip(batch,probs)];progress.set_postfix(bce=f'{float(loss):.4f}',patients_seen=seen,train_views=args.train_views_per_patient)
    frame=pd.DataFrame(rows);m=metrics(frame);return {'train_bce':_bce(frame),'train_auroc':m['auroc'],'train_probability_mean':m['probability_mean'],'train_probability_std':m['probability_std'],'sampled_patient_rows':len(frame),'unique_sampled_patients':frame.patient_key.nunique(),'duplicate_patients':int(frame.patient_key.duplicated().sum()),'actual_batches':len(plans),'expected_batches':math.ceil(len(keys)/args.batch_size),'epoch_wall_time_seconds':time.time()-started}
def _assert_capacity(views,args):
    for view in views:
        if len(view['seizures'])>args.max_seizures_per_view:raise RuntimeError('eval seizure cap violated')
        for seizure in view['seizures']:
            if len(seizure['channel_names'])>args.max_channels_per_seizure:raise RuntimeError('eval channel cap violated')
            phase=seizure['phase_ids'];mask=seizure['window_mask']
            for channel in range(mask.shape[0]):
                for phase_id in range(3):
                    if int((mask[channel]&(phase[channel]==phase_id)).sum())>args.max_windows_per_phase:raise RuntimeError('eval phase-window cap violated')
@torch.inference_mode()
def infer_multiview(model,cohort,keys,labels,fold,args,description,epoch=None):
    model.eval();patients=[];view_rows=[];core_rows=[]
    for key in tqdm(keys,desc=description,disable=not args.show_progress):
        views=cohort.eval_views(key,fold,args.eval_views_per_patient);_assert_capacity(views,args)
        with _autocast(args):result=model.forward_patient_views([views],args.encoder_window_batch_size_eval)[0]
        logits=result['view_logits'].float();patient_logit=float(logits.mean());prob=float(torch.sigmoid(logits.mean()));assert abs(patient_logit-float(logits.mean()))<=1e-7;patients.append({'patient_key':key,'center':views[0]['center'],'outcome_true':labels[key],'success_logit':patient_logit,'probability_success':prob,'prediction_05':int(prob>=.5),'n_eval_views':len(views)})
        for index,(view,logit,output) in enumerate(zip(views,logits,result['view_outputs'])):
            view_rows.append({'patient_key':key,'outer_fold':fold,'view_index':index,'outcome_true':labels[key],'view_logit':float(logit),'view_probability_success':float(torch.sigmoid(logit)),'n_selected_seizures':len(view['seizures']),'mean_selected_channels':float(np.mean([len(s['channel_names']) for s in view['seizures']])),'n_selected_windows':sum(int(s['window_mask'].sum()) for s in view['seizures'])})
            for audit in output['core_audits']:core_rows.append({'patient_key':key,'outer_fold':fold,'epoch':epoch, 'view_index':index,**audit})
    return pd.DataFrame(patients),pd.DataFrame(view_rows),pd.DataFrame(core_rows)
def cotar_diagnostics(core,fold,epoch):
    rows=[]
    for layer,group in core.groupby('layer_index'):
        valid=group.n_valid_channels.clip(lower=1);entropy=group.mean_channel_weight_entropy;rows.append({'outer_fold':fold,'epoch':epoch,'layer_index':layer,'mean_effective_channel_ratio':float((group.effective_channel_count/valid).mean()),'median_effective_channel_ratio':float((group.effective_channel_count/valid).median()),'mean_max_weight_uniform_ratio':float((group.maximum_channel_weight*valid).mean()),'mean_normalized_entropy':float((entropy/np.log(valid).replace(0,np.nan)).fillna(0).mean())})
    return rows
def better_checkpoint(current_auroc,current_bce,best_auroc,best_bce,auroc_delta=1e-6,bce_delta=1e-4):return np.isfinite(current_auroc) and (current_auroc>best_auroc+auroc_delta or (abs(current_auroc-best_auroc)<=auroc_delta and current_bce<best_bce-bce_delta))
