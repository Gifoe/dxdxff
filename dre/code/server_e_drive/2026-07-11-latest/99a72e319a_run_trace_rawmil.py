from __future__ import annotations
import argparse, os, sys
from pathlib import Path
import numpy as np, pandas as pd, torch
from sklearn.model_selection import StratifiedShuffleSplit
PROJECT=Path(__file__).resolve().parents[2]; REPO=PROJECT.parent
for p in (REPO,PROJECT): sys.path.insert(0,str(p))
from neuroez_c.task2.data import load_cache,filtered_cache
from neuroez_c.task2.exclusions import load_exclusion_manifest
from neuroez_c.task2.outcomes import load_outcome_table
from neuroez_c.task2.protocol import load_fold_manifest,manifest_patient_keys
from neuroez_c.task2.trace_rawmil.raw_dataset import build_patients
from neuroez_c.task2.trace_rawmil.model import TRACERawMIL
from neuroez_c.task2.trace_rawmil.trainer import seed_all,train_epoch,infer,ModelEMA,validation_statistics,better_checkpoint
from neuroez_c.task2.trace_rawmil.raw_dataset import patient_tensors
from neuroez_c.task2.trace_rawmil.losses import trace_loss
from neuroez_c.task2.trace_rawmil.evaluation import metric_row,bootstrap_ci
from neuroez_c.task2.trace_rawmil.audit import protocol
from neuroez_c.task2.trace_rawmil.cache import atomic_json

def parser():
 p=argparse.ArgumentParser(); e=os.getenv
 for key,env in [('raw_cache','DRE_TASK1_RAW_CACHE_PATH'),('outcome_table','DRE_TASK2_OUTCOME_TABLE'),('fold_manifest','DRE_TASK2_FOLD_MANIFEST'),('exclusion_manifest','DRE_TASK2_EXCLUSION_MANIFEST'),('output_dir','DRE_TASK2_TRACE_OUTPUT_DIR'),('cache_dir','DRE_TASK2_TRACE_CACHE_DIR')]: p.add_argument('--'+key,default=e(env),required=e(env) is None)
 p.add_argument('--seed',type=int,default=42);p.add_argument('--folds',default='all',help='Comma-separated outer folds; use 1 for the requested Fold 1 rerun.')
 p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu');p.add_argument('--batch_size',type=int,default=4);p.add_argument('--max_epochs',type=int,default=15);p.add_argument('--patience',type=int,default=3);p.add_argument('--min_delta',type=float,default=1e-4);p.add_argument('--learning_rate',type=float,default=3e-4);p.add_argument('--weight_decay',type=float,default=1e-3);p.add_argument('--gradient_clip_norm',type=float,default=1.0);p.add_argument('--gradient_accumulation_steps',type=int,default=2);p.add_argument('--ema_decay',type=float,default=.995)
 p.add_argument('--encoder-window-batch-size-train',dest='encoder_window_batch_size_train',type=int,default=1024);p.add_argument('--encoder-window-batch-size-eval',dest='encoder_window_batch_size_eval',type=int,default=4096);p.add_argument('--gradient-checkpoint-encoder',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--auto-window-batch-backoff',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--min-window-batch-size',type=int,default=128);p.add_argument('--vram-budget-fraction',type=float,default=.75);p.add_argument('--compile-encoder',action=argparse.BooleanOptionalAction,default=False)
 p.add_argument('--max_seizures_train',type=int,default=2);p.add_argument('--max_ez_channels_train',type=int,default=24);p.add_argument('--max_nez_channels_train',type=int,default=32);p.add_argument('--max_windows_per_phase_train',type=int,default=4);p.add_argument('--resume',action=argparse.BooleanOptionalAction,default=True);p.add_argument('--recompute-training',action='store_true');p.add_argument('--strict',action='store_true');p.add_argument('--audit_only',action='store_true');p.add_argument('--smoke_test',action='store_true'); return p

def csv(path,frame):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_suffix('.tmp');frame.to_csv(tmp,index=False);tmp.replace(path)
def split_rows(train,fold,seed,manifest_path):
 """Persisted, patient-level 80:20 inner split; centre/outcome strata when feasible."""
 existing=pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
 old=existing[existing.outer_fold.eq(fold)] if not existing.empty and 'outer_fold' in existing else pd.DataFrame()
 by={x.patient_key:x for x in train}
 if not old.empty and set(old.patient_key).issubset(by) and set(old.partition)>={'inner_train','inner_validation'}:
  tr=[by[k] for k in old.loc[old.partition.eq('inner_train'),'patient_key']]; va=[by[k] for k in old.loc[old.partition.eq('inner_validation'),'patient_key']]
  return tr,va,existing
 labels=np.array([x.outcome_success for x in train]); centres=np.array([x.center for x in train]); pair=np.array([f'{c}|{y}' for c,y in zip(centres,labels)])
 strata=pair if all((pair==s).sum()>=2 for s in np.unique(pair)) else labels
 try: ti,vi=next(StratifiedShuffleSplit(1,test_size=.20,random_state=seed+int(fold)).split(np.zeros(len(train)),strata))
 except ValueError: ti,vi=next(StratifiedShuffleSplit(1,test_size=.20,random_state=seed+int(fold)).split(np.zeros(len(train)),labels))
 tr=[train[i] for i in ti];va=[train[i] for i in vi]
 current=pd.DataFrame([{'patient_key':x.patient_key,'outer_fold':fold,'partition':'inner_train'} for x in tr]+[{'patient_key':x.patient_key,'outer_fold':fold,'partition':'inner_validation'} for x in va])
 return tr,va,pd.concat([existing[~existing.outer_fold.eq(fold)] if not existing.empty else existing,current],ignore_index=True)

def autotune(model, patients, device, args, out):
 total=torch.cuda.get_device_properties(device).total_memory if device.type=='cuda' else 0
 report={'total_dedicated_vram_gb':total/2**30,'vram_budget_fraction':args.vram_budget_fraction,'vram_budget_gb':total*args.vram_budget_fraction/2**30,'candidates':[],'selected_train_window_batch_size':args.encoder_window_batch_size_train,'selected_eval_window_batch_size':args.encoder_window_batch_size_eval,'compile_encoder':False}
 if device.type!='cuda' or not patients: atomic_json(out/'trace_memory_autotune.json',report);return
 sample=patients[:min(4,len(patients))]; tensors=[patient_tensors(p,'cpu',np.random.default_rng(args.seed+i),args.max_seizures_train,args.max_ez_channels_train,args.max_nez_channels_train,args.max_windows_per_phase_train) for i,p in enumerate(sample)]
 tensors=[x for x in tensors if x]
 for size in (256,512,1024,2048):
  try:
   torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(device);model.train();model.configure_encoder(size,args.gradient_checkpoint_encoder,True);start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();outputs=model(tensors);loss,_=trace_loss(outputs,[p.outcome_success for p in sample[:len(tensors)]],1);loss.backward();end.record();torch.cuda.synchronize(device);elapsed=max(start.elapsed_time(end)/1000,1e-6);reserved=torch.cuda.max_memory_reserved(device);entry={'window_batch_size':size,'peak_memory_allocated':torch.cuda.max_memory_allocated(device),'peak_memory_reserved':reserved,'windows_per_second':sum(model.encoder.last_n_flat_windows for _ in [0])/elapsed,'fits_budget':reserved < total*args.vram_budget_fraction};report['candidates'].append(entry);model.zero_grad(set_to_none=True)
  except RuntimeError as error:
   if 'out of memory' not in str(error).lower(): raise
   report['candidates'].append({'window_batch_size':size,'oom':True});model.zero_grad(set_to_none=True);torch.cuda.empty_cache()
 eligible=[x['window_batch_size'] for x in report['candidates'] if x.get('fits_budget')]
 if '--encoder-window-batch-size-train' not in sys.argv and eligible: report['selected_train_window_batch_size']=max(eligible)
 args.current_train_window_batch_size=report['selected_train_window_batch_size'];args.current_eval_window_batch_size=args.encoder_window_batch_size_eval;atomic_json(out/'trace_memory_autotune.json',report)

def main():
 a=parser().parse_args();seed_all(a.seed);out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);atomic_json(out/'run_args.json',vars(a));Path(a.cache_dir).mkdir(parents=True,exist_ok=True)
 source=load_cache(a.raw_cache);cache=filtered_cache(source,load_exclusion_manifest(a.exclusion_manifest)); outcomes,_=load_outcome_table(a.outcome_table,cache=cache);outcomes=outcomes[outcomes.outcome_group.isin(['success','failure'])];folds=load_fold_manifest(a.fold_manifest,outcomes,strict=a.strict);outcomes=outcomes[outcomes.patient_key.isin(manifest_patient_keys(a.fold_manifest))];labels=outcomes.set_index('patient_key').outcome_label.astype(int).to_dict();patients,audit=build_patients(cache['run_records'],labels);by={x.patient_key:x for x in patients};patients=[by[x] for x in folds.patient_key if x in by]
 if a.strict and len(patients)!=len(folds): raise ValueError('TRACE cohort missing raw-valid patients')
 af=pd.DataFrame(audit);csv(out/'trace_raw_schema_audit.csv',af);atomic_json(out/'trace_raw_schema_audit.json',{'n_records':len(af),'columns':list(af.columns),'records':af.to_dict(orient='records')});csv(out/'trace_channel_alignment_audit.csv',af);csv(out/'trace_patient_cohort.csv',pd.DataFrame([{'patient_key':x.patient_key,'center':x.center,'outcome_true':x.outcome_success,'n_seizures':len(x.seizures)} for x in patients]));atomic_json(out/'trace_label_direction_audit.json',{'nez_equals':1,'ez_equals':0,'unknown_ignored':True});m0=TRACERawMIL();atomic_json(out/'trace_model_audit.json',{'model_version':'TRACE_RAWMIL_V1_5_15EPOCH','parameter_count':m0.parameter_count(),'under_200k':m0.parameter_count()<200000,'ema_decay':a.ema_decay,'phase_delta_enabled':False,'seizure_centering_enabled':False,'residual_pooling':'sparsemax_ez_conditioned','cross_seizure_pooling':'reliability_weighted_plus_max','ranking_max_weight':.05,'early_stopping_patience':a.patience});atomic_json(out/'trace_protocol_audit.json',protocol(folds,a.seed))
 if a.audit_only:return 0
 requested=sorted(folds.outer_fold.unique()) if a.folds=='all' else [int(x) for x in a.folds.split(',')]; device=torch.device(a.device);a.current_train_window_batch_size=a.encoder_window_batch_size_train;a.current_eval_window_batch_size=a.encoder_window_batch_size_eval;allpred=[];allatt=[];history=[];summary=[];manifest_path=out/'trace_inner_validation_manifest.csv'
 for fold in requested:
  test_keys=set(folds.loc[folds.outer_fold==fold,'patient_key']);train=[x for x in patients if x.patient_key not in test_keys];test=[x for x in patients if x.patient_key in test_keys];tr,va,manifest=split_rows(train,fold,a.seed,manifest_path);csv(manifest_path,manifest)
  model=TRACERawMIL().to(device);autotune(model,tr,device,a,out);ema=ModelEMA(model,a.ema_decay);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay);scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode='min',factor=.5,patience=5,min_lr=1e-5);best=float('inf');best_epoch=0;best_auroc=-float('inf');wait=0;best_model=best_ema=best_auc_ema=None
  raw0,_=infer(model,va,device,a.current_eval_window_batch_size,a.auto_window_batch_backoff,a.min_window_batch_size,a.vram_budget_fraction);ema0,_=infer(ema.model,va,device,a.current_eval_window_batch_size,a.auto_window_batch_backoff,a.min_window_batch_size,a.vram_budget_fraction);raw_initial=validation_statistics(raw0);ema_initial=validation_statistics(ema0);ema_diff=max((x-y).abs().max().item() for x,y in zip(model.parameters(),ema.model.parameters()));raw_auc=metric_row(pd.DataFrame(raw0)).get('auroc',np.nan);ema_auc=metric_row(pd.DataFrame(ema0)).get('auroc',np.nan);print(f'[TRACE-v1.5] epoch 0 raw_initial_val_selection={raw_initial["validation_selection_loss"]:.6f} raw_initial_val_auroc={raw_auc} ema_initial_val_selection={ema_initial["validation_selection_loss"]:.6f} ema_initial_val_auroc={ema_auc} raw_ema_max_parameter_difference={ema_diff:.3g}',flush=True)
  if ema_diff!=0 or raw_initial['validation_selection_loss']>.90: raise RuntimeError('Initial validation safety check failed: inspect EMA/checkpoint/output bias before training.')
  print(f'[TRACE-RawMIL] fold {fold}: train={len(tr)} validation={len(va)} test={len(test)} device={device}; selection=validation BCE + 0.2 rank',flush=True)
  for epoch in range(1,a.max_epochs+1):
   train_stats=train_epoch(model,tr,opt,ema,device,a.seed+epoch,a,epoch);rows,_=infer(ema.model,va,device,a.current_eval_window_batch_size,a.auto_window_batch_backoff,a.min_window_batch_size,a.vram_budget_fraction);a.current_eval_window_batch_size=ema.model.encoder_window_batch_size;v=validation_statistics(rows);vf=pd.DataFrame(rows);v['validation_auroc']=metric_row(vf).get('auroc',np.nan) if len(vf) and vf.outcome_true.nunique()==2 else np.nan;scheduler.step(v['validation_selection_loss'])
   vl=np.asarray([x['logit'] for x in rows],float);vy=np.asarray([x['outcome_true'] for x in rows],float);row={'outer_fold':fold,'epoch':epoch,'train_total_loss':train_stats.get('total',np.nan),'train_bce':train_stats.get('bce',np.nan),'train_rank_raw':train_stats.get('rank_raw',np.nan),'train_rank_weighted':train_stats.get('rank_weighted',np.nan),'lambda_rank':train_stats.get('rank_weight',np.nan),'train_success_logit_mean':train_stats.get('success_logit_mean',np.nan),'train_failure_logit_mean':train_stats.get('failure_logit_mean',np.nan),'train_logit_std':train_stats.get('logit_std',np.nan),'validation_success_logit_mean':float(vl[vy==1].mean()) if (vy==1).any() else np.nan,'validation_failure_logit_mean':float(vl[vy==0].mean()) if (vy==0).any() else np.nan,'validation_logit_std':float(vl.std()),'gradient_norm':train_stats.get('gradient_norm',np.nan),'learning_rate':opt.param_groups[0]['lr'],'current_train_window_batch_size':a.current_train_window_batch_size,'current_eval_window_batch_size':a.current_eval_window_batch_size,**train_stats,**getattr(ema.model,'last_infer_performance',{}),**v};history.append(row)
   print(f"[TRACE-v2] fold {fold} epoch {epoch}/{a.max_epochs}: train_total={row['train_total_loss']:.5f} rank={row['train_rank_raw']:.5f} lambda_rank={row['lambda_rank']:.4f} val_selection={v['validation_selection_loss']:.5f} val_auroc={v['validation_auroc']} best_epoch={best_epoch} wait={wait}/{a.patience}",flush=True)
   if v['validation_selection_loss'] < best-a.min_delta:
    best=v['validation_selection_loss'];best_epoch=epoch;wait=0;best_model={k:x.detach().cpu().clone() for k,x in model.state_dict().items()};best_ema={k:x.detach().cpu().clone() for k,x in ema.state_dict().items()}
   else: wait+=1
   if np.isfinite(v['validation_auroc']) and v['validation_auroc']>best_auroc: best_auroc=v['validation_auroc'];best_auc_ema={k:x.detach().cpu().clone() for k,x in ema.state_dict().items()}
   if wait>=a.patience:
    print(f'[TRACE-RawMIL] fold {fold}: early stopping at epoch {epoch}; best selection loss={best:.6f}',flush=True);break
  last_cp=Path(a.cache_dir)/'checkpoints_v1_5_15epoch'/f'fold_{fold}_seed_{a.seed}_last.pt';last_cp.parent.mkdir(parents=True,exist_ok=True);torch.save({'model_version':'TRACE_RAWMIL_V1_5_15EPOCH','model_state_dict':model.state_dict(),'ema_state_dict':ema.state_dict(),'fold':int(fold)},last_cp);model.load_state_dict(best_model);ema.load_state_dict(best_ema);cp=Path(a.cache_dir)/'checkpoints_v1_5_15epoch'/f'fold_{fold}_seed_{a.seed}.pt';Path(a.cache_dir,'training_state_v1_5_15epoch').mkdir(parents=True,exist_ok=True);torch.save({'model_version':'TRACE_RAWMIL_V1_5_15EPOCH','model_state_dict':model.state_dict(),'ema_state_dict':ema.state_dict(),'fold':int(fold),'best_epoch':best_epoch,'best_validation_selection_loss':best,'selection_metric':'validation_bce + 0.2 * validation_pairwise_rank_loss'},cp);torch.save({'model_version':'TRACE_RAWMIL_V1_5_15EPOCH','ema_state_dict':best_auc_ema,'fold':int(fold),'best_validation_auroc':best_auroc},cp.with_name(cp.stem+'_best_auroc.pt'));print(f'[TRACE-v1.5-15] loading best selection checkpoint: fold={fold} epoch={best_epoch} selection={best:.6f} checkpoint={cp}',flush=True)
  pr,at=infer(ema.model,test,device,a.current_eval_window_batch_size,a.auto_window_batch_backoff,a.min_window_batch_size,a.vram_budget_fraction);frame=pd.DataFrame(pr);frame['outer_fold']=fold;frame['prediction_05']=(frame.probability_success>=.5).astype(int);frame['seed']=a.seed;allpred.append(frame);allatt.extend(at);summary.append({'outer_fold':fold,'n_train':len(tr),'n_validation':len(va),'n_test':len(test),'best_validation_selection_loss':best,'checkpoint':str(cp),'used_ema_for_validation_and_test':True});
  if device.type=='cuda': torch.cuda.empty_cache()
 pred=pd.concat(allpred,ignore_index=True);att=pd.DataFrame(allatt);hist_frame=pd.DataFrame(history);training_rows=[]
 for item in summary:
  fold=item['outer_fold'];metrics=metric_row(pred[pred.outer_fold==fold]);epochs_run=int((hist_frame.outer_fold==fold).sum());best_row=hist_frame[(hist_frame.outer_fold==fold)&(hist_frame.validation_selection_loss==item['best_validation_selection_loss'])].iloc[0];training_rows.append({'outer_fold':fold,'train_n':item['n_train'],'validation_n':item['n_validation'],'test_n':item['n_test'],'epochs_run':epochs_run,'best_epoch':int(best_row.epoch),'best_validation_selection':item['best_validation_selection_loss'],'best_validation_auroc':best_row.validation_auroc,'test_auroc':metrics['auroc'],'test_accuracy':metrics['accuracy'],'test_balanced_accuracy':metrics['balanced_accuracy'],'test_macro_f1':metrics['macro_f1']})
 csv(out/'training_history.csv',hist_frame);csv(out/'fold_training_summary.csv',pd.DataFrame(summary));csv(out/'training_summary.csv',pd.DataFrame(training_rows));csv(out/'fold_checkpoints.csv',pd.DataFrame(summary));csv(out/'oof_patient_predictions.csv',pred);csv(out/'fold_metrics.csv',pd.DataFrame([{'outer_fold':f,**metric_row(pred[pred.outer_fold==f])} for f in sorted(pred.outer_fold.unique())]));csv(out/'summary_metrics.csv',pd.DataFrame([metric_row(pred)]));csv(out/'bootstrap_ci.csv',bootstrap_ci(pred,2000,a.seed));csv(out/'confusion_matrix.csv',pd.crosstab(pred.outcome_true,pred.prediction_05).reset_index());csv(out/'metrics_by_center.csv',pd.DataFrame([{'center':c,**metric_row(g)} for c,g in pred.groupby('center')]));csv(out/'residual_channel_attention.csv',att);csv(out/'seizure_reliability_summary.csv',att[['patient_key','seizure_id','seizure_reliability_beta']].drop_duplicates() if not att.empty else att);csv(out/'patient_residual_summary.csv',pred[['patient_key','patient_residual_strength']]);csv(out/'misclassified_patients.csv',pred[pred.outcome_true!=pred.prediction_05]);(out/'TRACE_RAWMIL_REPORT.md').write_text('TRACE-RawMIL V1.5 15-epoch five-fold diagnostic; best selection checkpoints only for outer test.\n',encoding='utf8');return 0
if __name__=='__main__':raise SystemExit(main())
