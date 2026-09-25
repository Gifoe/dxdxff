from __future__ import annotations
import argparse,hashlib,json,sys,time,warnings
from pathlib import Path
import numpy as np,pandas as pd,torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
PROJECT=Path(__file__).resolve().parents[2];REPO=PROJECT.parent
for path in (REPO,PROJECT):sys.path.insert(0,str(path))
from neuroez_c.task2.true_dualset import MODEL_VERSION,CACHE_VERSION,PROTOCOL
from neuroez_c.task2.true_dualset.data import CachedCohort,ViewConfig
from neuroez_c.task2.true_dualset.model import TrueEZNEZDualSet
from neuroez_c.task2.true_dualset.trainer import seed_all,EMA,train_epoch,infer
from neuroez_c.task2.true_dualset.evaluation import metric_row,bootstrap_ci
from neuroez_c.task2.true_dualset.audit import verify_patient,model_audit

def write_csv(path,rows):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');pd.DataFrame(rows).to_csv(tmp,index=False);tmp.replace(p)
def write_json(path,obj):
 p=Path(path);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,indent=2,default=str),encoding='utf8');tmp.replace(p)
def stable_hash(value):return hashlib.sha256(json.dumps(value,sort_keys=True,default=str).encode()).hexdigest()
def parse():
 p=argparse.ArgumentParser(description='TRUE_EZ_NEZ_DUALSET_V1 direct outer evaluation')
 p.add_argument('--cache-dir',required=True);p.add_argument('--fold-manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--folds',default='all');p.add_argument('--device',default='cuda');p.add_argument('--fixed-epochs',type=int,default=15);p.add_argument('--batch-size',type=int,default=4);p.add_argument('--gradient-accumulation-steps',type=int,default=2);p.add_argument('--learning-rate',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-3);p.add_argument('--gradient-clip',type=float,default=1.0);p.add_argument('--ema-decay',type=float,default=.98);p.add_argument('--rank-weight',type=float,default=.05);p.add_argument('--n-deterministic-views',type=int,default=4);p.add_argument('--max-seizures-per-view',type=int,default=2);p.add_argument('--max-ez-channels-per-view',type=int,default=24);p.add_argument('--max-nez-channels-per-view',type=int,default=40);p.add_argument('--max-windows-per-phase',type=int,default=4);p.add_argument('--strict',action='store_true');p.add_argument('--smoke-test',action='store_true');p.add_argument('--verify-only',action='store_true');p.add_argument('--no-resume-old-checkpoint',action='store_true');return p.parse_args()
def patient_counts(patient):
 rows=patient['seizures'];ez=[int((s['channel_labels_nez']==0).sum()) for s in rows];nez=[int((s['channel_labels_nez']==1).sum()) for s in rows];return {'mean_ez_count':float(np.mean(ez)),'mean_nez_count':float(np.mean(nez)),'mean_ez_fraction':float(np.mean([x/(x+y) for x,y in zip(ez,nez)])),'total_channel_count':int(sum(x+y for x,y in zip(ez,nez))),'seizure_count':len(rows)}
def validate_and_protocol(cohort,folds,selected,epochs,smoke):
 keys=set(cohort.keys());mf=folds[folds.patient_key.isin(keys)].copy()
 if mf.patient_key.duplicated().any() or set(mf.patient_key)!=keys:raise RuntimeError('every cache patient must appear exactly once in fold manifest')
 counts=[]
 for key in sorted(keys):
  patient=cohort.load(key);verify_patient(patient);counts.append({'patient_key':key,'outcome_true':patient['outcome_success'],**patient_counts(patient)})
 rows=[];all_test=set()
 for fold in selected:
  test=set(mf[mf.outer_fold==fold].patient_key);train=keys-test
  if not test:raise RuntimeError(f'fold {fold} is empty')
  labels=[cohort.load(k)['outcome_success'] for k in test]
  if not smoke and set(labels)!={0,1}:raise RuntimeError(f'fold {fold} does not include both outcomes')
  if all_test&test:raise RuntimeError('outer folds overlap')
  all_test|=test;rows.append({'outer_fold':fold,'train_n':len(train),'test_n':len(test),'train_success':sum(cohort.load(k)['outcome_success'] for k in train),'train_failure':sum(1-cohort.load(k)['outcome_success'] for k in train),'test_success':sum(labels),'test_failure':sum(1-x for x in labels),'train_test_overlap':len(train&test),'train_patient_hash':stable_hash(sorted(train)),'test_patient_hash':stable_hash(sorted(test))})
 if set(selected)==set(mf.outer_fold.unique()) and all_test!=keys:raise RuntimeError('five test folds do not cover the full cache cohort')
 audit={'protocol':PROTOCOL,'model_version':MODEL_VERSION,'n_outer_folds':5,'uses_inner_validation':False,'uses_nested_cv':False,'uses_early_stopping':False,'fixed_epochs':epochs,'final_model':'ema_final','outer_test_used_for_training':False,'outer_test_used_for_epoch_selection':False,'outer_test_used_for_hyperparameter_selection':False,'outer_test_inference_count_per_fold':1,'uses_true_ez_nez_labels':True,'predicts_task1_labels':False,'uses_task1_loss':False,'uses_task1_checkpoint':False,'threshold':.5,'primary_metric':'pooled_oof_auroc','folds':rows}
 return mf,audit,pd.DataFrame(counts)
def shortcut_oof(counts,mf,selected):
 feats=['mean_ez_count','mean_nez_count','mean_ez_fraction','total_channel_count','seizure_count'];rows=[]
 for fold in selected:
  test=mf[mf.outer_fold==fold].patient_key.tolist();train=mf[mf.outer_fold!=fold].patient_key.tolist();tr=counts[counts.patient_key.isin(train)];te=counts[counts.patient_key.isin(test)]
  if tr.outcome_true.nunique()<2:prob=np.full(len(te),.5)
  else:prob=make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,random_state=42)).fit(tr[feats],tr.outcome_true).predict_proba(te[feats])[:,1]
  rows += [{'patient_key':k,'outer_fold':fold,'outcome_true':int(y),'shortcut_probability_success':float(p)} for k,y,p in zip(te.patient_key,te.outcome_true,prob)]
 frame=pd.DataFrame(rows);m=metric_row(frame.rename(columns={'shortcut_probability_success':'probability_success'})) if len(frame) else {};return frame,m
def main():
 a=parse();seed_all(a.seed)
 if a.smoke_test:warnings.filterwarnings('ignore',category=UserWarning,module='sklearn.metrics')
 started=time.time();out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=True);root=Path(a.cache_dir);sig=json.loads((root/'cache_signature.json').read_text(encoding='utf8'));cfg=ViewConfig(a.n_deterministic_views,a.max_seizures_per_view,a.max_ez_channels_per_view,a.max_nez_channels_per_view,a.max_windows_per_phase);cohort=CachedCohort(root,root/'cache_manifest.csv',a.seed,cfg)
 # Cache verification is deliberately independent of a fold manifest.  This
 # also makes it usable on a locally cropped smoke cache.
 if a.verify_only:
  for key in sorted(cohort.keys()):verify_patient(cohort.load(key))
  write_json(out/'cache_verify_audit.json',{'verified_patients':len(cohort.keys()),'model_version':MODEL_VERSION,'cache_version':CACHE_VERSION});print(json.dumps({'verify_only':True,'patients':len(cohort.keys())},indent=2));return 0
 folds=pd.read_csv(a.fold_manifest);selected=sorted(folds.outer_fold.unique()) if a.folds=='all' else [int(x) for x in a.folds.split(',')];mf,protocol,counts=validate_and_protocol(cohort,folds,selected,a.fixed_epochs,a.smoke_test);write_json(out/'true_dualset_protocol_audit.json',protocol);write_csv(out/'cache_patient_validation.csv',counts)
 device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu');probe=TrueEZNEZDualSet(strict=a.strict);audit=model_audit(probe);write_json(out/'true_dualset_model_audit.json',audit);config={'model_version':MODEL_VERSION,'cache_version':CACHE_VERSION,'cache_signature':sig['signature_hash'],'seed':a.seed,'fixed_epochs':a.fixed_epochs,'model_config':{'strict':a.strict},'view_config':cfg.__dict__,'learning_rate':a.learning_rate,'weight_decay':a.weight_decay,'ema_decay':a.ema_decay,'rank_weight':a.rank_weight};config_hash=stable_hash(config);print(json.dumps({'protocol':PROTOCOL,'fixed_epochs':a.fixed_epochs,'final_model':'ema_final',**audit,'checkpoint_root':str(out/'checkpoints'/MODEL_VERSION/f'seed_{a.seed}')},indent=2),flush=True)
 allpred=[];history=[];summary=[];arts={k:[] for k in ('views','seizures','patients','stats','attention')};times=[];wins=0
 for fold in selected:
  test=sorted(mf[mf.outer_fold==fold].patient_key);train=sorted(set(cohort.keys())-set(test));print(f'[TRUE-DUALSET direct outer] fold={fold} train={len(train)} test={len(test)} fixed_epochs={a.fixed_epochs}',flush=True);seed_all(int(a.seed)+int(fold));model=TrueEZNEZDualSet(strict=a.strict).to(device);ema=EMA(model,a.ema_decay);opt=torch.optim.AdamW(model.parameters(),lr=a.learning_rate,weight_decay=a.weight_decay)
  for epoch in range(1,a.fixed_epochs+1):
   stat=train_epoch(model,ema,cohort,train,opt,device,epoch,a);stat.update({'outer_fold':fold,'epoch':epoch,'learning_rate':opt.param_groups[0]['lr']});history.append(stat);times.append(stat['epoch_wall_time_seconds']);wins+=stat['n_windows_encoded'];print(f'[TRUE-DUALSET] fold {fold} epoch {epoch}/{a.fixed_epochs}: loss={stat["train_total"]:.5f} windows={stat["n_windows_encoded"]}',flush=True)
  cp=out/'checkpoints'/MODEL_VERSION/f'seed_{a.seed}'/f'fold_{fold}';cp.mkdir(parents=True,exist_ok=True);meta={'model_version':MODEL_VERSION,'cache_version':CACHE_VERSION,'protocol':PROTOCOL,'outer_fold':fold,'seed':a.seed,'fixed_epochs':a.fixed_epochs,'model_config':{'strict':a.strict},'view_config':cfg.__dict__,'config_hash':config_hash,'uses_inner_validation':False,'uses_early_stopping':False,'outer_test_used_for_selection':False,'uses_true_ez_nez_labels':True,'predicts_localization':False,'uses_task1_checkpoint':False,'uses_recruitment':False,'uses_vdr':False};torch.save({**meta,'selected_model_type':'raw_final','state_dict':model.state_dict()},cp/'final_raw.pt');torch.save({**meta,'selected_model_type':'ema_final','state_dict':ema.model.state_dict()},cp/'final_ema.pt');ck=torch.load(cp/'final_ema.pt',map_location=device,weights_only=False)
  for key in ('model_version','cache_version','config_hash','outer_fold','seed'):
   if ck[key]!=meta[key]:raise RuntimeError(f'checkpoint {key} mismatch')
  if ck['selected_model_type']!='ema_final':raise RuntimeError('outer test must load final EMA')
  final=TrueEZNEZDualSet(strict=a.strict).to(device);final.load_state_dict(ck['state_dict']);pred,art=infer(final,cohort,test,device,a.n_deterministic_views)
  for row in pred:row['outer_fold']=fold
  for name,rows in art.items():
   for row in rows:row['outer_fold']=fold
   arts[name]+=rows
  allpred+=pred;m=metric_row(pd.DataFrame(pred));summary.append({'outer_fold':fold,'train_n':len(train),'test_n':len(test),'fixed_epochs':a.fixed_epochs,'selected_model_type':'ema_final','uses_inner_validation':False,'uses_early_stopping':False,'outer_test_inference_count':1,'test_auroc':m['auroc'],'test_accuracy':m['accuracy'],'test_balanced_accuracy':m['balanced_accuracy'],'test_macro_f1':m['macro_f1'],'loaded_checkpoint_path':str(cp/'final_ema.pt')})
 pred=pd.DataFrame(allpred);fold_metrics=pd.DataFrame([{'outer_fold':f,**metric_row(g)} for f,g in pred.groupby('outer_fold')]);pooled=metric_row(pred);overall={**pooled,'fold_mean_auroc':fold_metrics.auroc.mean(),'fold_std_auroc':fold_metrics.auroc.std(ddof=0),'worst_fold_auroc':fold_metrics.auroc.min()};shortcut,shortcut_metrics=shortcut_oof(counts,mf,selected);warning=bool(np.isfinite(overall.get('auroc',np.nan)) and np.isfinite(shortcut_metrics.get('auroc',np.nan)) and abs(overall['auroc']-shortcut_metrics['auroc'])<.03);shortcut_metrics['warning']='performance may be driven by channel-label composition rather than electrophysiological dynamics' if warning else ''
 boot=bootstrap_ci(pred,2000,a.seed) if not a.smoke_test else pd.DataFrame(columns=['metric','estimate','ci_lower','ci_upper','repeats'])
 write_csv(out/'training_history.csv',history);write_csv(out/'training_summary.csv',summary);write_csv(out/'fold_metrics.csv',fold_metrics);write_csv(out/'summary_metrics.csv',[overall]);write_csv(out/'metrics_by_center.csv',[{'center':c,**metric_row(g)} for c,g in pred.groupby('center')]);write_csv(out/'bootstrap_ci.csv',boot);write_csv(out/'oof_patient_predictions.csv',pred);write_csv(out/'view_level_predictions.csv',arts['views']);write_csv(out/'seizure_embeddings.csv',arts['seizures']);write_csv(out/'patient_embeddings.csv',arts['patients']);write_csv(out/'dualset_statistics.csv',arts['stats']);write_csv(out/'cross_set_attention_audit.csv',arts['attention']);write_csv(out/'misclassified_patients.csv',pred[pred.outcome_true!=pred.prediction_05]);write_csv(out/'label_shortcut_audit.csv',shortcut);write_json(out/'label_shortcut_summary.json',shortcut_metrics);write_json(out/'runtime_profile.json',{'n_windows_encoded':wins,'windows_per_second':wins/max(sum(times),1e-9),'epoch_wall_time_seconds':times,'total_wall_time_seconds':time.time()-started,'gpu_peak_allocated_gb':torch.cuda.max_memory_allocated(device)/2**30 if device.type=='cuda' else 0.});(out/'TRUE_EZ_NEZ_DUALSET_V1_REPORT.md').write_text(f'# {MODEL_VERSION}\n\nDirect outer fixed-epoch result.\n\nShortcut warning: {shortcut_metrics["warning"] or "none"}\n',encoding='utf8');return 0
if __name__=='__main__':raise SystemExit(main())
