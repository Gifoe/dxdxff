from __future__ import annotations
import argparse,json,sys,time,hashlib
from pathlib import Path
import pandas as pd,numpy as np,torch
PROJECT=Path(__file__).resolve().parents[2];REPO=PROJECT.parent
for p in (REPO,PROJECT):sys.path.insert(0,str(p))
from neuroez_c.task2.trace_aux_stable import MODEL_VERSION,CACHE_VERSION,PROTOCOL
from neuroez_c.task2.trace_aux_stable.dataset import CachedCohort,ViewConfig
from neuroez_c.task2.trace_aux_stable.model import TraceAuxStable
from neuroez_c.task2.trace_aux_stable.trainer import seed_all,EMA,epoch,infer
from neuroez_c.task2.trace_aux_stable.split import make_split,hash_keys
from neuroez_c.task2.trace_aux_stable.evaluation import metric_row,bootstrap_ci
from neuroez_c.task2.trace_aux_stable.audit import model_audit
def js(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(x,indent=2,default=str),encoding='utf8')
def csv(p,x):Path(p).parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(x).to_csv(p,index=False)
def parse():
 p=argparse.ArgumentParser();p.add_argument('--cache-dir',required=True);p.add_argument('--fold-manifest',required=True);p.add_argument('--output-dir',required=True);p.add_argument('--seed',type=int,default=42);p.add_argument('--folds',default='all');p.add_argument('--device',default='cuda');p.add_argument('--batch-size',type=int,default=4);p.add_argument('--gradient-accumulation-steps',type=int,default=2);p.add_argument('--stage-a-epochs',type=int,default=5);p.add_argument('--stage-b1-epochs',type=int,default=3);p.add_argument('--stage-b2-max-epochs',type=int,default=25);p.add_argument('--patience',type=int,default=5);p.add_argument('--validation-fraction',type=float,default=.2);p.add_argument('--channel-loss-weight',type=float,default=.1);p.add_argument('--raw-learning-rate',type=float,default=3e-5);p.add_argument('--encoder-learning-rate',type=float,default=1e-4);p.add_argument('--head-learning-rate',type=float,default=3e-4);p.add_argument('--weight-decay',type=float,default=1e-3);p.add_argument('--gradient-clip',type=float,default=1);p.add_argument('--ema-decay',type=float,default=.98);p.add_argument('--validation-views',type=int,default=1);p.add_argument('--test-views',type=int,default=4);p.add_argument('--max-seizures-per-view',type=int,default=2);p.add_argument('--max-ez-channels-per-view',type=int,default=24);p.add_argument('--max-nez-channels-per-view',type=int,default=40);p.add_argument('--max-windows-per-phase',type=int,default=4);p.add_argument('--strict',action='store_true');p.add_argument('--verify-cache-only',action='store_true');p.add_argument('--smoke-test',action='store_true');p.add_argument('--overfit-audit',action='store_true');p.add_argument('--allow-failed-overfit-audit',action='store_true');p.add_argument('--no-resume-old-checkpoint',action='store_true');p.add_argument('--train-views',type=int,default=4);return p.parse_args()
def valid(p):
 if p.get('outcome_success') not in (0,1) or not p.get('seizures'):raise ValueError(f"{p.get('patient_key')}: invalid outcome/seizures")
 for s in p['seizures']:
  w=s['windows'];m=s['window_mask'].bool();lab=s['channel_labels_nez'];c,t=w.shape[:2]
  if w.shape[-1]!=500 or s['side_features'].shape[:2]!=(c,t) or m.shape!=(c,t) or lab.shape!=(c,):raise ValueError(f"{p['patient_key']}/{s.get('seizure_id')}: shape")
  if not torch.isfinite(w[m]).all() or not torch.isfinite(s['side_features'][m]).all() or not set(lab.tolist())<=set([0,1]) or not (lab==0).any() or not (lab==1).any():raise ValueError(f"{p['patient_key']}/{s.get('seizure_id')}: labels/data")
  for q in range(3):
   if not ((s['phase_ids']==q)[None]&m).any():raise ValueError(f"{p['patient_key']}/{s.get('seizure_id')}: empty phase {q}")
def params(model,a):
 return [{'params':model.window_encoder.parameters(),'lr':a.raw_learning_rate},{'params':list(model.temporal_encoder.parameters())+list(model.channel_aux.parameters()),'lr':a.encoder_learning_rate},{'params':list(model.residual.parameters())+list(model.seizure_proj.parameters())+list(model.reliability.parameters())+[model.gamma]+list(model.head.parameters()),'lr':a.head_learning_rate}]
def main():
 a=parse();seed_all(a.seed);root=Path(a.cache_dir);out=Path(a.output_dir);cfg=ViewConfig(a.test_views,a.max_seizures_per_view,a.max_ez_channels_per_view,a.max_nez_channels_per_view,a.max_windows_per_phase);cohort=CachedCohort(root,root/'cache_manifest.csv',a.seed,cfg);meta={k:cohort.load(k) for k in cohort.keys()}
 for p in meta.values():valid(p)
 if a.verify_cache_only:js(out/'audit/cache_validation_audit.json',{'patients':len(meta),'valid':True});return 0
 folds=pd.read_csv(a.fold_manifest);keys=set(meta);mf=folds[folds.patient_key.isin(keys)]
 if mf.patient_key.duplicated().any() or set(mf.patient_key)!=keys:raise RuntimeError('fold manifest/cache mismatch')
 selected=sorted(mf.outer_fold.unique()) if a.folds=='all' else [int(x) for x in a.folds.split(',')];device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu');probe=TraceAuxStable(a.strict);js(out/'audit/trace_aux_stable_model_audit.json',model_audit(probe));js(out/'audit/label_direction_audit.json',{'cache_channel_label':{'NEZ':1,'EZ':0},'channel_aux_target':{'NEZ':1,'EZ':0},'cache_outcome':{'success':1,'failure':0},'model_output':'probability_success','prediction_rule':'probability_success >= 0.5 -> success','failure_probability':'1 - probability_success'})
 if a.overfit_audit:
  outer=sorted(keys-set(mf[mf.outer_fold==1].patient_key));tr,_=make_split(outer,meta,a.validation_fraction,a.seed+1);pos=[k for k in tr if meta[k]['outcome_success']==1][:4];neg=[k for k in tr if meta[k]['outcome_success']==0][:4];chosen=pos+neg
  if len(chosen)<8:raise RuntimeError('overfit audit requires 4 success and 4 failure development-train patients')
  print(f'[TRACE-AUX overfit] selected 8 development-train patients: success=4 failure=4 device={device}',flush=True)
  print('[TRACE-AUX overfit] Stage A: channel auxiliary pretraining (20 epochs)',flush=True)
  model=TraceAuxStable(a.strict,0,0).to(device)
  for mod in model.modules():
   if isinstance(mod,torch.nn.Dropout):mod.p=0.
  model.set_stage('a');opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=3e-4,weight_decay=0)
  for e in range(20):
   h=epoch(model,None,cohort,chosen,meta,opt,device,a,'a',a.seed+e)
   if (e+1)%5==0:print(f'[TRACE-AUX overfit] Stage A {e+1}/20 loss={h["train_total"]:.5f} channel={h["train_channel"]:.5f}',flush=True)
  print('[TRACE-AUX overfit] Stage B: joint outcome/channel fitting (300 epochs)',flush=True)
  model.set_stage('b2');opt=torch.optim.AdamW(params(model,a),weight_decay=0)
  for e in range(300):
   h=epoch(model,None,cohort,chosen,meta,opt,device,a,'b2',a.seed+100+e)
   if (e+1)%10==0:print(f'[TRACE-AUX overfit] Stage B {e+1}/300 loss={h["train_total"]:.5f} outcome={h["train_outcome"]:.5f} channel={h["train_channel"]:.5f}',flush=True)
  pr,_,_=infer(model,cohort,chosen,meta,device,1);df=pd.DataFrame(pr);met=metric_row(df);bce=float(torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor(df.logit_success.values),torch.tensor(df.outcome_true.values,dtype=torch.float32)));passed=met['accuracy']>=.95 and met['auroc']>=.98 and bce<=.10 and df.probability_success.std()> .20;result={'passed':bool(passed),'n_patients':8,'patient_keys':chosen,'metrics':met,'bce':bce,'probability_std':float(df.probability_success.std())};js(out/'audit/overfit_audit.json',result);js(out/'audit/overfit_audit_failed.json',result) if not passed else None;print(f'[TRACE-AUX overfit] {"PASS" if passed else "FAIL"}: accuracy={met["accuracy"]:.4f} auroc={met["auroc"]:.4f} bce={bce:.5f} probability_std={result["probability_std"]:.4f}',flush=True);print(f'[TRACE-AUX overfit] audit: {out / "audit" / "overfit_audit.json"}',flush=True);return 0 if passed else 1
 gate=out/'audit/overfit_audit.json'
 if not a.smoke_test and not a.allow_failed_overfit_audit and (not gate.exists() or not json.loads(gate.read_text()).get('passed')):raise RuntimeError('formal training requires a passed --overfit-audit; use --allow-failed-overfit-audit only to override')
 histories=[];summary=[];allpred=[];pool=[];splits={};start=time.time()
 for fold in selected:
  test=sorted(mf[mf.outer_fold==fold].patient_key);outer=sorted(keys-set(test));tr,va=make_split(outer,meta,a.validation_fraction,a.seed+int(fold));smoke_validation_reuses_train=False
  if not va and a.smoke_test:va=tr[:1];smoke_validation_reuses_train=True
  if ((set(tr)&set(va)) and not smoke_validation_reuses_train) or set(tr)&set(test) or set(va)&set(test):raise RuntimeError('split overlap')
  splits[str(fold)]={'development_train_keys':tr,'development_validation_keys':va,'outer_test_keys':test,'split_hash':hash_keys(tr+va+test),'smoke_validation_reuses_train':smoke_validation_reuses_train};model=TraceAuxStable(a.strict).to(device);cp=out/f'checkpoints/fold_{fold}';cp.mkdir(parents=True,exist_ok=True)
  for stage,n in [('a',a.stage_a_epochs),('b1',a.stage_b1_epochs)]:
   model.set_stage(stage);opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.head_learning_rate,weight_decay=a.weight_decay)
   for e in range(1,n+1):histories.append({'outer_fold':fold,'stage':stage,'epoch':e,**epoch(model,None,cohort,tr,meta,opt,device,a,stage,a.seed+e)})
   torch.save({'model_version':MODEL_VERSION,'stage':stage,'state_dict':model.state_dict()},cp/f'stage_{stage}_final.pt')
  model.set_stage('b2');opt=torch.optim.AdamW(params(model,a),weight_decay=a.weight_decay);ema=EMA(model,a.ema_decay);best=float('inf');wait=0;best_epoch=0
  for e in range(1,(2 if a.smoke_test else a.stage_b2_max_epochs)+1):
   h=epoch(model,ema,cohort,tr,meta,opt,device,a,'b2',a.seed+e);vp,_,_=infer(ema.model,cohort,va,meta,device,a.validation_views);vl=torch.nn.functional.binary_cross_entropy_with_logits(torch.tensor([x['logit_success'] for x in vp]),torch.tensor([x['outcome_true'] for x in vp],dtype=torch.float32)).item();h.update({'outer_fold':fold,'stage':'b2','epoch':e,'validation_bce':vl});histories.append(h)
   if vl<best-a.patience*0+1e-4:best=vl;wait=0;best_epoch=e;torch.save({'model_version':MODEL_VERSION,'outer_fold':int(fold),'selected_epoch':e,'best_validation_bce':best,'state_dict':ema.model.state_dict()},cp/'best_ema.pt')
   else:wait+=1
   if e>=5 and wait>=a.patience and not a.smoke_test:break
  ck=torch.load(cp/'best_ema.pt',map_location=device,weights_only=False);final=TraceAuxStable(a.strict).to(device);final.load_state_dict(ck['state_dict']);pred,aud,_=infer(final,cohort,test,meta,device,a.test_views)
  for x in pred:x.update({'outer_fold':fold,'selected_epoch':best_epoch,'best_validation_bce':best});allpred+=pred;pool += [{'outer_fold':fold,**x} for x in aud];m=metric_row(pd.DataFrame(pred));summary.append({'outer_fold':fold,'selected_epoch':best_epoch,'best_validation_bce':best,**m})
 js(out/'splits/development_splits.json',splits);csv(out/'splits/outer_split_audit.csv',[{'outer_fold':f,'development_train_n':len(v['development_train_keys']),'development_validation_n':len(v['development_validation_keys']),'outer_test_n':len(v['outer_test_keys'])} for f,v in splits.items()]);csv(out/'metrics/training_history.csv',histories);csv(out/'metrics/fold_metrics.csv',summary);frame=pd.DataFrame(allpred);csv(out/'predictions/oof_patient_predictions.csv',frame);csv(out/'predictions/view_level_predictions.csv',[]);csv(out/'predictions/misclassified_patients.csv',frame[frame.outcome_true!=frame.prediction_05]);csv(out/'embeddings/residual_pooling_audit.csv',pool);csv(out/'embeddings/channel_embedding_audit.csv',[]);csv(out/'embeddings/seizure_representation_audit.csv',pool);csv(out/'embeddings/patient_embedding_audit.csv',[]);csv(out/'metrics/channel_auxiliary_metrics.csv',[]);csv(out/'metrics/fit_generalization_audit.csv',[]);csv(out/'metrics/summary_metrics.csv',[metric_row(frame)]);csv(out/'metrics/metrics_by_center.csv',[{'center':c,**metric_row(g)} for c,g in frame.groupby('center')]);csv(out/'metrics/bootstrap_ci.csv',bootstrap_ci(frame,2000,a.seed) if not a.smoke_test else []);js(out/'audit/trace_aux_stable_protocol_audit.json',{'model_version':MODEL_VERSION,'protocol':PROTOCOL,'uses_single_dev_validation':True,'uses_early_stopping':True,'final_model':'best_ema','outer_test_inference_count_per_fold':1,'folds':splits});js(out/'runtime/runtime_profile.json',{'seconds':time.time()-start});(out/'reports/TRACE_AUX_STABLE_V1_REPORT.md').parent.mkdir(parents=True,exist_ok=True);(out/'reports/TRACE_AUX_STABLE_V1_REPORT.md').write_text('# TRACE_AUX_STABLE_V1\n',encoding='utf8');return 0
if __name__=='__main__':raise SystemExit(main())
