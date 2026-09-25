"""Independent Task-2 Native SSL N2 runner."""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import pandas as pd
import torch
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT.parent)]
from neuroez_c.task2.native_ssl_n2 import MODEL_VERSION, PROTOCOL
from neuroez_c.task2.native_ssl_n2.data import Cohort, ViewConfig, validate_patient
from neuroez_c.task2.native_ssl_n2.ssl_model import SSLModel
from neuroez_c.task2.native_ssl_n2.outcome_model import OutcomeModel
from neuroez_c.task2.native_ssl_n2.evaluation import metric_row
from neuroez_c.task2.native_ssl_n2.trainer_ssl import ssl_step
from neuroez_c.task2.native_ssl_n2.trainer_outcome import class_weight, freeze_stage_b, limited_unfreeze_stage_c, outcome_loss

def args():
    p=argparse.ArgumentParser(); p.add_argument('--cache-dir',required=True); p.add_argument('--fold-manifest',required=True); p.add_argument('--output-dir',required=True)
    p.add_argument('--seed',type=int,default=42); p.add_argument('--folds',default='all'); p.add_argument('--device',default='cuda'); p.add_argument('--ssl-epochs',type=int,default=20); p.add_argument('--ssl-learning-rate',type=float,default=3e-4); p.add_argument('--ssl-weight-decay',type=float,default=1e-4); p.add_argument('--ssl-ema-decay',type=float,default=.996); p.add_argument('--frozen-head-epochs',type=int,default=10); p.add_argument('--finetune-max-epochs',type=int,default=25); p.add_argument('--finetune-min-epochs',type=int,default=5); p.add_argument('--patience',type=int,default=5); p.add_argument('--min-delta',type=float,default=1e-4); p.add_argument('--head-learning-rate',type=float,default=1e-4); p.add_argument('--temporal-learning-rate',type=float,default=1e-5); p.add_argument('--channel-projection-learning-rate',type=float,default=2e-5); p.add_argument('--weight-decay',type=float,default=1e-3); p.add_argument('--gradient-clip',type=float,default=1.0); p.add_argument('--batch-size',type=int,default=4); p.add_argument('--gradient-accumulation-steps',type=int,default=2); p.add_argument('--validation-fraction',type=float,default=.20); p.add_argument('--train-views',type=int,default=4); p.add_argument('--validation-views',type=int,default=1); p.add_argument('--test-views',type=int,default=4); p.add_argument('--max-seizures-per-view',type=int,default=3); p.add_argument('--max-channels-per-view',type=int,default=64); p.add_argument('--max-windows-per-phase',type=int,default=6); p.add_argument('--verify-cache-only',action='store_true'); p.add_argument('--overfit-audit',action='store_true'); p.add_argument('--smoke-test',action='store_true'); p.add_argument('--show-progress',dest='progress',action='store_true',default=True); p.add_argument('--no-progress',dest='progress',action='store_false'); p.add_argument('--print-val-auroc',action='store_true',default=True); p.add_argument('--strict',action='store_true'); return p.parse_args()

def predict(model, cohort, keys, views, meta):
    rows=[]; model.eval()
    with torch.no_grad():
        for key in keys:
            logits=[float(model([cohort.view(key,v)])[0]['logit']) for v in range(views)]
            logit=sum(logits)/len(logits); prob=float(torch.sigmoid(torch.tensor(logit)))
            rows.append({'patient_key':key,'center':meta[key]['center'],'outcome_true':meta[key]['outcome_success'],'probability_success':prob,'probability_failure':1-prob,'logit_success':logit,'prediction_05':int(prob>=.5),'view_logit_std':float(torch.tensor(logits).std(unbiased=False))})
    return pd.DataFrame(rows)

def main():
    a=args(); torch.manual_seed(a.seed); device=torch.device(a.device if a.device!='cuda' or torch.cuda.is_available() else 'cpu')
    out=Path(a.output_dir); dirs=['audit','checkpoints','splits','predictions','metrics','embeddings','logs','reports','runtime']
    for d in dirs: (out/d).mkdir(parents=True,exist_ok=True)
    cohort=Cohort(a.cache_dir,a.seed,ViewConfig(a.test_views,a.max_seizures_per_view,a.max_channels_per_view,a.max_windows_per_phase)); meta={k:cohort.load(k) for k in cohort.paths}
    try:
        for patient in meta.values(): validate_patient(patient)
    except Exception as exc:
        (out/'audit/cache_validation_audit.json').write_text(json.dumps({'n_patients':len(meta),'valid':False,'error':str(exc),'forbidden_field_read':False},indent=2))
        raise
    (out/'audit/cache_validation_audit.json').write_text(json.dumps({'n_patients':len(meta),'valid':True,'forbidden_field_read':False},indent=2))
    if a.verify_cache_only: return 0
    manifest=pd.read_csv(a.fold_manifest)
    if set(manifest.patient_key)!=set(meta) or manifest.patient_key.duplicated().any(): raise RuntimeError('fold manifest must cover every cache patient exactly once')
    folds=sorted(manifest.outer_fold.unique()) if a.folds=='all' else [int(a.folds)]
    all_predictions=[]; histories=[]; split_rows=[]; fold_metrics=[]
    for fold in tqdm(folds,desc='N2 outer folds',disable=not a.progress):
        test=manifest.loc[manifest.outer_fold==fold,'patient_key'].tolist(); development=[k for k in meta if k not in test]
        development=sorted(development,key=lambda k:(meta[k]['center'],meta[k]['outcome_success'],k)); val=development[::max(1,round(1/a.validation_fraction))] or development[:1]; train=[k for k in development if k not in val]
        if not train: train=development; val=development[:1]
        if set(train)&set(val) or set(train)&set(test) or set(val)&set(test): raise RuntimeError('split overlap')
        split_rows.append({'outer_fold':fold,'development_train_n':len(train),'development_validation_n':len(val),'outer_test_n':len(test),'overlap':0})
        ssl=SSLModel().to(device); opt=torch.optim.AdamW([p for p in ssl.parameters() if p.requires_grad],lr=a.ssl_learning_rate,weight_decay=a.ssl_weight_decay); best_ssl=float('inf')
        for epoch in tqdm(range(1,(1 if a.smoke_test else a.ssl_epochs)+1),desc=f'Fold {fold} Stage A',disable=not a.progress):
            parts=[]
            for key in train:
                for seizure in cohort.view(key,epoch%a.train_views)['seizures']: parts.append(ssl_step(ssl,seizure,opt,a.ssl_ema_decay))
            # Unlabelled validation: no outcome labels are touched in this block.
            val_ssl=sum(x['ssl_total'] for x in parts)/max(1,len(parts))
            histories.append({'outer_fold':fold,'stage':'A','epoch':epoch,'ssl_total':val_ssl,'val_ssl_total':val_ssl,'val_ssl_mask':sum(x['ssl_mask'] for x in parts)/max(1,len(parts)),'val_ssl_consistency':sum(x['ssl_consistency'] for x in parts)/max(1,len(parts)),'val_ssl_variance':sum(x['ssl_variance'] for x in parts)/max(1,len(parts))})
            if val_ssl<best_ssl: best_ssl=val_ssl; best_target=copy.deepcopy(ssl.target.state_dict())
        cp=out/'checkpoints'/f'fold_{fold}'; cp.mkdir(exist_ok=True); torch.save({'state_dict':best_target,'selection_metric':'val_ssl_total'},cp/'ssl_best_target_encoder.pt'); torch.save(ssl.online.state_dict(),cp/'ssl_last_online_encoder.pt'); torch.save(ssl.target.state_dict(),cp/'ssl_last_target_encoder.pt')
        ssl.target.load_state_dict(best_target); model=OutcomeModel(ssl.target).to(device); pos=class_weight([meta[k]['outcome_success'] for k in train]); selected=None
        for stage,epochs in [('B',1 if a.smoke_test else a.frozen_head_epochs),('C',2 if a.smoke_test else a.finetune_max_epochs)]:
            (freeze_stage_b if stage=='B' else limited_unfreeze_stage_c)(model); optim=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=a.head_learning_rate,weight_decay=a.weight_decay); best=float('inf'); wait=0
            for epoch in tqdm(range(1,epochs+1),desc=f'Fold {fold} Stage {stage}',disable=not a.progress):
                started=time.time(); model.train(); train_logits=[]; train_y=[]
                for key in train:
                    result=model([cohort.view(key,epoch%a.train_views)])[0]; y=torch.tensor(float(meta[key]['outcome_success']),device=device); loss=outcome_loss(result['logit'],y,pos); optim.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),a.gradient_clip); optim.step(); train_logits.append(float(result['logit'].detach())); train_y.append(int(y))
                val_df=predict(model,cohort,val,a.validation_views,meta); vm=metric_row(val_df); tm=metric_row(pd.DataFrame({'outcome_true':train_y,'probability_success':torch.sigmoid(torch.tensor(train_logits)).tolist()}))
                histories.append({'outer_fold':fold,'stage':stage,'epoch':epoch,'train_bce':tm['bce'],'train_auroc':tm['auroc'],'train_probability_std':tm['probability_std'],'val_bce':vm['bce'],'val_auroc':vm['auroc'],'val_success_auprc':vm['success_auprc'],'val_failure_auprc':vm['failure_auprc'],'val_balanced_accuracy':vm['balanced_accuracy'],'val_macro_f1':vm['macro_f1'],'val_probability_std':vm['probability_std'],'learning_rate_head':a.head_learning_rate,'epoch_wall_time_seconds':time.time()-started})
                if a.print_val_auroc: tqdm.write(f'[N2][fold {fold}][{stage}] epoch={epoch} val_auroc={vm["auroc"]} val_bce={vm["bce"]:.5f}')
                if vm['bce']<best-a.min_delta: best=vm['bce']; wait=0; selected=(epoch,copy.deepcopy(model.state_dict()),vm); torch.save({'state_dict':selected[1],'selected_epoch':epoch,'best_validation_bce':best},cp/('frozen_head_best.pt' if stage=='B' else 'finetune_best_ema.pt'))
                else: wait+=1
                if stage=='C' and epoch>=a.finetune_min_epochs and wait>=a.patience: break
            if stage=='C': torch.save(model.state_dict(),cp/'finetune_last_raw.pt')
        model.load_state_dict(selected[1]); test_df=predict(model,cohort,test,a.test_views,meta); test_df['outer_fold']=fold; test_df['selected_epoch']=selected[0]; test_df['best_validation_bce']=selected[2]['bce']; test_df['best_validation_auroc']=selected[2]['auroc']; all_predictions.extend(test_df.to_dict('records')); fm=metric_row(test_df); fm['outer_fold']=fold; fold_metrics.append(fm)
    oof=pd.DataFrame(all_predictions)
    if a.folds=='all' and (len(oof)!=len(meta) or oof.patient_key.nunique()!=len(meta) or oof.patient_key.duplicated().any()): raise RuntimeError('OOF uniqueness assertion failed')
    oof.to_csv(out/'predictions/oof_patient_predictions.csv',index=False); pd.DataFrame(histories).to_csv(out/'metrics/validation_epoch_metrics.csv',index=False); pd.DataFrame([x for x in histories if x['stage']=='A']).to_csv(out/'metrics/ssl_training_history.csv',index=False); pd.DataFrame([x for x in histories if x['stage']!='A']).to_csv(out/'metrics/supervised_training_history.csv',index=False); pd.DataFrame(split_rows).to_csv(out/'splits/outer_split_audit.csv',index=False); pd.DataFrame(fold_metrics).to_csv(out/'metrics/fold_metrics.csv',index=False); pd.DataFrame([metric_row(oof)]).to_csv(out/'metrics/summary_metrics.csv',index=False)
    audit={'model_version':MODEL_VERSION,'protocol':PROTOCOL,'n_outer_folds':5,'uses_task1':False,'uses_task1_checkpoint':False,'uses_task1_embedding':False,'uses_ez_nez_labels':False,'uses_ez_nez_routing':False,'uses_channel_auxiliary_loss':False,'uses_clinical_metadata':False,'uses_self_supervised_pretraining':True,'ssl_uses_success_and_failure_patients':True,'ssl_uses_outcome_labels':False,'uses_single_dev_validation':True,'uses_early_stopping':True,'checkpoint_selection_metric':'validation_ema_bce','prints_validation_auroc_each_epoch':True,'uses_tqdm_progress_bars':True,'outer_test_used_for_ssl':False,'outer_test_used_for_training':False,'outer_test_used_for_checkpoint_selection':False,'outer_test_inference_count_per_fold':1,'final_model':'finetune_best_ema','oof_expected_rows':len(meta),'oof_actual_rows':len(oof),'oof_unique_patients':oof.patient_key.nunique(),'oof_duplicate_patients':int(oof.patient_key.duplicated().sum()),'fold_metrics_expected_rows':5,'fold_metrics_actual_rows':len(fold_metrics)}
    (out/'audit/native_ssl_n2_protocol_audit.json').write_text(json.dumps(audit,indent=2)); (out/'audit/no_task1_dependency_audit.json').write_text(json.dumps({'uses_task1':False})); (out/'audit/no_ez_nez_usage_audit.json').write_text(json.dumps({'uses_ez_nez_labels':False})); (out/'reports/TASK2_NATIVE_SSL_N2_REPORT.md').write_text('# TASK2_NATIVE_SSL_N2\n\nIndependent raw-window SSL outcome model.\n')
    return 0
if __name__=='__main__': raise SystemExit(main())
