from __future__ import annotations
import copy
from pathlib import Path
import torch
from torch.nn.parameter import UninitializedParameter
from torch.utils.data import DataLoader
from ez_dataset import flatten_window_samples
from .dataset import build_examples,collate_patient_ez_batch,feature_args,fit_window_tensor_normalizer
from .io import write_csv,write_json
from .losses import CleanNEZLoss
from .metrics import patient_metrics,select_validation_threshold
from .model import CleanNEZConfig,CleanNEZModel

def _device(name): return torch.device("cuda" if name=="auto" and torch.cuda.is_available() else "cpu" if name=="auto" else name)
def _move(b,d): return {k:(v.to(d) if torch.is_tensor(v) else v) for k,v in b.items()}
def _loader(examples,batch_size,shuffle,num_workers): return DataLoader(examples,batch_size=batch_size,shuffle=shuffle,num_workers=num_workers,collate_fn=collate_patient_ez_batch)
def dry_initialize_model(model,loader,device):
    was=model.training; model.eval(); first=next(iter(loader))
    with torch.no_grad(): model(_move(first,device))
    model.train(was)
    names=[n for n,p in model.named_parameters() if isinstance(p,UninitializedParameter)]
    if names: raise RuntimeError(f"Uninitialized parameters before optimizer creation: {names}")

def predict(model,loader,device):
    model.eval(); rows=[]
    with torch.no_grad():
      for raw in loader:
        o=model(_move(raw,device))
        for i,sid in enumerate(raw["subject_id"]):
          valid_idx=torch.where(o["effective_channel_mask"][i])[0].tolist(); order=sorted(valid_idx,key=lambda c:float(o["candidate_risk"][i,c]),reverse=True); ranks={c:j+1 for j,c in enumerate(order)}
          for c in valid_idx:
            rows.append({"subject_id":sid,"center":raw["center"][i],"channel_id":c,"channel_name":raw["canonical_channels"][i][c],"raw_channel_mask":int(raw["channel_mask"][i,c]),"observed_channel":int(o["observed_channel_mask"][i,c]),"effective_channel":int(o["effective_channel_mask"][i,c]),"true_nez":int(raw["labels_nez"][i,c]>.5),"true_ez":int(raw["labels_ez"][i,c]>.5),"score_nez_probability":float(o["score_nez_probability"][i,c]),"candidate_risk":float(o["candidate_risk"][i,c]),"rank_candidate_risk_desc":ranks[c],**{k:float(o[k][i,c]) for k in ("anchor_distance","anchor_distance_z","early_fast_score","risk_mean_across_seizures","risk_std_across_seizures","risk_median_across_seizures","risk_max_across_seizures","top20_recurrence")},"valid_seizure_count":float(o["valid_seizure_count_per_channel"][i,c]),"post_onset_fallback_count":float(o["post_onset_fallback_count_per_channel"][i,c]),"post_onset_fallback_rate":float(o["post_onset_fallback_rate_per_channel"][i,c]),"early_fast_gate_mean":float(o["early_fast_gate_mean"][i,c]),**{k:float(o[k]) for k in ("a_seizure_early","a_anchor","a_early","a_recurrence")}})
    return rows
def evaluate_loss(model,loader,lossfn,device):
    model.eval(); vals=[]
    with torch.no_grad():
      for raw in loader: vals.append(float(lossfn(model(_move(raw,device)),_move(raw,device))["loss"]))
    return sum(vals)/max(1,len(vals))
def _decorate(rows,fold_idx,split,threshold,best_epoch):
    for r in rows: r.update(fold_idx=fold_idx,split=split,threshold=threshold,predicted_nez=int(r["score_nez_probability"]>=threshold),predicted_ez=int(r["score_nez_probability"]<threshold),best_epoch=best_epoch)

def run_fold(args,fold,run_records,patient_index):
    torch.manual_seed(args.random_seed+fold["fold_idx"]); fa=feature_args(); train,val,test=fold["train_subjects"],fold["val_subjects"],fold["test_subjects"]
    train_samples=flatten_window_samples(run_records,subject_ids=train); norm=fit_window_tensor_normalizer(train_samples,args=fa)
    datasets=[build_examples(flatten_window_samples(run_records,subject_ids=x),patient_index,normalizer=norm,subject_ids=x,args=fa) for x in (train,val,test)]
    loaders=[_loader(d,args.batch_size,i==0,args.num_workers) for i,d in enumerate(datasets)]; device=_device(args.device); model=CleanNEZModel(CleanNEZConfig(args.model_dim,args.num_heads,args.dropout,args.anchor_dim,args.early_pool_frac,args.lse_pool_tau)).to(device); lossfn=CleanNEZLoss()
    dry_initialize_model(model,loaders[0],device); opt=torch.optim.AdamW(model.parameters(),lr=args.learning_rate,weight_decay=args.weight_decay)
    best=None; history=[]; stale=0
    for epoch in range(1,args.epochs+1):
      model.train(); sums={k:0. for k in ("loss","loss_clean_nez","loss_weak_ez_mil","loss_anchor","loss_nez_consistency")}; batches=0
      for raw in loaders[0]:
        b=_move(raw,device); opt.zero_grad(); ls=lossfn(model(b),b); ls["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); opt.step(); batches+=1
        for k in sums:sums[k]+=float(ls[k].detach())
      vr=predict(model,loaders[1],device); threshold,vm=select_validation_threshold(vr); vl=evaluate_loss(model,loaders[1],lossfn,device); weights=[float(torch.nn.functional.softplus(x).detach()) for x in (model.raw_seizure_early,model.raw_anchor_weight,model.raw_early_weight,model.raw_recurrence_weight)]
      key=(vm["patient_macro_f1"],vm["patient_macro_nez_f1"],vm["patient_macro_balanced_accuracy"],-abs(threshold-.5),-vl)
      history.append({"epoch":epoch,"train_total_loss":sums["loss"]/batches,"train_clean_nez_loss":sums["loss_clean_nez"]/batches,"train_weak_ez_mil_loss":sums["loss_weak_ez_mil"]/batches,"train_anchor_loss":sums["loss_anchor"]/batches,"train_consistency_loss":sums["loss_nez_consistency"]/batches,"validation_loss":vl,"validation_patient_macro_f1":vm["patient_macro_f1"],"validation_patient_macro_nez_f1":vm["patient_macro_nez_f1"],"validation_patient_macro_ez_f1":vm["patient_macro_ez_f1"],"validation_patient_macro_balanced_accuracy":vm["patient_macro_balanced_accuracy"],"validation_threshold":threshold,"a_seizure_early":weights[0],"a_anchor":weights[1],"a_early":weights[2],"a_recurrence":weights[3]})
      if best is None or key>best[0]: best=(key,copy.deepcopy(model.state_dict()),threshold,vm,epoch,vr); stale=0
      else: stale+=1
      if epoch>=args.min_epochs_before_early_stop and stale>=args.patience: break
    model.load_state_dict(best[1]); threshold=best[2]; _decorate(best[5],fold["fold_idx"],"validation",threshold,best[4]); tr=predict(model,loaders[2],device); _decorate(tr,fold["fold_idx"],"test",threshold,best[4])
    out=Path(args.output_dir)/f"fold_{fold['fold_idx']}"; out.mkdir(parents=True,exist_ok=True); torch.save({"model_state_dict":best[1],"config":vars(args)},out/"best_checkpoint.pt")
    meta={**fold,"validation_subjects":val,"normalizer_fit_subjects":train,"threshold_selection_subjects":val,"checkpoint_selection_subjects":val,"threshold":threshold,"threshold_source":"validation","best_epoch":best[4]}
    write_csv(out/"training_history.csv",history); write_csv(out/"validation_predictions.csv",best[5]); write_csv(out/"test_predictions.csv",tr); write_json(out/"threshold.json",{"threshold":threshold,"source":"validation","best_epoch":best[4]}); metrics=patient_metrics(tr,None); weights=[float(torch.nn.functional.softplus(x).detach()) for x in (model.raw_seizure_early,model.raw_anchor_weight,model.raw_early_weight,model.raw_recurrence_weight)]; metrics.update(dict(zip(("a_seizure_early","a_anchor","a_early","a_recurrence"),weights))); write_json(out/"fold_metrics.json",metrics); write_json(out/"fold_config.json",meta)
    return tr,metrics,meta
__all__=["run_fold","predict","dry_initialize_model"]
