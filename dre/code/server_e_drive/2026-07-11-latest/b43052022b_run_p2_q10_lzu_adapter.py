#!/usr/bin/env python3
"""Optimize the frozen P2-Q10 LZU-only bounded residual adapter."""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd
import torch
from torch.optim import AdamW

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neuroez_c.p2_q10_lzu_adapter import P2Q10LZUBoundedAdapter, apply_p2_q10_lzu_adapter
from neuroez_c.p2_q10_lzu_adapter_loss import compute_p2_lzu_adapter_loss
from neuroez_c.p2_q10_lzu_adapter_protocol import (
    build_frozen_adapter_features, canonicalize_channel_frame, evaluate_channel_ledger,
    frozen_threshold, non_lzu_invariance_audit, split_lzu_fit_validation,
)
from neuroez_c.p2_q10_lzu_patient_batch import PatientAdapterDataset, PatientBatchSampler, collate_complete_patients


def _seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def _tensor(frame: pd.DataFrame, column: str, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.tensor(frame[column].to_numpy(), dtype=dtype)


def _parameter_norm(parameters) -> float:
    values=[parameter.detach().square().sum() for parameter in parameters]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.0


def _gradient_norm(parameters) -> float:
    values=[parameter.grad.detach().square().sum() for parameter in parameters if parameter.grad is not None]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.0


def _state_update_norm(initial: dict[str,torch.Tensor], final: dict[str,torch.Tensor], names: list[str] | None = None) -> float:
    keys=names if names is not None else list(initial)
    values=[(final[key].detach()-initial[key].detach()).square().sum() for key in keys]
    return float(torch.sqrt(torch.stack(values).sum())) if values else 0.0


def _checkpoint_parameter_count(path: str) -> int | None:
    try:
        payload=torch.load(path,map_location="cpu",weights_only=False)
        state=payload.get("model_state_dict",payload.get("state_dict",payload)) if isinstance(payload,dict) else {}
        return int(sum(value.numel() for value in state.values() if isinstance(value,torch.Tensor))) if isinstance(state,dict) else None
    except Exception:
        return None


def _adapt(model: P2Q10LZUBoundedAdapter, frame: pd.DataFrame, *, is_lzu: bool | None = None):
    features,_=build_frozen_adapter_features(frame); base=_tensor(frame,"base_nez_logit")
    gate=torch.ones(len(frame),dtype=torch.bool) if is_lzu is True else torch.zeros(len(frame),dtype=torch.bool) if is_lzu is False else torch.tensor(frame.center.eq("lzu").to_numpy(),dtype=torch.bool)
    delta=model(features.detach(),gate); return base,delta,apply_p2_q10_lzu_adapter(base,delta,gate)


def _add_predictions(frame: pd.DataFrame, model: P2Q10LZUBoundedAdapter, threshold: float) -> pd.DataFrame:
    model.eval()
    with torch.no_grad(): base,delta,adapted=_adapt(model,frame)
    result=frame.copy(); base_np,delta_np,adapted_np=base.numpy(),delta.numpy(),adapted.numpy()
    result["base_nez_logit"]=base_np; result["adapted_nez_logit"]=adapted_np; result["adapter_delta"]=delta_np
    result["adapter_delta_applied"]=np.where(result.center.eq("lzu"),delta_np,0.0)
    result["base_score_nez"]=1/(1+np.exp(-base_np)); result["adapted_score_nez"]=1/(1+np.exp(-adapted_np))
    result["base_score_ez"]=1-result.base_score_nez; result["adapted_score_ez"]=1-result.adapted_score_nez
    result["frozen_threshold"]=threshold; result["selected_threshold"]=threshold
    result["base_pred_nez"]=(result.base_score_nez>=threshold).astype(int); result["adapted_pred_nez"]=(result.adapted_score_nez>=threshold).astype(int)
    result["base_pred_ez"]=1-result.base_pred_nez; result["adapted_pred_ez"]=1-result.adapted_pred_nez
    result["is_lzu"]=result.center.eq("lzu").astype(int); result["decision_rule"]="frozen_p2_q10_global_threshold_with_lzu_bounded_residual"
    result["threshold_source"]="frozen_p2_q10_outer_validation"; result["threshold_refit_after_adapter"]=False
    result["center_specific_threshold"]=False; result["patient_specific_threshold"]=False; result["center_specific_residual"]=True
    result["center_used_as_model_input"]=False; result["center_used_for_hard_gate"]=True; result["true_count_used_for_prediction"]=False
    result["analysis_status"]="PRIMARY_LEGAL_DOMAIN_ADAPTATION"
    return result


def _residual_stats(delta: np.ndarray, max_delta: float) -> dict[str,float]:
    absolute=np.abs(np.asarray(delta,dtype=float))
    if not len(absolute):
        return {key:0.0 for key in ("mean_abs_adapter_delta","median_abs_adapter_delta","p90_abs_adapter_delta","p95_abs_adapter_delta","max_abs_adapter_delta","positive_delta_fraction","negative_delta_fraction","saturation_rate","fraction_abs_delta_gt_0_001","fraction_abs_delta_gt_0_005","fraction_abs_delta_gt_0_010")}
    return {"mean_abs_adapter_delta":float(absolute.mean()),"median_abs_adapter_delta":float(np.median(absolute)),"p90_abs_adapter_delta":float(np.quantile(absolute,.90)),"p95_abs_adapter_delta":float(np.quantile(absolute,.95)),"max_abs_adapter_delta":float(absolute.max()),"positive_delta_fraction":float((delta>0).mean()),"negative_delta_fraction":float((delta<0).mean()),"saturation_rate":float((absolute>=.95*max_delta).mean()),"fraction_abs_delta_gt_0_001":float((absolute>.001).mean()),"fraction_abs_delta_gt_0_005":float((absolute>.005).mean()),"fraction_abs_delta_gt_0_010":float((absolute>.010).mean())}


def _loss_components(frame: pd.DataFrame, model: P2Q10LZUBoundedAdapter, args) -> tuple[dict[str,float],pd.DataFrame]:
    predicted=_add_predictions(frame,model,frozen_threshold(frame,int(frame.outer_fold.iloc[0])))
    adapted=torch.tensor(predicted.adapted_nez_logit.to_numpy(float),dtype=torch.float32)
    labels=_tensor(predicted,"label_nez"); patient_index=torch.tensor(pd.factorize(predicted.subject_id)[0],dtype=torch.long)
    delta=torch.tensor(predicted.adapter_delta.to_numpy(float),dtype=torch.float32)
    components=compute_p2_lzu_adapter_loss(adapted,labels,patient_index,delta,balanced_bce_weight=args.p2_lzu_adapter_balanced_bce_weight,unweighted_bce_weight=args.p2_lzu_adapter_unweighted_bce_weight,pairwise_weight=args.p2_lzu_adapter_pairwise_weight,residual_weight=args.p2_lzu_adapter_residual_weight)
    return {key:float(value.detach()) for key,value in components.items()},predicted


def _validation_metrics(frame: pd.DataFrame, model: P2Q10LZUBoundedAdapter, threshold: float, args) -> dict[str,float]:
    components,predicted=_loss_components(frame,model,args)
    _,formal_summary,_,_=evaluate_channel_ledger(predicted,logit_column="adapted_nez_logit")
    _,truek_summary,_,_=evaluate_channel_ledger(predicted,logit_column="adapted_nez_logit",truek=True)
    summary=formal_summary.iloc[0]; diagnostic=truek_summary.iloc[0]; stats=_residual_stats(predicted.adapter_delta.to_numpy(float),args.p2_lzu_adapter_max_delta)
    changes=int((predicted.adapted_pred_nez!=predicted.base_pred_nez).sum())
    return {"val_total_loss":components["total_loss"],"val_classification_loss":args.p2_lzu_adapter_balanced_bce_weight*components["patient_balanced_bce"]+args.p2_lzu_adapter_unweighted_bce_weight*components["patient_mean_unweighted_bce"],"val_balanced_bce":components["patient_balanced_bce"],"val_unweighted_bce":components["patient_mean_unweighted_bce"],"val_pairwise_loss":components["pairwise_loss"],"val_residual_loss":components["residual_loss"],"val_lzu_formal_macro_f1":float(summary.patient_macro_f1),"val_lzu_ez_f1":float(summary.patient_macro_ez_f1),"val_lzu_nez_f1":float(summary.patient_macro_nez_f1),"val_lzu_ez_auprc":float(summary.patient_macro_auprc_ez),"val_lzu_ez_mrr":float(summary.patient_macro_ez_mrr),"val_lzu_truek_macro_f1":float(diagnostic.patient_macro_f1),"val_mean_abs_delta":stats["mean_abs_adapter_delta"],"val_max_abs_delta":stats["max_abs_adapter_delta"],"val_fraction_abs_delta_gt_0_005":stats["fraction_abs_delta_gt_0_005"],"val_fraction_abs_delta_gt_0_010":stats["fraction_abs_delta_gt_0_010"],"val_prediction_changes_vs_base":changes,"val_saturation_rate":stats["saturation_rate"]}


def _selection_key(metrics: dict[str,float], epoch: int) -> tuple:
    return (metrics["val_total_loss"],-metrics["val_lzu_formal_macro_f1"],-metrics["val_lzu_ez_auprc"],-metrics["val_lzu_truek_macro_f1"],-metrics["val_lzu_ez_f1"],metrics["val_mean_abs_delta"],epoch)


def _checkpoint_eligible(epoch: int, cumulative_steps: int, min_epochs: int, min_steps: int) -> bool:
    return int(epoch)>=int(min_epochs) and int(cumulative_steps)>=int(min_steps)


def _optimization_status(total_steps: int, min_steps: int, output_norm: float, update_norm: float, train_stats: dict) -> tuple[str,bool]:
    if total_steps<min_steps: return "OPTIMIZATION_INSUFFICIENT_STEPS",False
    passed=output_norm>1e-3 and update_norm>1e-3 and train_stats["mean_abs_adapter_delta"]>.002 and train_stats["fraction_abs_delta_gt_0_001"]>.05
    return ("OPTIMIZATION_EFFECTIVE" if passed else "OPTIMIZATION_NO_OP"),passed


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--p2_q10_root",required=True); parser.add_argument("--base_manifest",required=True); parser.add_argument("--window_cache_path",required=True)
    parser.add_argument("--allowed_subjects_ledger",required=True); parser.add_argument("--fixed_fold_manifest",required=True); parser.add_argument("--require_n_patients",type=int,default=80); parser.add_argument("--output_dir",required=True)
    parser.add_argument("--p2_lzu_adapter_hidden_dim",type=int,default=16); parser.add_argument("--p2_lzu_adapter_dropout",type=float,default=.10); parser.add_argument("--p2_lzu_adapter_max_delta",type=float,default=.30)
    parser.add_argument("--p2_lzu_adapter_lr",type=float,default=None,help="Deprecated compatibility alias; used for both groups only when explicit group LRs are omitted")
    parser.add_argument("--p2_lzu_adapter_trunk_lr",type=float,default=None); parser.add_argument("--p2_lzu_adapter_output_lr",type=float,default=None); parser.add_argument("--p2_lzu_adapter_weight_decay",type=float,default=1e-4)
    parser.add_argument("--p2_lzu_adapter_epochs",type=int,default=40); parser.add_argument("--p2_lzu_adapter_patience",type=int,default=10); parser.add_argument("--p2_lzu_adapter_min_epochs",type=int,default=10); parser.add_argument("--p2_lzu_adapter_min_optimizer_steps",type=int,default=100)
    parser.add_argument("--p2_lzu_adapter_patient_batch_size",type=int,default=None); parser.add_argument("--p2_lzu_adapter_batch_size",type=int,default=None,help="Compatibility alias: number of complete patients per optimizer batch")
    parser.add_argument("--p2_lzu_adapter_balanced_bce_weight",type=float,default=.60); parser.add_argument("--p2_lzu_adapter_unweighted_bce_weight",type=float,default=.40); parser.add_argument("--p2_lzu_adapter_pairwise_weight",type=float,default=.02); parser.add_argument("--p2_lzu_adapter_residual_weight",type=float,default=.02)
    parser.add_argument("--random_seed",type=int,default=42); parser.add_argument("--max_outer_folds",type=int,default=0); parser.add_argument("--dry_run",action="store_true")
    args=parser.parse_args(); output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True)
    patient_batch_size=args.p2_lzu_adapter_patient_batch_size or args.p2_lzu_adapter_batch_size or 2
    if args.p2_lzu_adapter_lr is not None and args.p2_lzu_adapter_trunk_lr is None and args.p2_lzu_adapter_output_lr is None:
        trunk_lr=output_lr=float(args.p2_lzu_adapter_lr)
    else:
        trunk_lr=float(args.p2_lzu_adapter_trunk_lr if args.p2_lzu_adapter_trunk_lr is not None else 1e-3); output_lr=float(args.p2_lzu_adapter_output_lr if args.p2_lzu_adapter_output_lr is not None else 3e-3)
    base=json.loads(Path(args.base_manifest).read_text(encoding="utf-8"))
    if not Path(args.window_cache_path).is_file(): raise FileNotFoundError(f"Feature cache not found: {args.window_cache_path}")
    if base.get("P2_profile")!="P2_TEMPORAL_Q10": raise RuntimeError("Adapter only accepts a P2_TEMPORAL_Q10 base manifest")
    if len(base.get("per_fold",[]))!=5: raise RuntimeError("Base manifest must contain all five audited outer folds")
    dry={"status":"passed","mode":"dry_run","patient_batch_size":patient_batch_size,"trunk_lr":trunk_lr,"output_lr":output_lr,"max_delta":args.p2_lzu_adapter_max_delta,"epochs":args.p2_lzu_adapter_epochs,"min_epochs":args.p2_lzu_adapter_min_epochs,"min_optimizer_steps":args.p2_lzu_adapter_min_optimizer_steps,"threshold_refit_after_adapter":False}
    if args.dry_run: print(json.dumps(dry,indent=2)); return

    split_rows=[]; batch_rows=[]; step_rows=[]; epoch_rows=[]; selection_rows=[]; test_ledgers=[]; residual_rows=[]; parameter_rows=[]; optimization_folds=[]
    for fold in range(1,(args.max_outer_folds or 5)+1):
        meta=next((item for item in base["per_fold"] if int(item["outer_fold"])==fold),None)
        if not meta or meta.get("status")!="passed": raise RuntimeError(f"Fold {fold} is not an audited P2-Q10 base fold")
        _seed(args.random_seed+fold)
        fit=canonicalize_channel_frame(pd.read_csv(meta["fit_path"])); validation=canonicalize_channel_frame(pd.read_csv(meta["validation_path"])); test=canonicalize_channel_frame(pd.read_csv(meta["test_path"]))
        for frame in (fit,validation,test): frame["outer_fold"]=fold
        threshold=frozen_threshold(validation,fold)
        if abs(threshold-frozen_threshold(test,fold))>1e-15: raise RuntimeError(f"fold {fold}: frozen threshold changed")
        fit_lzu,validation_lzu=split_lzu_fit_validation(fit,validation,fold); test_lzu=test[test.center.eq("lzu")]
        for split,frame in (("outer_fit",fit),("adapter_fit",fit_lzu),("outer_validation",validation),("adapter_validation",validation_lzu),("outer_test",test)):
            for subject,group in frame.groupby("subject_id"):
                split_rows.append({"outer_fold":fold,"subject_id":subject,"center":group.center.iloc[0],"split":split,"n_channels":len(group),"n_ez":int((group.label_nez==0).sum()),"n_nez":int((group.label_nez==1).sum()),"included_in_adapter_training":split=="adapter_fit","included_in_adapter_validation":split=="adapter_validation"})
        dataset=PatientAdapterDataset(fit_lzu); input_dim=dataset[0].features.shape[1]; embedding_dim=input_dim-7
        model=P2Q10LZUBoundedAdapter(input_dim,args.p2_lzu_adapter_hidden_dim,args.p2_lzu_adapter_dropout,args.p2_lzu_adapter_max_delta)
        initial_state=copy.deepcopy(model.state_dict()); trunk_parameters=list(model.trunk_parameters()); output_parameters=list(model.output_parameters())
        optimizer=AdamW([{"params":trunk_parameters,"lr":trunk_lr,"weight_decay":args.p2_lzu_adapter_weight_decay,"group_name":"adapter_trunk"},{"params":output_parameters,"lr":output_lr,"weight_decay":args.p2_lzu_adapter_weight_decay,"group_name":"adapter_output"}])
        groups=[{"group_name":group["group_name"],"learning_rate":group["lr"],"weight_decay":group["weight_decay"],"n_parameters":sum(parameter.numel() for parameter in group["params"])} for group in optimizer.param_groups]
        shared_count=_checkpoint_parameter_count(meta["checkpoint_path"])
        parameter_rows.append({"outer_fold":fold,"n_total_shared_parameters":shared_count,"n_trainable_shared_parameters":0,"n_total_adapter_parameters":sum(parameter.numel() for parameter in model.parameters()),"n_trainable_adapter_parameters":sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),"optimizer_shared_parameter_count":0,"optimizer_parameter_groups":groups,"shared_outputs_detached":True})
        fold_out=output/f"fold_{fold}"; fold_out.mkdir(exist_ok=True)
        (fold_out/"optimizer_groups.json").write_text(json.dumps(groups,indent=2),encoding="utf-8")
        best_state=None; best_key=None; best_epoch=0; best_step=0; best_metrics=None; wait=0; cumulative_steps=0
        for epoch in range(1,args.p2_lzu_adapter_epochs+1):
            _seed(args.random_seed+fold*10000+epoch)
            sampler=PatientBatchSampler(dataset,patient_batch_size,global_seed=args.random_seed,outer_fold=fold,epoch=epoch)
            expected_steps=math.ceil(len(dataset)/patient_batch_size); epoch_step_rows=[]; seen=[]
            model.train()
            for batch_index,indices in enumerate(sampler,1):
                examples=[dataset[index] for index in indices]; batch=collate_complete_patients(examples); seen.extend(batch["subject_ids"])
                features=batch["features"].detach(); base_logit=batch["base_nez_logit"]; labels=batch["label_nez"]; patient_index=batch["patient_index"]
                gate=torch.ones(len(base_logit),dtype=torch.bool); delta=model(features,gate); adapted=apply_p2_q10_lzu_adapter(base_logit,delta,gate)
                components=compute_p2_lzu_adapter_loss(adapted,labels,patient_index,delta,balanced_bce_weight=args.p2_lzu_adapter_balanced_bce_weight,unweighted_bce_weight=args.p2_lzu_adapter_unweighted_bce_weight,pairwise_weight=args.p2_lzu_adapter_pairwise_weight,residual_weight=args.p2_lzu_adapter_residual_weight)
                optimizer.zero_grad(); components["total_loss"].backward()
                before=_gradient_norm(model.parameters()); trunk_gradient=_gradient_norm(trunk_parameters); output_gradient=_gradient_norm(output_parameters)
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); after=_gradient_norm(model.parameters()); optimizer.step(); cumulative_steps+=1
                stats=_residual_stats(delta.detach().numpy(),args.p2_lzu_adapter_max_delta)
                row={"outer_fold":fold,"epoch":epoch,"batch_index":batch_index,"subject_ids":"|".join(batch["subject_ids"]),"n_batch_patients":batch["n_batch_patients"],"n_batch_channels":batch["n_batch_channels"],"optimizer_step":cumulative_steps,"train_total_loss":float(components["total_loss"].detach()),"train_balanced_bce":float(components["patient_balanced_bce"].detach()),"train_unweighted_bce":float(components["patient_mean_unweighted_bce"].detach()),"train_pairwise_loss":float(components["pairwise_loss"].detach()),"train_residual_loss":float(components["residual_loss"].detach()),"total_gradient_norm_before_clip":before,"total_gradient_norm_after_clip":after,"trunk_gradient_norm":trunk_gradient,"output_layer_gradient_norm":output_gradient,"trunk_parameter_norm":_parameter_norm(trunk_parameters),"output_layer_weight_norm":float(model.output_layer.weight.detach().norm()),"output_layer_bias_abs":float(model.output_layer.bias.detach().abs().max()),"mean_abs_batch_delta":stats["mean_abs_adapter_delta"],"max_abs_batch_delta":stats["max_abs_adapter_delta"],"fraction_abs_delta_gt_0_001":stats["fraction_abs_delta_gt_0_001"],"fraction_abs_delta_gt_0_005":stats["fraction_abs_delta_gt_0_005"],"fraction_abs_delta_gt_0_010":stats["fraction_abs_delta_gt_0_010"]}
                step_rows.append(row); epoch_step_rows.append(row)
                for subject in batch["subject_ids"]: batch_rows.append({"outer_fold":fold,"epoch":epoch,"batch_index":batch_index,"subject_id":subject,"patient_batch_size":patient_batch_size})
            if len(epoch_step_rows)!=expected_steps or sorted(seen)!=sorted(example.subject_id for example in dataset): raise RuntimeError(f"fold {fold} epoch {epoch}: patient batch coverage mismatch")
            metrics=_validation_metrics(validation_lzu,model,threshold,args)
            aggregate={"outer_fold":fold,"epoch":epoch,"n_fit_lzu_patients":len(dataset),"patient_batch_size":patient_batch_size,"optimizer_steps_this_epoch":len(epoch_step_rows),"cumulative_optimizer_steps":cumulative_steps,"mean_train_total_loss":float(np.mean([row["train_total_loss"] for row in epoch_step_rows])),"mean_train_balanced_bce":float(np.mean([row["train_balanced_bce"] for row in epoch_step_rows])),"mean_train_unweighted_bce":float(np.mean([row["train_unweighted_bce"] for row in epoch_step_rows])),"mean_train_pairwise_loss":float(np.mean([row["train_pairwise_loss"] for row in epoch_step_rows])),"mean_train_residual_loss":float(np.mean([row["train_residual_loss"] for row in epoch_step_rows])),"mean_gradient_norm":float(np.mean([row["total_gradient_norm_before_clip"] for row in epoch_step_rows])),"max_gradient_norm":float(np.max([row["total_gradient_norm_before_clip"] for row in epoch_step_rows])),"mean_output_gradient_norm":float(np.mean([row["output_layer_gradient_norm"] for row in epoch_step_rows])),"mean_abs_train_delta":float(np.mean([row["mean_abs_batch_delta"] for row in epoch_step_rows])),"max_abs_train_delta":float(np.max([row["max_abs_batch_delta"] for row in epoch_step_rows])),**metrics,"diagnostic_composite_score":.50*metrics["val_lzu_formal_macro_f1"]+.20*metrics["val_lzu_ez_auprc"]+.15*metrics["val_lzu_truek_macro_f1"]+.10*metrics["val_lzu_ez_f1"]+.05*metrics["val_lzu_ez_mrr"]-.05*metrics["val_saturation_rate"],"checkpoint_eligible":_checkpoint_eligible(epoch,cumulative_steps,args.p2_lzu_adapter_min_epochs,args.p2_lzu_adapter_min_optimizer_steps),"selected_checkpoint":False}
            epoch_rows.append(aggregate); selection_rows.append(dict(aggregate))
            key=_selection_key(metrics,epoch)
            if aggregate["checkpoint_eligible"]:
                if best_key is None or key<best_key:
                    best_key=key; best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; best_step=cumulative_steps; best_metrics=metrics.copy(); wait=0
                else: wait+=1
                if wait>=args.p2_lzu_adapter_patience: break
        if best_state is None:
            # Persist the failure audit before refusing to interpret an under-optimized fold.
            best_state=copy.deepcopy(model.state_dict()); best_epoch=epoch; best_step=cumulative_steps; best_metrics=_validation_metrics(validation_lzu,model,threshold,args)
        model.load_state_dict(best_state)
        fold_selection=[row for row in selection_rows if row["outer_fold"]==fold]
        for row in fold_selection: row["selected_checkpoint"]=row["epoch"]==best_epoch
        if sum(bool(row["selected_checkpoint"]) for row in fold_selection)!=1: raise RuntimeError(f"fold {fold}: expected exactly one selected checkpoint")
        fit_pred=_add_predictions(fit_lzu,model,threshold); val_pred=_add_predictions(validation_lzu,model,threshold); test_pred=_add_predictions(test,model,threshold); test_ledgers.append(test_pred)
        train_stats=_residual_stats(fit_pred.adapter_delta.to_numpy(float),args.p2_lzu_adapter_max_delta); val_stats=_residual_stats(val_pred.adapter_delta.to_numpy(float),args.p2_lzu_adapter_max_delta); test_stats=_residual_stats(test_pred.loc[test_pred.center.eq("lzu"),"adapter_delta"].to_numpy(float),args.p2_lzu_adapter_max_delta)
        final_state=model.state_dict(); trunk_names=[name for name in initial_state if not name.startswith("net.4")]; output_names=[name for name in initial_state if name.startswith("net.4")]
        total_update=_state_update_norm(initial_state,final_state); trunk_update=_state_update_norm(initial_state,final_state,trunk_names); output_update=_state_update_norm(initial_state,final_state,output_names); output_norm=float(model.output_layer.weight.detach().norm())
        status,passed=_optimization_status(cumulative_steps,args.p2_lzu_adapter_min_optimizer_steps,output_norm,total_update,train_stats)
        saturation_status="RESIDUAL_SATURATION_FAILURE" if test_stats["saturation_rate"]>=.20 else "RESIDUAL_SATURATION_WARNING" if test_stats["saturation_rate"]>=.05 else "OK"
        audit={"outer_fold":fold,"n_fit_lzu_patients":int(fit_lzu.subject_id.nunique()),"n_validation_lzu_patients":int(validation_lzu.subject_id.nunique()),"n_test_lzu_patients":int(test_lzu.subject_id.nunique()),"patient_batch_size":patient_batch_size,"total_optimizer_steps":cumulative_steps,"min_optimizer_steps_required":args.p2_lzu_adapter_min_optimizer_steps,"best_epoch":best_epoch,"best_optimizer_step":best_step,"best_val_total_loss":float(best_metrics["val_total_loss"]),"selection_primary_metric":"val_total_loss","selection_tie_break_reason":"continuous_loss_then_f1_auprc_truek_ezf1_delta_epoch","total_parameter_update_norm":total_update,"trunk_parameter_update_norm":trunk_update,"output_parameter_update_norm":output_update,"output_layer_weight_norm":output_norm,"output_layer_bias_abs":float(model.output_layer.bias.detach().abs().max()),"mean_abs_train_delta":train_stats["mean_abs_adapter_delta"],"mean_abs_validation_delta":val_stats["mean_abs_adapter_delta"],"mean_abs_test_lzu_delta":test_stats["mean_abs_adapter_delta"],"fraction_train_abs_delta_gt_0_001":train_stats["fraction_abs_delta_gt_0_001"],"fraction_train_abs_delta_gt_0_005":train_stats["fraction_abs_delta_gt_0_005"],"fraction_validation_abs_delta_gt_0_005":val_stats["fraction_abs_delta_gt_0_005"],"fraction_test_abs_delta_gt_0_005":test_stats["fraction_abs_delta_gt_0_005"],"validation_prediction_changes":int((val_pred.adapted_pred_nez!=val_pred.base_pred_nez).sum()),"test_lzu_prediction_changes":int((test_pred.loc[test_pred.center.eq("lzu"),"adapted_pred_nez"]!=test_pred.loc[test_pred.center.eq("lzu"),"base_pred_nez"]).sum()),"optimization_status":status,"optimization_passed":passed,"saturation_status":saturation_status}
        optimization_folds.append(audit); residual_rows.append({"outer_fold":fold,**test_stats})
        checkpoint={"adapter_state_dict":model.state_dict(),"outer_fold":fold,"frozen_p2_checkpoint":meta["checkpoint_path"],"frozen_threshold":threshold,"fold_seed":args.random_seed+fold,"input_dim":input_dim,"embedding_dim":embedding_dim,"label_semantics":"NEZ=1,EZ=0","threshold_refit_after_adapter":False,"best_epoch":best_epoch,"best_optimizer_step":best_step,"best_val_total_loss":best_metrics["val_total_loss"]}
        torch.save(checkpoint,fold_out/"best_p2_q10_lzu_adapter.pt")
        (fold_out/"optimization_audit.json").write_text(json.dumps(audit,indent=2),encoding="utf-8"); (fold_out/"selected_checkpoint.json").write_text(json.dumps({"best_epoch":best_epoch,"best_cumulative_optimizer_steps":best_step,"best_val_total_loss":best_metrics["val_total_loss"],"selection_primary_metric":"val_total_loss"},indent=2),encoding="utf-8")
    ledger=pd.concat(test_ledgers,ignore_index=True); ledger.to_csv(output/"p2_lzu_adapter_oof_channel_ledger.csv",index=False)
    pd.DataFrame(split_rows).to_csv(output/"p2_lzu_adapter_split_audit.csv",index=False); pd.DataFrame(batch_rows).to_csv(output/"p2_lzu_adapter_patient_batch_audit.csv",index=False); pd.DataFrame(step_rows).to_csv(output/"p2_lzu_adapter_optimizer_step_audit.csv",index=False); pd.DataFrame(epoch_rows).to_csv(output/"p2_lzu_adapter_training_components.csv",index=False); pd.DataFrame(selection_rows).to_csv(output/"p2_lzu_adapter_checkpoint_selection.csv",index=False); pd.DataFrame(residual_rows).to_csv(output/"p2_lzu_adapter_residual_diagnostics.csv",index=False)
    optimization={"folds":optimization_folds,"all_folds_optimization_passed":all(row["optimization_passed"] for row in optimization_folds),"optimization_status":"OPTIMIZATION_EFFECTIVE" if all(row["optimization_passed"] for row in optimization_folds) else "OPTIMIZATION_INSUFFICIENT_STEPS" if any(row["optimization_status"]=="OPTIMIZATION_INSUFFICIENT_STEPS" for row in optimization_folds) else "OPTIMIZATION_NO_OP"}
    (output/"p2_lzu_adapter_optimization_audit.json").write_text(json.dumps(optimization,indent=2),encoding="utf-8")
    (output/"p2_lzu_adapter_parameter_audit.json").write_text(json.dumps({"folds":parameter_rows,"n_trainable_shared_parameters":0,"optimizer_shared_parameter_count":0,"shared_model_frozen":True,"optimizer_contains_only_adapter_parameters":True},indent=2),encoding="utf-8")
    invariance=non_lzu_invariance_audit(ledger); (output/"non_lzu_invariance_audit.json").write_text(json.dumps(invariance,indent=2),encoding="utf-8")
    if not invariance["passed"]: raise RuntimeError("Non-LZU bitwise invariance failed")
    for name,column,truek in (("formal","adapted_nez_logit",False),("truek","adapted_nez_logit",True)):
        patients,summary,by_fold,by_center=evaluate_channel_ledger(ledger,logit_column=column,truek=truek)
        patients.to_csv(output/f"p2_lzu_adapter_{name}_by_patient.csv",index=False); summary.to_csv(output/f"p2_lzu_adapter_{name}_summary.csv",index=False); by_fold.to_csv(output/f"p2_lzu_adapter_{name}_by_fold.csv",index=False); by_center.to_csv(output/f"p2_lzu_adapter_{name}_by_center.csv",index=False)
    report={"optimization_status":optimization["optimization_status"],"performance_status":"PENDING_SUMMARY","non_lzu_invariance_status":"PASSED","threshold_freeze_status":"PASSED","full_five_fold_run":args.max_outer_folds in (0,5)}
    (output/"P2_Q10_LZU_ADAPTER_OPTIMIZATION_REPORT.md").write_text("# P2-Q10 LZU Adapter Optimization Report\n\n```json\n"+json.dumps(report,indent=2)+"\n```\n",encoding="utf-8")
    print(json.dumps({"status":"passed","output_dir":str(output),"n_patients":int(ledger.subject_id.nunique()),"optimization_status":optimization["optimization_status"],"non_lzu_invariance":True},indent=2))
    if not optimization["all_folds_optimization_passed"]: raise RuntimeError(f"Adapter optimization audit failed: {optimization['optimization_status']}")


if __name__=="__main__": main()
