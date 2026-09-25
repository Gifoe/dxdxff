from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT=Path(__file__).resolve().parents[2];REPO=PROJECT.parent
for path in (REPO,PROJECT):
    if str(path) not in sys.path:sys.path.insert(0,str(path))

from neuroez_c.task2.clinical_target import build_clinical_target_lookup
from neuroez_c.task2.cop.evaluation import bootstrap_ci,center_metrics,metric_row,summarize_fold_metrics
from neuroez_c.task2.cop.feature_groups import match_feature_families
from neuroez_c.task2.cop.feature_phenotype import build_feature_phenotype
from neuroez_c.task2.cop.negative_controls import CONTROLS,permute_target_lookup,permute_training_outcomes
from neuroez_c.task2.cop.nested_elastic_net import choose_groups,fit_outer_model
from neuroez_c.task2.cop.p2_nez_auxiliary import build_p2_auxiliary,fit_seizure_clean_nez_prototype
from neuroez_c.task2.cop.patient_table import build_patient_table,model_columns
from neuroez_c.task2.cop.profiles import cop_profile_names,get_cop_profile
from neuroez_c.task2.cop.raw_propagation import build_raw_propagation
from neuroez_c.task2.cop.schema import adapt_feature_record,adapt_raw_record,align_feature_raw,inspect_cache_schema
from neuroez_c.task2.data import filtered_cache,load_cache
from neuroez_c.task2.exclusions import load_exclusion_manifest
from neuroez_c.task2.outcomes import load_outcome_table
from neuroez_c.task2.p2_adapter import P23Runtime,P2ExportAdapter,load_p2_args,locate_p2_config
from neuroez_c.task2.protocol import assert_checkpoint_safe,checkpoint_training_subjects,load_fold_manifest,manifest_hash,manifest_patient_keys
from neuroez_c.task2.training import make_loader,prepare_examples,seed_everything


def parser()->argparse.ArgumentParser:
    p=argparse.ArgumentParser(description="Compact Outcome Phenotype nested Elastic-Net")
    p.add_argument("--profile",required=True,choices=cop_profile_names());p.add_argument("--feature_cache",default=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH"),required=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH") is None);p.add_argument("--raw_cache",default=os.getenv("DRE_TASK1_RAW_CACHE_PATH"),required=os.getenv("DRE_TASK1_RAW_CACHE_PATH") is None)
    p.add_argument("--outcome_table",default=os.getenv("DRE_TASK2_OUTCOME_TABLE","cache://patient_index"));p.add_argument("--fold_manifest",default=os.getenv("DRE_TASK2_FOLD_MANIFEST"),required=os.getenv("DRE_TASK2_FOLD_MANIFEST") is None);p.add_argument("--exclusion_manifest",default=os.getenv("DRE_TASK2_EXCLUSION_MANIFEST",str(PROJECT/"configs"/"data_exclusions.csv")))
    p.add_argument("--p2_checkpoint_root",default=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT"));p.add_argument("--p2_runtime_root",default=os.getenv("DRE_TASK1_P2_RUNTIME_ROOT",str(REPO/"P23_TRN_NEZ_80")));p.add_argument("--p2_config");p.add_argument("--p2_training_manifest",default=os.getenv("DRE_TASK1_P2_TRAINING_MANIFEST"))
    p.add_argument("--protocol",choices=("quick","outer_cv"),default="outer_cv");p.add_argument("--outer_folds",type=int,default=5);p.add_argument("--inner_folds",type=int,default=3);p.add_argument("--max_outer_folds",type=int,default=0);p.add_argument("--max_patients",type=int,default=0);p.add_argument("--batch_size",type=int,default=8)
    p.add_argument("--seed",type=int,default=42);p.add_argument("--bootstrap_repeats",type=int,default=2000);p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu");p.add_argument("--negative_control",choices=CONTROLS,default="none")
    p.add_argument("--phenotype_cache_dir",default=os.getenv("DRE_TASK2_COP_PHENOTYPE_CACHE_DIR"));p.add_argument("--p2_evidence_cache_dir",default=os.getenv("DRE_TASK2_COP_P2_CACHE_DIR"));p.add_argument("--output_dir",default=os.getenv("DRE_TASK2_COP_OUTPUT_DIR"),required=os.getenv("DRE_TASK2_COP_OUTPUT_DIR") is None)
    p.add_argument("--audit_only",action="store_true");p.add_argument("--strict",action="store_true");p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True);return p


def _json(path:Path,value:Any)->None:
    temporary=path.with_suffix(path.suffix+".tmp");temporary.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding="utf-8");temporary.replace(path)


def _csv(path:Path,frame:pd.DataFrame)->None:
    temporary=path.with_suffix(path.suffix+".tmp");frame.to_csv(temporary,index=False);temporary.replace(path)


def _balanced_take(subjects:Iterable[str],labels:dict[str,int],count:int)->list[str]:
    values=sorted(map(str,subjects))
    if count<=0 or len(values)<=count:return values
    buckets={label:[p for p in values if labels[p]==label] for label in (0,1)};selected=[]
    while len(selected)<count and any(buckets.values()):
        for label in (0,1):
            if buckets[label] and len(selected)<count:selected.append(buckets[label].pop(0))
    return selected


def _checkpoint_fold(path:Path)->int|None:
    payload=torch.load(path,map_location="cpu",weights_only=False)
    for mapping in (payload,payload.get("metadata",{}) if isinstance(payload,dict) else {}):
        if isinstance(mapping,dict):
            for key in ("outer_fold","fold","fold_idx"):
                if key in mapping:return int(mapping[key])
    return None


def _extract_p2(adapter:P2ExportAdapter,loader)->list[dict[str,Any]]:
    adapter.eval();records=[]
    with torch.no_grad():
        for batch in loader:
            export={k:v.detach().cpu() for k,v in adapter(batch).items() if torch.is_tensor(v)}
            for p,subject in enumerate(batch["subject_id"]):
                c=int(batch["channel_mask"][p].sum());s=int(batch["seizure_mask"][p].sum());records.append({"patient_key":str(subject),"center":str(batch["center"][p]),"clinical_target_mask":batch["clinical_target_mask"][p:p+1,:c].cpu().bool(),"channel_mask":batch["channel_mask"][p:p+1,:c].cpu().bool(),"seizure_mask":batch["seizure_mask"][p:p+1,:s].cpu().bool(),"seizure_channel_mask":batch["seizure_channel_mask"][p:p+1,:s,:c].cpu().bool(),"final_nez_logit":export["final_nez_logit"][p:p+1,:c],"seizure_channel_embedding":export["seizure_channel_embedding"][p:p+1,:s,:c]})
    return records


def _p2_records(args:Any,fold:int,train:set[str],test:set[str],runtime:P23Runtime,p2_args:Any,feature_cache:dict[str,Any],labels:dict[str,int],target_lookup:dict[str,Any])->tuple[list[dict[str,Any]],dict[str,Any]]:
    checkpoint=Path(args.p2_checkpoint_root)/f"fold_{fold}"/"best_model.pt"
    if not checkpoint.exists():raise FileNotFoundError(checkpoint)
    metadata=_checkpoint_fold(checkpoint);match=metadata==fold
    if args.strict and not match:raise ValueError(f"P2 checkpoint fold mismatch: expected {fold}, found {metadata}")
    leakage=False
    if args.p2_training_manifest:assert_checkpoint_safe(checkpoint_training_subjects(args.p2_training_manifest,fold),test);leakage=True
    elif args.strict:raise ValueError("Strict COP with P2 requires --p2_training_manifest")
    cache_root=Path(args.p2_evidence_cache_dir) if args.p2_evidence_cache_dir else Path(args.output_dir).parent.parent/"_cop_p2_evidence"/f"seed_{args.seed}"
    signature=hashlib.sha256(json.dumps({"fold":fold,"train":sorted(train),"test":sorted(test),"checkpoint":str(checkpoint.resolve()),"feature":str(Path(args.feature_cache).resolve()),"control":args.negative_control},sort_keys=True).encode()).hexdigest();path=cache_root/f"fold_{fold}.pt"
    records=None
    if args.resume and path.exists():
        value=torch.load(path,map_location="cpu",weights_only=False)
        if value.get("signature")==signature:records=value["records"]
    if records is None:
        examples,_=prepare_examples(runtime,feature_cache,p2_args,normalizer_subjects=sorted(train),output_subjects=sorted(train|test));loader=make_loader(examples,runtime,labels,None,batch_size=args.batch_size,shuffle=False,seed=args.seed,clinical_target_lookup=target_lookup);adapter=P2ExportAdapter(runtime,p2_args,checkpoint,device=args.device,limited_finetune=False)
        if any(p.requires_grad for p in adapter.parameters()):raise RuntimeError("COP requires fully frozen P2")
        records=_extract_p2(adapter,loader);path.parent.mkdir(parents=True,exist_ok=True);temporary=path.with_suffix(".tmp");torch.save({"signature":signature,"records":records},temporary);temporary.replace(path);del adapter
        if torch.cuda.is_available():torch.cuda.empty_cache()
    audit={"outer_fold":fold,"checkpoint":str(checkpoint.resolve()),"checkpoint_outer_fold":metadata,"checkpoint_outer_fold_matches":match,"training_manifest_leakage_verified":leakage,"p2_frozen":True,"n_records":len(records)};return records,audit


def _coefficient_rows(fitted,fold:int)->list[dict[str,Any]]:
    rows=[]
    for name,value in zip(fitted.preprocessor.retained_features,fitted.model.coef_[0]):
        group="raw" if name.startswith("raw__") else "p2" if name.startswith("p2__") else "feature" if name.startswith("feature__") else "missing_indicator"
        rows.append({"outer_fold":fold,"feature":name,"coefficient":float(value),"absolute_coefficient":abs(float(value)),"selected_nonzero":bool(abs(value)>1e-12),"feature_group":group})
    return rows


def _write_final(output:Path,args:Any,predictions:pd.DataFrame,fold_fixed:list[dict],fold_inner:list[dict],feature_table:pd.DataFrame,raw_table:pd.DataFrame,p2_table:pd.DataFrame,filter_rows:list[dict],group_rows:list[dict],inner_rows:list[dict],coeff_rows:list[dict],missing_rows:list[dict],alignment:pd.DataFrame,p2_safety:list[dict],folds:pd.DataFrame,outcomes:pd.DataFrame,fold_details:list[dict])->None:
    fixed=pd.DataFrame(fold_fixed);inner=pd.DataFrame(fold_inner);_csv(output/"oof_patient_predictions.csv",predictions);_csv(output/"fold_metrics_fixed05.csv",fixed);_csv(output/"fold_metrics_inner_threshold.csv",inner);_csv(output/"summary_metrics_fixed05.csv",summarize_fold_metrics(fixed,predictions,"prediction_fixed05"));_csv(output/"summary_metrics_inner_threshold.csv",summarize_fold_metrics(inner,predictions,"prediction_inner_threshold"));_csv(output/"center_metrics_fixed05.csv",center_metrics(predictions,"prediction_fixed05"));_csv(output/"center_metrics_inner_threshold.csv",center_metrics(predictions,"prediction_inner_threshold"));_csv(output/"bootstrap_ci_fixed05.csv",bootstrap_ci(predictions,"prediction_fixed05",args.bootstrap_repeats,args.seed));_csv(output/"bootstrap_ci_inner_threshold.csv",bootstrap_ci(predictions,"prediction_inner_threshold",args.bootstrap_repeats,args.seed))
    combined=outcomes[["patient_key","outcome_label"]].rename(columns={"outcome_label":"outcome_true"}).merge(folds,on="patient_key",validate="one_to_one").merge(feature_table,on="patient_key",how="left",validate="one_to_one")
    if not raw_table.empty:combined=combined.merge(raw_table.drop(columns="center",errors="ignore"),on="patient_key",how="left",validate="one_to_one")
    if not p2_table.empty:
        assigned=p2_table.merge(folds,on="patient_key",suffixes=("_p2","_assigned"));assigned=assigned[assigned.outer_fold_p2==assigned.outer_fold_assigned].drop(columns=["outer_fold_p2","outer_fold_assigned","center"],errors="ignore");combined=combined.merge(assigned,on="patient_key",how="left",validate="one_to_one")
    _csv(output/"cop_patient_feature_table.csv",combined);_csv(output/"cop_feature_phenotype_table.csv",feature_table);_csv(output/"cop_raw_propagation_table.csv",raw_table);_csv(output/"cop_p2_auxiliary_table.csv",p2_table);_csv(output/"cop_feature_filter_audit.csv",pd.DataFrame(filter_rows));_csv(output/"cop_group_selection_audit.csv",pd.DataFrame(group_rows));_csv(output/"cop_inner_cv_results.csv",pd.DataFrame(inner_rows));_csv(output/"cop_missingness_audit.csv",pd.DataFrame(missing_rows));_csv(output/"cop_raw_alignment_audit.csv",alignment)
    coefficient=pd.DataFrame(coeff_rows)
    if not coefficient.empty:
        summary=coefficient.groupby("feature").coefficient.agg([("selection_frequency",lambda x:float((x.abs()>1e-12).mean())),("mean_coefficient","mean"),("std_coefficient","std")]).reset_index();sign=coefficient.assign(sign=np.sign(coefficient.coefficient)).groupby("feature").sign.apply(lambda x:float(abs(x[x!=0].mean())) if (x!=0).any() else np.nan).rename("sign_consistency").reset_index();summary=summary.merge(sign,on="feature");coefficient=coefficient.merge(summary,on="feature",how="left");coefficient["flag"]=np.where(coefficient.sign_consistency<.8,"UNSTABLE_FEATURE_DIRECTION","")
        for detail in fold_details:
            mask=coefficient.outer_fold==detail["outer_fold"]
            if int(coefficient.loc[mask,"selected_nonzero"].sum())>detail["n_train"]/5:coefficient.loc[mask,"flag"]=coefficient.loc[mask,"flag"].map(lambda value:"|".join(x for x in (value,"MODEL_TOO_DENSE") if x))
    _csv(output/"cop_model_coefficients.csv",coefficient);_json(output/"cop_p2_fold_safety_audit.json",p2_safety)
    completed=predictions.outer_fold.nunique()==5 and set(predictions.patient_key)==set(folds.patient_key);paper_valid=bool(args.protocol=="outer_cv" and args.strict and completed and args.negative_control=="none" and all(x.get("p2_safe",True) for x in fold_details))
    protocol={"pipeline":"COP-Outcome","profile":args.profile,"fixed_outer_folds":5,"completed_outer_folds":int(predictions.outer_fold.nunique()),"inner_folds":args.inner_folds,"inner_cv_used":True,"outer_test_used_for_selection":False,"center_as_input":False,"success_label":1,"failure_label":0,"manifest_hash":manifest_hash(folds),"negative_control":args.negative_control,"folds":fold_details,"paper_valid":paper_valid};_json(output/"cop_protocol_audit.json",protocol);_json(output/"run_args.json",vars(args))
    pooled=summarize_fold_metrics(fixed,predictions,"prediction_fixed05").iloc[0].to_dict();flags=[]
    if args.profile=="O0_FEATURE_PHENOTYPE" and pooled.get("auroc",0)<.60:flags.append("FEATURE_OUTCOME_SIGNAL_WEAK")
    if args.profile=="O3_COP_GROUP_GATED":
        if not any(row["candidate_group"]=="raw" and row["selected"] for row in group_rows):flags.append("RAW_PROPAGATION_NOT_COMPLEMENTARY")
        if not any(row["candidate_group"]=="p2" and row["selected"] for row in group_rows):flags.append("P2_NEZ_NOT_COMPLEMENTARY")
        fold_good=sum((row.get("auroc",0)>.5) for row in fold_fixed)
        if pooled.get("accuracy",0)<.65 and pooled.get("auroc",0)<.65:flags.append("CURRENT_EPHYSIOLOGY_SIGNAL_INSUFFICIENT_FOR_0P8")
        if pooled.get("accuracy",0)>=.70 and pooled.get("auroc",0)>=.72 and fold_good>=4:flags.append("PROMISING_OUTCOME_PHENOTYPE_SIGNAL")
        if pooled.get("accuracy",0)>=.80 and pooled.get("balanced_accuracy",0)>=.75 and pooled.get("macro_f1",0)>=.75 and pooled.get("auroc",0)>=.80 and pooled.get("failure_recall",0)>=.70:flags.append("STRONG_OUTCOME_RESULT_REQUIRES_LOCKED_VALIDATION")
    status=[]
    if not completed:status.append("FULL_OUTER_CV_NOT_RUN")
    if args.protocol=="quick":status.append("QUICK_SCREENING_NOT_PAPER_VALID")
    if not paper_valid:status.append("OUTER_CV_NOT_PAPER_VALID")
    report=["# Compact Outcome Phenotype Model Report","",f"Profile: `{args.profile}`",f"Patients: {len(predictions)}","","COP-Outcome directly derives a compact patient-level electrophysiological phenotype from handcrafted seizure features, raw propagation dynamics, and four fold-safe P2 NEZ verification scores.","","P2 is a weak auxiliary verifier and is not interpreted as biological true EZ.","","## Status",*(f"- {x}" for x in status+flags)];(output/"COP_OUTCOME_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")


def main()->int:
    args=parser().parse_args();profile=get_cop_profile(args.profile);output=Path(args.output_dir).expanduser();output.mkdir(parents=True,exist_ok=True);seed_everything(args.seed)
    if args.outer_folds!=5 and args.protocol=="outer_cv":raise ValueError("Formal COP requires five outer folds")
    exclusions=load_exclusion_manifest(args.exclusion_manifest);feature_cache=filtered_cache(load_cache(args.feature_cache),exclusions);raw_cache=filtered_cache(load_cache(args.raw_cache),exclusions)
    _json(output/"cop_feature_cache_schema.json",inspect_cache_schema(feature_cache,"feature"));_json(output/"cop_raw_cache_schema.json",inspect_cache_schema(raw_cache,"raw"))
    outcomes,_=load_outcome_table(args.outcome_table,cache=feature_cache);outcomes=outcomes[outcomes.outcome_group.isin(["success","failure"])]
    outcomes=outcomes[outcomes.patient_key.isin(manifest_patient_keys(args.fold_manifest))].copy();folds=load_fold_manifest(args.fold_manifest,outcomes,strict=args.strict);cohort=set(folds.patient_key);outcomes=outcomes[outcomes.patient_key.isin(cohort)].copy();labels=outcomes.set_index("patient_key").outcome_label.astype(int).to_dict()
    target_lookup,target_alignment,target_distribution=build_clinical_target_lookup(feature_cache,sorted(cohort),strict=args.strict)
    if args.negative_control=="COP_TARGET_PERMUTATION":target_lookup=permute_target_lookup(target_lookup,args.seed)
    feature_records=[];axis_rows=[]
    for record in feature_cache["run_records"]:
        if str(record.get("subject_id")) in cohort:value,audit=adapt_feature_record(record,target_lookup);feature_records.append(value);axis_rows.append(audit)
    raw_records=[adapt_raw_record(record,target_lookup) for record in raw_cache["run_records"] if str(record.get("subject_id")) in cohort]
    alignment=pd.DataFrame(align_feature_raw(feature_records,raw_records,strict=args.strict));_csv(output/"cop_raw_alignment_audit.csv",alignment);_csv(output/"cop_feature_axis_audit.csv",pd.DataFrame(axis_rows));_csv(output/"clinical_target_alignment_audit.csv",target_alignment);_csv(output/"clinical_target_distribution_audit.csv",target_distribution)
    names=feature_records[0].feature_names;groups,name_audit=match_feature_families(names);_csv(output/"cop_feature_name_audit.csv",pd.DataFrame(name_audit))
    phenotype_root=Path(args.phenotype_cache_dir) if args.phenotype_cache_dir else output.parent.parent/"_cop_phenotype_cache";phenotype_root.mkdir(parents=True,exist_ok=True);control_suffix=args.negative_control if args.negative_control in ("COP_TARGET_PERMUTATION","COP_RAW_CHANNEL_PERMUTATION") else "none";cache_path=phenotype_root/f"phenotypes_{control_suffix}.pkl"
    cached=None
    if args.resume and cache_path.exists():
        with cache_path.open("rb") as handle:cached=pickle.load(handle)
    signature=hashlib.sha256(json.dumps({"feature":str(Path(args.feature_cache).resolve()),"raw":str(Path(args.raw_cache).resolve()),"cohort":sorted(cohort),"control":control_suffix,"version":"cop_v2"},sort_keys=True).encode()).hexdigest()
    cache_usable=bool(cached and cached.get("signature")==signature and (not (profile.raw or profile.group_gated) or not cached.get("raw",pd.DataFrame()).empty))
    if cache_usable:feature_table=cached["feature"];definition=cached["definition"];feature_seizure=cached["feature_seizure"];raw_table=cached.get("raw",pd.DataFrame());channel_audit=cached.get("channel",pd.DataFrame());seizure_audit=cached.get("seizure",pd.DataFrame())
    else:
        feature_table,definition,feature_seizure=build_feature_phenotype(feature_records,groups);raw_table=channel_audit=seizure_audit=pd.DataFrame()
        if profile.raw or profile.group_gated:raw_table,channel_audit,seizure_audit=build_raw_propagation(raw_records,permutation=args.negative_control=="COP_RAW_CHANNEL_PERMUTATION",seed=args.seed)
        temporary=cache_path.with_suffix(".tmp")
        with temporary.open("wb") as handle:pickle.dump({"signature":signature,"feature":feature_table,"definition":definition,"feature_seizure":feature_seizure,"raw":raw_table,"channel":channel_audit,"seizure":seizure_audit},handle,pickle.HIGHEST_PROTOCOL)
        temporary.replace(cache_path)
    _csv(output/"cop_feature_phenotype_table.csv",feature_table);_csv(output/"cop_feature_phenotype_definition.csv",definition);_csv(output/"cop_recruitment_channel_audit.csv",channel_audit);_csv(output/"cop_recruitment_seizure_audit.csv",seizure_audit)
    if args.audit_only:
        p2_audit=[]
        if args.p2_checkpoint_root:
            for fold in sorted(folds.outer_fold.unique()):
                test=set(folds.loc[folds.outer_fold==fold,"patient_key"]);checkpoint=Path(args.p2_checkpoint_root)/f"fold_{int(fold)}"/"best_model.pt";entry={"outer_fold":int(fold),"checkpoint_exists":checkpoint.exists(),"checkpoint_outer_fold":_checkpoint_fold(checkpoint) if checkpoint.exists() else None}
                if args.p2_training_manifest:assert_checkpoint_safe(checkpoint_training_subjects(args.p2_training_manifest,int(fold)),test);entry["training_manifest_leakage_verified"]=True
                p2_audit.append(entry)
        _json(output/"cop_p2_fold_safety_audit.json",p2_audit);(output/"COP_OUTCOME_REPORT.md").write_text("# COP-Outcome Audit\n\nAUDIT_ONLY_NO_OUTCOME_MODEL_RUN\n",encoding="utf-8");return 0
    if profile.raw and raw_table.empty:raise ValueError("MISSING_REQUIRED_FIELD: raw propagation phenotype")
    runtime=p2_args=None
    if profile.p2:
        if not args.p2_checkpoint_root:raise ValueError("MISSING_REQUIRED_FIELD: --p2_checkpoint_root")
        runtime=P23Runtime(args.p2_runtime_root);p2_args=load_p2_args(locate_p2_config(args.p2_checkpoint_root,args.p2_config),feature_cache=args.feature_cache)
    fold_values=list(map(int,sorted(folds.outer_fold.unique())[:args.max_outer_folds or None]));prediction_frames=[];fixed_rows=[];threshold_rows=[];filter_rows=[];group_rows=[];inner_rows=[];coeff_rows=[];missing_rows=[];p2_rows=[];p2_safety=[];fold_details=[]
    for fold in fold_values:
        test=set(folds.loc[folds.outer_fold==fold,"patient_key"]);train=cohort-test
        if args.max_patients:
            test=set(_balanced_take(test,labels,max(4,args.max_patients//5)));train=set(_balanced_take(train,labels,max(args.inner_folds*4,args.max_patients-len(test))))
        fold_dir=output/f"fold_{fold}";fold_dir.mkdir(parents=True,exist_ok=True);result_path=fold_dir/"fold_result.pkl"
        fold_signature=hashlib.sha256(json.dumps({"profile":args.profile,"fold":fold,"train":sorted(train),"test":sorted(test),"seed":args.seed,"inner_folds":args.inner_folds,"protocol":args.protocol,"control":args.negative_control,"feature_cache":str(Path(args.feature_cache).resolve()),"raw_cache":str(Path(args.raw_cache).resolve()),"p2_root":str(Path(args.p2_checkpoint_root).resolve()) if args.p2_checkpoint_root else None,"version":"cop_v2"},sort_keys=True).encode()).hexdigest()
        if args.resume and result_path.exists():
            with result_path.open("rb") as handle:bundle=pickle.load(handle)
            if bundle.get("signature")==fold_signature:
                prediction_frames.append(bundle["prediction"]);fixed_rows.append(bundle["fixed_metric"]);threshold_rows.append(bundle["threshold_metric"]);filter_rows.extend(bundle["filter_rows"]);group_rows.extend(bundle["group_rows"]);inner_rows.extend(bundle["inner_rows"]);coeff_rows.extend(bundle["coefficient_rows"]);missing_rows.extend(bundle["missing_rows"]);fold_details.append(bundle["fold_detail"])
                if not bundle["p2_table"].empty:p2_rows.append(bundle["p2_table"])
                if bundle.get("p2_safety") is not None:p2_safety.append(bundle["p2_safety"])
                print(f"[{args.profile}] fold {fold}: resumed completed fold",flush=True);continue
        p2_table=pd.DataFrame();p2_safe=True
        if profile.p2:
            evidence,safety=_p2_records(args,fold,train,test,runtime,p2_args,feature_cache,labels,target_lookup);prototype=fit_seizure_clean_nez_prototype(evidence,train);p2_table=build_p2_auxiliary(evidence,prototype,target_permutation=False,seed=args.seed+fold);p2_table["outer_fold"]=fold;p2_rows.append(p2_table);safety["prototype_audit"]=vars(prototype.audit());p2_safety.append(safety);p2_safe=safety["checkpoint_outer_fold_matches"] and safety["training_manifest_leakage_verified"]
        current_raw=raw_table if not raw_table.empty else None;table=build_patient_table(feature_table,current_raw,p2_table if not p2_table.empty else None,outcomes,folds);table=table[table.patient_key.isin(train|test)].copy();train_frame=table[table.patient_key.isin(train)].reset_index(drop=True);test_frame=table[table.patient_key.isin(test)].reset_index(drop=True)
        group_features={name:model_columns(train_frame,[name]) for name in ("feature","raw","p2")};selected=["feature"];local_group=[];local_inner=[];y=train_frame.outcome_true.to_numpy(dtype=int)
        if args.negative_control=="COP_OUTCOME_PERMUTATION":y=permute_training_outcomes(y,args.seed+fold)
        if profile.group_gated:selected,local_group,local_inner=choose_groups(train_frame,group_features,y,train_frame.center.tolist(),outer_fold=fold,inner_folds=args.inner_folds,seed=args.seed+fold,quick=args.protocol=="quick")
        else:
            if profile.raw:selected.append("raw")
            if profile.p2:selected.append("p2")
        features=[name for group in selected for name in group_features[group]];fitted=fit_outer_model(train_frame,features,y,train_frame.center.tolist(),selected,outer_fold=fold,inner_folds=args.inner_folds,seed=args.seed+fold,quick=args.protocol=="quick");probability=fitted.predict_probability(test_frame);prediction=test_frame[["patient_key","center","outcome_true","feature_missing_fraction","raw_missing_fraction","p2_missing_fraction"]].copy();prediction["outer_fold"]=fold;prediction["seed"]=args.seed;prediction["profile"]=args.profile;prediction["probability_success"]=probability;prediction["probability_failure"]=1-probability;prediction["prediction_fixed05"]=(probability>=.5).astype(int);prediction["prediction_inner_threshold"]=(probability>=fitted.inner.threshold).astype(int);prediction["inner_selected_threshold"]=fitted.inner.threshold;prediction["selected_groups"]="+".join(selected);prediction["selected_C"]=fitted.inner.C;prediction["selected_l1_ratio"]=fitted.inner.l1_ratio
        fixed_metric=metric_row(prediction,"prediction_fixed05",.5,fold);threshold_metric=metric_row(prediction,"prediction_inner_threshold","inner_selected_threshold",fold);local_filter=fitted.preprocessor.audit;local_inner_all=list(local_inner)+[dict(row,candidate_groups="final:"+"+".join(selected)) for row in fitted.inner.rows];local_coeff=_coefficient_rows(fitted,fold);local_missing=table[["patient_key","feature_missing_fraction","raw_missing_fraction","p2_missing_fraction"]].assign(outer_fold=fold,partition=lambda x:np.where(x.patient_key.isin(train),"train","test")).to_dict("records");fold_detail={"outer_fold":fold,"n_train":len(train_frame),"n_test":len(test_frame),"train_test_overlap":len(train&test),"selected_groups":selected,"selected_C":fitted.inner.C,"selected_l1_ratio":fitted.inner.l1_ratio,"inner_threshold":fitted.inner.threshold,"p2_safe":p2_safe}
        prediction_frames.append(prediction);fixed_rows.append(fixed_metric);threshold_rows.append(threshold_metric);filter_rows.extend(local_filter);group_rows.extend(local_group);inner_rows.extend(local_inner_all);coeff_rows.extend(local_coeff);missing_rows.extend(local_missing);fold_details.append(fold_detail)
        temporary=fold_dir/"final_model.pkl.tmp"
        with temporary.open("wb") as handle:pickle.dump({"model":fitted,"selected_groups":selected,"features":features,"outer_fold":fold},handle,pickle.HIGHEST_PROTOCOL)
        temporary.replace(fold_dir/"final_model.pkl");result_temporary=result_path.with_suffix(".tmp")
        with result_temporary.open("wb") as handle:pickle.dump({"signature":fold_signature,"prediction":prediction,"fixed_metric":fixed_metric,"threshold_metric":threshold_metric,"filter_rows":local_filter,"group_rows":local_group,"inner_rows":local_inner_all,"coefficient_rows":local_coeff,"missing_rows":local_missing,"fold_detail":fold_detail,"p2_table":p2_table,"p2_safety":safety if profile.p2 else None},handle,pickle.HIGHEST_PROTOCOL)
        result_temporary.replace(result_path);print(f"[{args.profile}] fold {fold}: complete, groups={'+'.join(selected)}, C={fitted.inner.C}, l1={fitted.inner.l1_ratio}",flush=True)
    predictions=pd.concat(prediction_frames,ignore_index=True)
    if predictions.patient_key.duplicated().any():raise ValueError("COP OOF contains duplicate patients")
    p2_output=pd.concat(p2_rows,ignore_index=True) if p2_rows else pd.DataFrame();_write_final(output,args,predictions,fixed_rows,threshold_rows,feature_table,raw_table,p2_output,filter_rows,group_rows,inner_rows,coeff_rows,missing_rows,alignment,p2_safety,folds,outcomes,fold_details);print(json.dumps({"profile":args.profile,"n_predictions":len(predictions),"folds":fold_values,"output":str(output.resolve())},indent=2));return 0


if __name__=="__main__":raise SystemExit(main())
