from __future__ import annotations

import argparse,json,os,sys,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm

PROJECT=Path(__file__).resolve().parents[2]; REPO=PROJECT.parent
for p in (REPO,PROJECT):
    if str(p) not in sys.path: sys.path.insert(0,str(p))

from neuroez_c.task2.clinical_target import build_clinical_target_lookup
from neuroez_c.task2.data import load_cache,filtered_cache
from neuroez_c.task2.exclusions import load_exclusion_manifest,build_exclusion_audit
from neuroez_c.task2.outcomes import load_outcome_table
from neuroez_c.task2.protocol import (load_fold_manifest,manifest_patient_keys,
                                      checkpoint_training_subjects,assert_checkpoint_safe)
from neuroez_c.task2.ngbr.schema import adapt_raw_record,adapt_feature_record,validate_target
from neuroez_c.task2.ngbr.channel_alignment import audit_raw_feature_alignment,align_p2_to_raw
from neuroez_c.task2.mosaic import PROFILE
from neuroez_c.task2.mosaic.schema import from_ngbr
from neuroez_c.task2.mosaic.pilot_cohort import select_pilot16,build_pilot_outer_folds,validate_pilot_outer_folds
from neuroez_c.task2.mosaic.temporal_phases import phase_intervals
from neuroez_c.task2.mosaic.cache import AtomicCache,trajectory_signature,file_identity
from neuroez_c.task2.mosaic.fragility_trajectory import ALGORITHM_VERSION as FRAG_VERSION,FragilityConfig,compute_fragility_trajectory
from neuroez_c.task2.mosaic.fragility_expert import seizure_fragility_features,patient_fragility_features
from neuroez_c.task2.mosaic.phase_transfer_entropy import ALGORITHM_VERSION as PTE_VERSION,PTEConfig,compute_pte_windows
from neuroez_c.task2.mosaic.cii_expert import seizure_cii_features,patient_cii_features
from neuroez_c.task2.mosaic.recruitment_expert import seizure_recruitment_features,patient_recruitment_features
from neuroez_c.task2.mosaic.spectral_state import ALGORITHM_VERSION as SPEC_VERSION,compute_spectral_state
from neuroez_c.task2.mosaic.spectral_expert import seizure_spectral_features,patient_spectral_features
from neuroez_c.task2.mosaic.nez_expert import load_p2_oof_channel_ledger,patient_nez_features
from neuroez_c.task2.mosaic.crossfit import lopo_crossfit_experts,fit_outer_experts
from neuroez_c.task2.mosaic.meta_fusion import META_FEATURES,evidence_bottleneck,fit_meta
from neuroez_c.task2.mosaic.evaluation import metric_row,bootstrap_ci
from neuroez_c.task2.mosaic.audit import protocol_audit


def parser():
    p=argparse.ArgumentParser(description="MOSAIC-Outcome fixed pilot16 model")
    env=lambda n:os.getenv(n)
    p.add_argument("--feature_cache",default=env("DRE_TASK1_FEATURE_CACHE_PATH"),required=not env("DRE_TASK1_FEATURE_CACHE_PATH")); p.add_argument("--raw_cache",default=env("DRE_TASK1_RAW_CACHE_PATH"),required=not env("DRE_TASK1_RAW_CACHE_PATH"))
    p.add_argument("--outcome_table",default=env("DRE_TASK2_OUTCOME_TABLE") or "cache://patient_index"); p.add_argument("--original_fold_manifest",default=env("DRE_TASK2_FOLD_MANIFEST"),required=not env("DRE_TASK2_FOLD_MANIFEST")); p.add_argument("--exclusion_manifest",default=env("DRE_TASK2_EXCLUSION_MANIFEST"))
    p.add_argument("--p2_checkpoint_root",default=env("DRE_TASK1_P2_CHECKPOINT_ROOT")); p.add_argument("--p2_runtime_root",default=env("DRE_TASK1_P2_RUNTIME_ROOT")); p.add_argument("--p2_config"); p.add_argument("--p2_training_manifest",default=env("DRE_TASK1_P2_TRAINING_MANIFEST")); p.add_argument("--p2_oof_channel_ledger",default=env("DRE_TASK1_P2_OOF_CHANNEL_LEDGER"))
    p.add_argument("--output_dir",default=env("DRE_TASK2_MOSAIC_OUTPUT_DIR"),required=not env("DRE_TASK2_MOSAIC_OUTPUT_DIR")); p.add_argument("--cache_dir",default=env("DRE_TASK2_MOSAIC_CACHE_DIR"),required=not env("DRE_TASK2_MOSAIC_CACHE_DIR")); p.add_argument("--fragility_trajectory_cache_dir",default=env("DRE_TASK2_MOSAIC_FRAGILITY_CACHE_SOURCE"))
    p.add_argument("--pilot_size",type=int,default=16); p.add_argument("--patients_per_center",type=int,default=4); p.add_argument("--full_cohort",action="store_true",help="use every eligible patient in the original fixed five-fold cohort"); p.add_argument("--seed",type=int,default=42); p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu"); p.add_argument("--batch_size",type=int,default=8); p.add_argument("--n_jobs",type=int,default=8); p.add_argument("--biomarker_workers",type=int,default=4,help="CPU threads for Fragility/PTE windows; batch_size does not control biomarkers"); p.add_argument("--bootstrap_repeats",type=int,default=2000)
    p.add_argument("--strict",action="store_true"); p.add_argument("--resume",action=argparse.BooleanOptionalAction,default=True); p.add_argument("--audit_only",action="store_true"); p.add_argument("--skip_inner_crossfit",action="store_true",help="fast screen: skip LOPO/meta and average the five outer-test experts"); p.add_argument("--disable_fragility",action="store_true",help="ablation: skip expensive Fragility extraction and use neutral p_fragility=0.5"); p.add_argument("--disable_pte",action="store_true",help="ablation: skip PTE/CII extraction and use neutral p_cii=0.5")
    for x in ("fragility","cii","recruitment","spectral","nez"): p.add_argument(f"--recompute_{x}",action="store_true")
    return p


def atomic_csv(path,frame):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); frame.to_csv(tmp,index=False); tmp.replace(path)
def atomic_json(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,default=str),encoding="utf-8"); tmp.replace(path)
def atomic_text(path,value):
    path=Path(path); tmp=path.with_suffix(path.suffix+".tmp"); tmp.write_text(value,encoding="utf-8"); tmp.replace(path)
def atomic_npz(path,**arrays):
    path=Path(path); tmp=path.with_suffix(".tmp.npz"); np.savez_compressed(tmp,**arrays); tmp.replace(path)
def _frame(rows): return pd.DataFrame(rows) if rows else pd.DataFrame()


def prepare(args):
    if not args.full_cohort and (args.pilot_size!=16 or args.patients_per_center!=4): raise ValueError("MOSAIC_FULL pilot is fixed at 16 patients, four per center")
    feature_source=load_cache(args.feature_cache); raw_source=load_cache(args.raw_cache); exclusions=load_exclusion_manifest(args.exclusion_manifest)
    exclusion_audit=build_exclusion_audit(exclusions,feature_source["run_records"],raw_source["run_records"])
    feature_cache=filtered_cache(feature_source,exclusions); raw_cache=filtered_cache(raw_source,exclusions)
    outcomes,_=load_outcome_table(args.outcome_table,cache=feature_cache); outcomes=outcomes[outcomes.outcome_group.isin(["success","failure"])].copy(); outcomes["outcome_true"]=outcomes.outcome_label.astype(int)
    outcomes=outcomes[outcomes.patient_key.isin(manifest_patient_keys(args.original_fold_manifest))]; original=load_fold_manifest(args.original_fold_manifest,outcomes,strict=args.strict); cohort=set(original.patient_key.astype(str)); outcomes=outcomes[outcomes.patient_key.astype(str).isin(cohort)]
    target_lookup,target_alignment,target_distribution=build_clinical_target_lookup(feature_cache,sorted(cohort),strict=args.strict)
    features=[]; feature_audit=[]
    for source in feature_cache["run_records"]:
        if str(source.get("subject_id")) in cohort:
            rec,audit=adapt_feature_record(source,target_lookup); validate_target(rec); features.append(rec); feature_audit.append(audit)
    raws=[]; raw_audit=[]
    for source in raw_cache["run_records"]:
        if str(source.get("subject_id")) in cohort:
            rec,audit=adapt_raw_record(source,target_lookup); validate_target(rec); raws.append(rec); raw_audit.append(audit)
    alignment=audit_raw_feature_alignment(features,raws,strict=args.strict)
    by_raw={}
    for r in raws: by_raw.setdefault(r.patient_key,[]).append(r)
    patient_align=alignment.groupby("patient_key").match_rate.mean().to_dict()
    fold_map=original.set_index("patient_key").outer_fold.to_dict(); outcome_map=outcomes.set_index("patient_key").outcome_true.to_dict()
    if args.p2_oof_channel_ledger:
        preview=pd.read_csv(args.p2_oof_channel_ledger); key="patient_key" if "patient_key" in preview else "subject_id"; p2_available=set(preview[key].astype(str))
    else:
        checkpoints_ok=bool(args.p2_checkpoint_root) and all((Path(args.p2_checkpoint_root)/f"fold_{int(f)}"/"best_model.pt").exists() for f in sorted(original.outer_fold.unique()))
        p2_available=set(cohort) if checkpoints_ok and bool(args.p2_training_manifest) else set()
    candidates=[]
    for patient in sorted(cohort):
        runs=by_raw.get(patient,[]); valid_channels=[int(r.valid_channel_mask.sum()) for r in runs]; durations=[r.signal.shape[1]/r.sampling_rate for r in runs]
        target=target_lookup.get(patient,{})
        candidates.append({"patient_key":patient,"center":str(outcomes.set_index("patient_key").loc[patient,"center"]) if "center" in outcomes else (runs[0].center if runs else ""),"outcome_true":outcome_map[patient],"original_outer_fold":fold_map[patient],"n_seizures":len(runs),"n_valid_channels":int(np.median(valid_channels)) if valid_channels else 0,"valid_channel_ratio":float(np.mean([r.valid_channel_mask.mean() for r in runs])) if runs else 0,"raw_duration_sec":float(np.sum(durations)),"missing_field_count":0,"raw_feature_match_rate":patient_align.get(patient,0),"target_source":target.get("label_source",target.get("target_source",target.get("source","clinical_target"))),"in_original_folds":True,"feature_available":any(x.patient_key==patient for x in features),"raw_available":bool(runs),"target_complete":patient in target_lookup,"p2_oof_available":patient in p2_available,"has_valid_seizure":bool(runs),"target_has_both_classes":bool(runs and runs[0].clinical_target_mask.any() and not runs[0].clinical_target_mask.all())})
    output=Path(args.output_dir); output.mkdir(parents=True,exist_ok=True); candidate_frame=pd.DataFrame(candidates)
    if args.full_cohort:
        required=(candidate_frame.in_original_folds & candidate_frame.feature_available & candidate_frame.raw_available & candidate_frame.target_complete & candidate_frame.p2_oof_available & candidate_frame.has_valid_seizure & candidate_frame.target_has_both_classes & candidate_frame.raw_feature_match_rate.ge(.95))
        if args.strict and not required.all():
            bad=candidate_frame.loc[~required,"patient_key"].astype(str).tolist(); raise ValueError(f"FULL_COHORT_INELIGIBLE_PATIENTS: {bad}")
        pilot=candidate_frame.loc[required].copy().sort_values("patient_key").reset_index(drop=True)
        if len(pilot)!=len(cohort): raise ValueError(f"FULL_COHORT_INCOMPLETE: eligible={len(pilot)}, original={len(cohort)}")
        pilot_folds=original[["patient_key","outer_fold"]].merge(pilot[["patient_key","center","outcome_true"]],on="patient_key",validate="one_to_one").sort_values(["outer_fold","patient_key"]).reset_index(drop=True)
        atomic_csv(output/"mosaic_full_cohort_manifest.csv",pilot); atomic_csv(output/"mosaic_full_outer_folds.csv",pilot_folds)
    else:
        manifest_path=output/"mosaic_pilot16_manifest.csv"; cache_manifest_dir=Path(args.cache_dir)/"pilot_manifest"; cache_manifest_dir.mkdir(parents=True,exist_ok=True); cached_manifest=cache_manifest_dir/manifest_path.name
        source_manifest=manifest_path if manifest_path.exists() else cached_manifest
        pilot=select_pilot16(candidate_frame,source_manifest,args.resume); atomic_csv(manifest_path,pilot); atomic_csv(cached_manifest,pilot)
        folds_path=output/"mosaic_pilot16_outer_folds.csv"; cached_folds=cache_manifest_dir/folds_path.name; source_folds=folds_path if folds_path.exists() else cached_folds
        if args.resume and source_folds.exists(): pilot_folds=pd.read_csv(source_folds); validate_pilot_outer_folds(pilot_folds,pilot)
        else: pilot_folds=build_pilot_outer_folds(pilot)
        atomic_csv(folds_path,pilot_folds); atomic_csv(cached_folds,pilot_folds)
    selected=set(pilot.patient_key.astype(str)); raws=[x for x in raws if x.patient_key in selected]
    return feature_cache,raw_cache,outcomes,original,target_lookup,raws,pilot,pilot_folds,alignment,target_alignment,target_distribution,raw_audit,feature_audit,exclusion_audit


def p2_ledger(args,feature_cache,outcomes,original,target_lookup,pilot):
    if args.p2_oof_channel_ledger:
        ledger=load_p2_oof_channel_ledger(args.p2_oof_channel_ledger,original,args.strict); return ledger[ledger.patient_key.isin(pilot.patient_key)].copy(),{"source":"provided_original_oof_ledger","p2_fold_safe":True}
    if not all((args.p2_checkpoint_root,args.p2_runtime_root,args.p2_training_manifest)): raise ValueError("P2 OOF requires --p2_oof_channel_ledger or checkpoint/runtime/training-manifest inputs")
    from run_ngbr_full import _load_p2_fold
    from neuroez_c.task2.p2_adapter import P23Runtime,load_p2_args,locate_p2_config
    runtime=P23Runtime(args.p2_runtime_root); p2args=load_p2_args(locate_p2_config(args.p2_checkpoint_root,args.p2_config),feature_cache=args.feature_cache); labels=outcomes.set_index("patient_key").outcome_true.astype(int).to_dict(); cohort=set(original.patient_key.astype(str)); tables=[]; safety=[]
    shim=argparse.Namespace(**vars(args)); shim.biomarker_cache_dir=args.cache_dir; shim.negative_control="none"; shim.recompute_p2=args.recompute_nez
    selected=set(pilot.patient_key.astype(str))
    for fold in sorted(original.outer_fold.unique()):
        test=set(original.loc[original.outer_fold==fold,"patient_key"].astype(str)); records,table,audit=_load_p2_fold(shim,int(fold),cohort-test,test,runtime,p2args,feature_cache,labels,target_lookup); tables.append(table[table.patient_key.isin(test&selected)]); safety.append(audit)
    ledger=pd.concat(tables,ignore_index=True).rename(columns={"q_nez":"probability_nez"})
    return ledger,{"source":"generated_from_original_five_fold_checkpoints","p2_fold_safe":all(x["P2_fold_safe"] for x in safety),"folds":safety}


def preview_p2_safety(args,original,pilot):
    if args.p2_oof_channel_ledger:
        ledger=load_p2_oof_channel_ledger(args.p2_oof_channel_ledger,original,args.strict)
        missing=set(pilot.patient_key.astype(str))-set(ledger.patient_key.astype(str))
        if missing: raise ValueError(f"P2 OOF ledger misses pilot patients: {sorted(missing)}")
        return {"source":"provided_original_oof_ledger","p2_fold_safe":True,"n_pilot_channel_rows":int(ledger.patient_key.isin(pilot.patient_key).sum())}
    if not all((args.p2_checkpoint_root,args.p2_training_manifest)): raise ValueError("strict P2 safety audit requires checkpoints and training manifest")
    from run_ngbr_full import _checkpoint_fold
    rows=[]
    for fold in sorted(original.outer_fold.unique()):
        checkpoint=Path(args.p2_checkpoint_root)/f"fold_{int(fold)}"/"best_model.pt"; test=set(original.loc[original.outer_fold==fold,"patient_key"].astype(str))
        if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
        metadata=_checkpoint_fold(checkpoint); training=checkpoint_training_subjects(args.p2_training_manifest,int(fold)); assert_checkpoint_safe(training,test)
        safe=int(metadata)==int(fold); rows.append({"outer_fold":int(fold),"checkpoint":str(checkpoint),"checkpoint_outer_fold":metadata,"checkpoint_outer_fold_matches":safe,"training_manifest_leakage_verified":True,"P2_fold_safe":safe})
        if args.strict and not safe: raise ValueError(f"P2 checkpoint fold mismatch: expected={fold}, found={metadata}")
    return {"source":"original_five_fold_checkpoints","p2_fold_safe":all(x["P2_fold_safe"] for x in rows),"folds":rows}


def extract(args,raws,pilot,p2,cache):
    raw_id=file_identity(args.raw_cache); feat_id=file_identity(args.feature_cache); p2_by={k:v for k,v in p2.groupby("patient_key")}
    frag_rows=[]; cii_rows=[]; escape_rows=[]; recruit_rows=[]; spec_rows=[]; vi_rows=[]; quality={k:[] for k in ("fragility","cii","recruitment","spectral")}; maps={k:[] for k in ("fragility","pte","cii","spectral")}
    by_patient={}
    progress=tqdm(raws,desc="MOSAIC biomarkers",unit="seizure",dynamic_ncols=True)
    cache_hits={"fragility":0,"pte":0,"recruitment":0,"spectral":0}; cache_misses={k:0 for k in cache_hits}
    for raw in progress:
        progress.set_description_str(f"MOSAIC {raw.patient_key[:24]}"); mosaic=from_ngbr(raw); channel=p2_by[raw.patient_key].copy(); channel["channel"]=channel.channel.astype(str)
        p2rec=type("P",(),{})(); p2rec.patient_key=raw.patient_key; p2rec.center=raw.center; p2rec.channel_names=channel.channel.tolist(); p2rec.final_nez_probability=channel.probability_nez.to_numpy(float); p2rec.direct_nez_probability=None; p2rec.valid_channel_mask=channel.get("valid",pd.Series(True,index=channel.index)).astype(bool).to_numpy(); p2rec.clinical_target_mask=channel.get("clinical_target",channel.get("true_ez",pd.Series(False,index=channel.index))).astype(bool).to_numpy()
        aligned=align_p2_to_raw(p2rec,raw,strict=args.strict); q=aligned.final_nez_probability
        patient=by_patient.setdefault(raw.patient_key,{"fragility":[],"cii":[],"recruitment":[],"spectral":[],"vi":[],"nez_channel":None})
        if patient["nez_channel"] is None:
            patient["nez_channel"]=pd.DataFrame({"probability_nez":q,"true_ez":raw.clinical_target_mask,"valid":aligned.valid_channel_mask,"channel":raw.channel_names})
        if not args.disable_fragility:
            cfg=FragilityConfig(); sig=trajectory_signature(mosaic,"fragility",FRAG_VERSION,vars(cfg),raw_id,feat_id); result=cache.load("fragility_trajectory",raw.patient_key,raw.seizure_id,sig,args.recompute_fragility)
            if result is None: cache_misses["fragility"]+=1; result=compute_fragility_trajectory(mosaic,cfg,args.biomarker_workers); cache.save("fragility_trajectory",raw.patient_key,raw.seizure_id,sig,result)
            else: cache_hits["fragility"]+=1
            trajectories,window_audit,fq=result; frow,fvi=seizure_fragility_features(trajectories,mosaic.true_ez_mask,q); frow.update(patient_key=raw.patient_key,seizure_id=raw.seizure_id); fvi.update(patient_key=raw.patient_key,seizure_id=raw.seizure_id); frag_rows.append(frow); vi_rows.append(fvi); patient["fragility"].append(frow); patient["vi"].append(fvi); quality["fragility"].append(fq)
            maps["fragility"].append((raw.patient_key,raw.seizure_id,trajectories))
        cii_combined={}; escape_combined={}; cii_valid=0
        if not args.disable_pte:
            valid=np.where(mosaic.valid_channel_mask)[0]
            for band in ("low","high"):
                pcfg=PTEConfig(); sig=trajectory_signature(mosaic,f"pte_{band}",PTE_VERSION,{**vars(pcfg),"band":band},raw_id,feat_id); pte=cache.load("pte",raw.patient_key,raw.seizure_id+band,sig,args.recompute_cii)
                if pte is None: cache_misses["pte"]+=1; pte=compute_pte_windows(mosaic.signal[valid],mosaic.sampling_rate,mosaic.onset_sample,band,pcfg,args.biomarker_workers); cache.save("pte",raw.patient_key,raw.seizure_id+band,sig,pte)
                else: cache_hits["pte"]+=1
                times,matrices,pq=pte
                if pq.get("valid"):
                    row,escape,cii=seizure_cii_features(times,matrices,mosaic.true_ez_mask[valid],band); cii_combined.update(row); escape_combined.update(escape); maps["pte"].append((raw.patient_key,raw.seizure_id,band,times,matrices)); maps["cii"].append((raw.patient_key,raw.seizure_id,band,times,cii)); cii_valid+=1
            cii_combined.update(patient_key=raw.patient_key,seizure_id=raw.seizure_id); escape_combined.update(patient_key=raw.patient_key,seizure_id=raw.seizure_id); cii_rows.append(cii_combined); escape_rows.append(escape_combined); patient["cii"].append(cii_combined); quality["cii"].append({"patient_key":raw.patient_key,"seizure_id":raw.seizure_id,"valid_bands":cii_valid,"cii_valid":cii_valid>0})
        sig=trajectory_signature(mosaic,"recruitment","ei+propagation_existing",{},raw_id,feat_id); rr=cache.load("recruitment",raw.patient_key,raw.seizure_id,sig,args.recompute_recruitment)
        if rr is None: cache_misses["recruitment"]+=1; rr=seizure_recruitment_features(raw); cache.save("recruitment",raw.patient_key,raw.seizure_id,sig,rr)
        else: cache_hits["recruitment"]+=1
        rrow,_,_,rq=rr; rrow.update(patient_key=raw.patient_key,seizure_id=raw.seizure_id); recruit_rows.append(rrow); patient["recruitment"].append(rrow); quality["recruitment"].append(rq)
        sig=trajectory_signature(mosaic,"spectral",SPEC_VERSION,{"window_sec":.5,"step_sec":.25},raw_id,feat_id); sr=cache.load("spectral",raw.patient_key,raw.seizure_id,sig,args.recompute_spectral)
        if sr is None: cache_misses["spectral"]+=1; sr=compute_spectral_state(mosaic); cache.save("spectral",raw.patient_key,raw.seizure_id,sig,sr)
        else: cache_hits["spectral"]+=1
        state,sq=sr; srow,svi=seizure_spectral_features(state,mosaic.true_ez_mask,q); srow.update(patient_key=raw.patient_key,seizure_id=raw.seizure_id); svi.update(patient_key=raw.patient_key,seizure_id=raw.seizure_id); spec_rows.append(srow); vi_rows.append(svi); patient["spectral"].append(srow); patient["vi"].append(svi); quality["spectral"].append(sq); maps["spectral"].append((raw.patient_key,raw.seizure_id,state)); progress.set_postfix_str(" ".join(f"{k}:H{cache_hits[k]}/M{cache_misses[k]}" for k in cache_hits),refresh=True)
    labels=pilot.set_index("patient_key").outcome_true.to_dict(); centers=pilot.set_index("patient_key").center.to_dict(); expert_frames={k:[] for k in ("fragility","cii","recruitment","spectral","nez")}; evidence=[]
    for patient,data in by_patient.items():
        expert_frames["fragility"].append({"patient_key":patient,"center":centers[patient],"outcome_true":labels[patient],**patient_fragility_features(data["fragility"])}); expert_frames["cii"].append({"patient_key":patient,"center":centers[patient],"outcome_true":labels[patient],**patient_cii_features(data["cii"])}); expert_frames["recruitment"].append({"patient_key":patient,"center":centers[patient],"outcome_true":labels[patient],**patient_recruitment_features(data["recruitment"])}); expert_frames["spectral"].append({"patient_key":patient,"center":centers[patient],"outcome_true":labels[patient],**patient_spectral_features(data["spectral"])})
        nez=patient_nez_features(data["nez_channel"]); expert_frames["nez"].append({"patient_key":patient,"center":centers[patient],"outcome_true":labels[patient],**nez})
        frag_capture=[]; frag_residual=[]; spec_capture=[]; spec_residual=[]
        for row in data["vi"]:
            for key,value in row.items():
                if "capture__mean" in key:
                    (frag_capture if key.startswith("fragility") else spec_capture).append(value)
                if "outside_residual__mean" in key:
                    (frag_residual if key.startswith("fragility") else spec_residual).append(value)
        recruit=data["recruitment"]; cii=data["cii"]
        safe_mean=lambda x:float(np.nanmean(x)) if np.isfinite(np.asarray(x,float)).any() else np.nan
        evidence.append({"patient_key":patient,
                         "capture_fragility":safe_mean(frag_capture),"capture_recruitment":safe_mean([r.get("ei_capture",np.nan) for r in recruit]),"capture_spectral":safe_mean(spec_capture),"capture_nez":nez["non_nez_capture"],
                         "outside_residual_fragility":safe_mean(frag_residual),"outside_residual_cii":safe_mean([v for row in cii for k,v in row.items() if "max_normalized_escape" in k]),"outside_residual_recruitment":safe_mean([r.get("outside_ei_q90",np.nan) for r in recruit]),"outside_residual_spectral":safe_mean(spec_residual),
                         "positive_net_escape":safe_mean([max(0,v) for row in cii for k,v in row.items() if "mean_net_causal_escape" in k]),"outside_recruited_3s_fraction":safe_mean([r.get("outside_recruited_3s_fraction",np.nan) for r in recruit])})
    return {k:pd.DataFrame(v) for k,v in expert_frames.items()},pd.DataFrame(evidence),{"frag":frag_rows,"cii":cii_rows,"escape":escape_rows,"recruit":recruit_rows,"spec":spec_rows,"vi":vi_rows,"quality":quality,"maps":maps}


def run_models(args,expert_frames,evidence,pilot,folds):
    inner_all=[]; outer_all=[]; meta_rows=[]; predictions=[]; coefficients=[]
    active_experts=tuple(x for x in ("fragility","cii","recruitment","spectral","nez") if not (x=="fragility" and args.disable_fragility) and not (x=="cii" and args.disable_pte))
    labels=pilot.set_index("patient_key").outcome_true.to_dict(); centers=pilot.set_index("patient_key").center.to_dict()
    for fold in tqdm(sorted(folds.outer_fold.unique()),desc="MOSAIC outer CV",unit="fold",dynamic_ncols=True):
        test=set(folds.loc[folds.outer_fold==fold,"patient_key"].astype(str)); train=set(pilot.patient_key.astype(str))-test
        outer,_=fit_outer_experts(expert_frames,train,test,int(fold),args.seed,args.n_jobs,active_experts); outer_all.append(outer)
        if args.skip_inner_crossfit:
            train_ev=evidence[evidence.patient_key.isin(train)]; test_ev=evidence[evidence.patient_key.isin(test)]; test_meta=evidence_bottleneck(outer,test_ev,train_ev)
            wide=outer.pivot(index="patient_key",columns="expert",values="probability_success")
            for expert in ("fragility","cii","recruitment","spectral","nez"):
                if expert not in wide: wide[expert]=.5
            probability=wide.loc[test_meta.patient_key,list(active_experts)].mean(axis=1).to_numpy(float)
            for patient,p in zip(test_meta.patient_key,probability):
                ev=test_meta.set_index("patient_key").loc[patient]; predictions.append({"patient_key":patient,"center":centers[patient],"outcome_true":labels[patient],"outer_fold":fold,**{f"p_{x}":wide.loc[patient,x] for x in wide.columns},"capture_consensus":ev.capture_consensus,"outside_residual_consensus":ev.outside_residual_consensus,"causal_spread_coupling":ev.causal_spread_coupling,"probability_success":float(p),"prediction_05":int(p>=.5),"seed":args.seed,"n_seizures":int(pilot.set_index("patient_key").loc[patient,"n_seizures"]),"n_valid_channels":int(pilot.set_index("patient_key").loc[patient,"n_valid_channels"])})
            meta_rows.append(test_meta.assign(outer_fold=fold,partition="outer_test_fast_equal_average")); continue
        inner=lopo_crossfit_experts(expert_frames,train,int(fold),args.seed,args.n_jobs,active_experts); inner_all.append(inner)
        train_ev=evidence[evidence.patient_key.isin(train)]; inner_meta=evidence_bottleneck(inner,train_ev,train_ev).merge(pd.DataFrame({"patient_key":sorted(train),"outcome_true":[labels[x] for x in sorted(train)]}),on="patient_key"); meta=fit_meta(inner_meta,args.seed+int(fold),forbidden_patients=test)
        test_ev=evidence[evidence.patient_key.isin(test)]; test_meta=evidence_bottleneck(outer,test_ev,train_ev); probability=meta.predict(test_meta)
        coef=np.ravel(meta.model.coef_)
        coefficients += [{"outer_fold":fold,"feature":name,"coefficient":value} for name,value in zip(META_FEATURES,coef)]
        wide=outer.pivot(index="patient_key",columns="expert",values="probability_success")
        for expert in ("fragility","cii","recruitment","spectral","nez"):
            if expert not in wide: wide[expert]=.5
        for patient,p in zip(test_meta.patient_key,probability):
            ev=test_meta.set_index("patient_key").loc[patient]; predictions.append({"patient_key":patient,"center":centers[patient],"outcome_true":labels[patient],"outer_fold":fold,**{f"p_{x}":wide.loc[patient,x] for x in wide.columns},"capture_consensus":ev.capture_consensus,"outside_residual_consensus":ev.outside_residual_consensus,"causal_spread_coupling":ev.causal_spread_coupling,"probability_success":float(p),"prediction_05":int(p>=.5),"seed":args.seed,"n_seizures":int(pilot.set_index("patient_key").loc[patient,"n_seizures"]),"n_valid_channels":int(pilot.set_index("patient_key").loc[patient,"n_valid_channels"])})
        meta_rows.append(test_meta.assign(outer_fold=fold,partition="outer_test")); meta_rows.append(inner_meta.assign(outer_fold=fold,partition="inner_crossfit_train"))
    inner_output=pd.concat(inner_all,ignore_index=True) if inner_all else pd.DataFrame(columns=["patient_key","outer_fold","expert","probability_success","true_label","fit_patient_keys","held_out_verified"])
    coefficient_output=pd.DataFrame(coefficients,columns=["outer_fold","feature","coefficient"])
    return inner_output,pd.concat(outer_all,ignore_index=True),pd.concat(meta_rows,ignore_index=True),pd.DataFrame(predictions),coefficient_output


def main():
    started=time.perf_counter(); args=parser().parse_args(); np.random.seed(args.seed); out=Path(args.output_dir); out.mkdir(parents=True,exist_ok=True); atomic_json(out/"run_args.json",vars(args))
    feature_cache,raw_cache,outcomes,original,target_lookup,raws,pilot,folds,alignment,target_alignment,target_distribution,raw_audit,feature_audit,exclusion_audit=prepare(args)
    atomic_csv(out/"mosaic_channel_alignment_audit.csv",alignment[alignment.patient_key.isin(pilot.patient_key)]); atomic_csv(out/"mosaic_true_ez_audit.csv",target_alignment[target_alignment.patient_key.isin(pilot.patient_key)]); atomic_csv(out/"mosaic_exclusion_audit.csv",exclusion_audit); atomic_json(out/"mosaic_schema_audit.json",{"raw_records":len(raws),"feature_axis_audited":len(feature_audit),"raw_axis_audited":len(raw_audit),"normalized_channel_names":True,"unmatched_channels_are_not_outside":True,"cohort_target_distribution":target_distribution.to_dict("records")})
    p2preview=preview_p2_safety(args,original,pilot); atomic_json(out/"mosaic_p2_oof_safety_audit.json",p2preview); phases=[]
    for raw in raws:
        for name,x in phase_intervals(raw.onset_sample,raw.sampling_rate,raw.signal.shape[1]).items(): phases.append({"patient_key":raw.patient_key,"seizure_id":raw.seizure_id,"phase":name,"actual_duration_sec":x.actual_duration_sec,"valid":x.valid})
    atomic_csv(out/"mosaic_phase_duration_audit.csv",pd.DataFrame(phases)); audit=protocol_audit(pilot,folds,args.seed,args.cache_dir,0)
    if args.full_cohort:
        audit.update({"pilot_only":False,"full_cohort":True,"cohort_size":int(len(pilot)),"outer_folds":int(folds.outer_fold.nunique()),"paper_valid":False,"status":"FULL_COHORT_ABLATION_NOT_PAPER_VALID"})
    if args.audit_only:
        audit.update({"audit_only":True,"p2_fold_safe":p2preview["p2_fold_safe"]}); atomic_json(out/"mosaic_protocol_audit.json",audit); atomic_text(out/"MOSAIC_PILOT16_REPORT.md","# MOSAIC-Outcome Audit\n\nNOT_PAPER_VALID\n\nNo biomarkers or models were computed (`--audit_only`).\n"); print(json.dumps({"pipeline":"MOSAIC-Outcome","audit_only":True,"patients":len(pilot),"output":str(out.resolve())},indent=2)); return 0
    p2,p2audit=p2_ledger(args,feature_cache,outcomes,original,target_lookup,pilot); atomic_json(out/"mosaic_p2_oof_safety_audit.json",p2audit)
    cache=AtomicCache(args.cache_dir,args.resume,args.fragility_trajectory_cache_dir); experts,evidence,art=extract(args,raws,pilot,p2,cache)
    for name,key in (("mosaic_fragility_quantile_features.csv","frag"),("mosaic_cii_quantile_features.csv","cii"),("mosaic_causal_escape_features.csv","escape"),("mosaic_recruitment_features.csv","recruit"),("mosaic_spectral_quantile_features.csv","spec"),("mosaic_virtual_intervention_features.csv","vi")): atomic_csv(out/name,_frame(art[key]))
    for expert in ("fragility","cii","recruitment","spectral"): atomic_csv(out/f"mosaic_{expert}_quality_audit.csv",_frame(art["quality"][expert]))
    atomic_csv(out/"mosaic_nez_expert_features.csv",experts["nez"]); allpatient=pd.concat([v.assign(expert=k) for k,v in experts.items()],ignore_index=True,sort=False); atomic_csv(out/"mosaic_patient_expert_features.csv",allpatient)
    for key,filename in (("fragility","mosaic_fragility_window_map.npz"),("pte","mosaic_pte_window_matrices.npz"),("cii","mosaic_cii_channel_trajectory.npz"),("spectral","mosaic_spectral_window_maps.npz")): atomic_npz(out/filename,items=np.asarray(art["maps"][key],dtype=object))
    inner,outer,meta,pred,coef=run_models(args,experts,evidence,pilot,folds); atomic_csv(out/"mosaic_inner_crossfit_expert_predictions.csv",inner); atomic_csv(out/"mosaic_outer_expert_predictions.csv",outer); atomic_csv(out/"mosaic_meta_features.csv",meta); atomic_csv(out/"mosaic_meta_coefficients.csv",coef); atomic_csv(out/"oof_patient_predictions.csv",pred)
    fold_metrics=pd.DataFrame([{"outer_fold":fold,**metric_row(pred[pred.outer_fold==fold])} for fold in sorted(pred.outer_fold.unique())]); scope="full_cohort" if args.full_cohort else "pooled_pilot16_fast_screen" if args.skip_inner_crossfit else "pooled_pilot16"; summary=pd.DataFrame([{"scope":scope,**metric_row(pred)}]); atomic_csv(out/"fold_metrics.csv",fold_metrics); atomic_csv(out/"summary_metrics.csv",summary); atomic_csv(out/"bootstrap_ci.csv",bootstrap_ci(pred,args.bootstrap_repeats,args.seed))
    matrix=pd.crosstab(pred.outcome_true,pred.prediction_05).reindex(index=[0,1],columns=[0,1],fill_value=0); atomic_csv(out/"confusion_matrix.csv",matrix.rename_axis("outcome_true").reset_index())
    diagnostics=[]
    for expert,group in outer.groupby("expert"):
        f=group.rename(columns={"true_label":"outcome_true","probability_success":"probability_success"}); diagnostics.append({"expert":expert,**metric_row(f)})
    atomic_csv(out/"expert_diagnostic_metrics.csv",pd.DataFrame(diagnostics)); audit=protocol_audit(pilot,folds,args.seed,args.cache_dir,len(folds.outer_fold.unique())); audit["p2_fold_safe"]=p2audit["p2_fold_safe"]; audit["inner_crossfit_skipped"]=bool(args.skip_inner_crossfit); audit["fragility_disabled"]=bool(args.disable_fragility); audit["pte_disabled"]=bool(args.disable_pte); audit["ablation_run"]=bool(args.disable_fragility or args.disable_pte); audit["full_cohort"]=bool(args.full_cohort); audit["cohort_size"]=int(len(pilot)); audit["expert_crossfit"]="disabled_fast_screen" if args.skip_inner_crossfit else "leave_one_patient_out_within_outer_train"; audit["meta_trained_on_crossfitted_expert_predictions"]=not args.skip_inner_crossfit; audit["fusion"]="equal_average_of_outer_test_experts" if args.skip_inner_crossfit else "eight_feature_ridge_meta_with_neutral_expert_logits" if args.disable_fragility or args.disable_pte else "eight_feature_ridge_meta"; audit["status"]="FULL_COHORT_NO_FRAGILITY_NO_PTE_ABLATION_NOT_PAPER_VALID" if args.full_cohort and args.disable_fragility and args.disable_pte else "FAST_SCREEN_NO_INNER_CROSSFIT_NOT_MOSAIC_FULL" if args.skip_inner_crossfit else "FRAGILITY_AND_PTE_ABLATION_PILOT_NOT_MOSAIC_FULL" if args.disable_fragility and args.disable_pte else "PTE_ABLATION_PILOT_NOT_MOSAIC_FULL" if args.disable_pte else "FRAGILITY_ABLATION_PILOT_NOT_MOSAIC_FULL" if args.disable_fragility else "PILOT_ONLY_NOT_PAPER_VALID"; atomic_json(out/"mosaic_protocol_audit.json",audit)
    elapsed=time.perf_counter()-started; pooled=summary.iloc[0].to_dict(); coverage={name:float(pd.DataFrame(rows).filter(regex="valid$").astype(bool).any(axis=1).mean()) if rows and not pd.DataFrame(rows).filter(regex="valid$").empty else np.nan for name,rows in art["quality"].items()}
    run_status="FAST_SCREEN_NO_INNER_CROSSFIT_NOT_MOSAIC_FULL" if args.skip_inner_crossfit else "FRAGILITY_AND_PTE_ABLATION_PILOT_NOT_MOSAIC_FULL" if args.disable_fragility and args.disable_pte else "PTE_ABLATION_PILOT_NOT_MOSAIC_FULL" if args.disable_pte else "FRAGILITY_ABLATION_PILOT_NOT_MOSAIC_FULL" if args.disable_fragility else "PILOT_ONLY_NOT_PAPER_VALID"
    protocol_line="- Fast screen: inner LOPO and the meta model were disabled; outer-test expert probabilities were averaged equally." if args.skip_inner_crossfit else "- Each active expert used LOPO probabilities inside the current outer-train set; the meta model never saw in-sample expert probabilities."
    fragility_line="- Fragility: disabled for this ablation; p_fragility is fixed at neutral 0.5 and is not fitted." if args.disable_fragility else "- Fragility: full structured-column unit-circle-resolvent trajectories and phase/region quantiles."
    cii_line="- CII/PTE: disabled for this ablation; p_cii is fixed at neutral 0.5 and is not fitted." if args.disable_pte else "- CII: discrete directed phase-transfer entropy and causal escape."
    report=["# MOSAIC-Outcome Pilot16 Report","",f"**{run_status}**","",
            "## Cohort and protocol",f"- Patients: {len(pred)}; centers: 4; success/failure: {int(pred.outcome_true.sum())}/{int((pred.outcome_true==0).sum())}.","- Four fixed patient-wise outer folds; each test fold has one patient per center and a 2/2 outcome split.","- All available valid seizures were pooled. Center, patient ID, seizure count, and channel count were audit-only fields.",protocol_line,"",
            "## Fixed experts",fragility_line,cii_line,"- Recruitment: the existing EI and ictal-propagation implementations.","- Spectral: ranked relative band power and low spectral entropy.","- NEZ: original five-fold P2 OOF channel probabilities only.","",
            "## Pooled pilot metrics",*(f"- {key}: {pooled.get(key):.4f}" for key in ("accuracy","balanced_accuracy","macro_f1","auroc","success_auprc","failure_auprc","success_precision","success_recall","failure_precision","failure_recall","brier","ece")),"",
            "## Mechanism validity",*(f"- {key}: {value:.3f}" if np.isfinite(value) else f"- {key}: unavailable" for key,value in coverage.items()),"",
            "## Runtime",f"- Pilot wall time: {elapsed/60:.1f} minutes.",f"- Linear 139-patient projection: {elapsed*139/16/3600:.1f} hours; actual time depends strongly on trajectory/PTE cache hits.","",
            "## Interpretation and limitations","- This run checks computability and preliminary signal only. It is not paper-valid.","- Bootstrap intervals reuse fixed OOF predictions and are not final generalization intervals.","- The 16-patient deterministic quality-quantile sample is too small for stable center-specific claims.","- True-EZ is a clinician-defined intervention proxy, not biological ground truth and not an exact resection mask.","- High-band PTE is unavailable when the sampling rate cannot support an upper cutoff above 45 Hz.","- No result, feature value, or prediction was used to select patients, tune models, choose modules, or tune the primary 0.5 threshold.",""]
    atomic_text(out/"MOSAIC_PILOT16_REPORT.md","\n".join(report))
    print(json.dumps({"pipeline":"MOSAIC-Outcome","profile":PROFILE,"n_predictions":len(pred),"output":str(out.resolve()),"paper_valid":False},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
