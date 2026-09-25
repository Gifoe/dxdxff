from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch

PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parent
for path in (REPO, PROJECT):
    if str(path) not in sys.path: sys.path.insert(0, str(path))

from neuroez_c.task2.clean_nez_prototype import CleanNEZPrototype
from neuroez_c.task2.clinical_target import build_clinical_target_lookup
from neuroez_c.task2.data import filtered_cache, load_cache, load_graph_cache
from neuroez_c.task2.exclusions import load_exclusion_manifest
from neuroez_c.task2.metrics import FIXED_THRESHOLD_SOURCE, bootstrap_metrics, compute_metrics
from neuroez_c.task2.nvr_features import ALL_SCALAR_FEATURES, build_nvr_features, make_counterfactual_targets, scalar_features_for_profile
from neuroez_c.task2.nvr_negative_controls import graph_permutation, target_permutation
from neuroez_c.task2.nvr_outcome_model import NVROutcomeModel
from neuroez_c.task2.nvr_training import fit_nvr_normalizer, make_nvr_loader, predict_nvr, train_nvr_fixed_epochs
from neuroez_c.task2.outcomes import load_outcome_table
from neuroez_c.task2.p2_adapter import P23Runtime, P2ExportAdapter, load_p2_args, locate_p2_config
from neuroez_c.task2.profiles import get_nvr_profile, nvr_profile_names
from neuroez_c.task2.protocol import assert_checkpoint_safe, checkpoint_training_subjects, load_fold_manifest, manifest_hash, manifest_patient_keys
from neuroez_c.task2.training import make_loader, prepare_examples, seed_everything


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Separate frozen-P2 NEZ-Verified Residual Network outcome model")
    value.add_argument("--profile", required=True, choices=nvr_profile_names())
    value.add_argument("--outcome_table", default=os.getenv("DRE_TASK2_OUTCOME_TABLE", "cache://patient_index"))
    value.add_argument("--feature_cache", default=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH"), required=os.getenv("DRE_TASK1_FEATURE_CACHE_PATH") is None)
    value.add_argument("--graph_cache", default=os.getenv("DRE_TASK2_GRAPH_CACHE"))
    value.add_argument("--p2_checkpoint_root", default=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT"), required=os.getenv("DRE_TASK1_P2_CHECKPOINT_ROOT") is None)
    value.add_argument("--p2_runtime_root", default=os.getenv("DRE_TASK1_P2_RUNTIME_ROOT", str(REPO/"P23_TRN_NEZ_80")))
    value.add_argument("--p2_config")
    value.add_argument("--p2_training_manifest", default=os.getenv("DRE_TASK1_P2_TRAINING_MANIFEST"))
    value.add_argument("--fold_manifest", default=os.getenv("DRE_TASK2_FOLD_MANIFEST"), required=os.getenv("DRE_TASK2_FOLD_MANIFEST") is None)
    value.add_argument("--exclusion_manifest", default=os.getenv("DRE_TASK2_EXCLUSION_MANIFEST", str(PROJECT/"configs"/"data_exclusions.csv")))
    value.add_argument("--protocol", choices=("quick", "outer_cv"), default="outer_cv")
    value.add_argument("--outer_folds", type=int, default=5); value.add_argument("--max_outer_folds", type=int, default=0); value.add_argument("--max_patients", type=int, default=0)
    value.add_argument("--epochs", type=int, default=0); value.add_argument("--batch_size", type=int, default=8); value.add_argument("--seed", type=int, default=42)
    value.add_argument("--bootstrap_repeats", type=int, default=2000); value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    value.add_argument("--negative_control", choices=("none", "NVR_TARGET_PERMUTATION", "NVR_P2_PERMUTATION", "NVR_GRAPH_PERMUTATION", "target_permutation", "p2_channel_permutation", "graph_permutation"), default="none")
    value.add_argument("--evidence_cache_dir", default=os.getenv("DRE_TASK2_NVR_EVIDENCE_CACHE_DIR"))
    value.add_argument("--output_dir", default=os.getenv("DRE_TASK2_NVR_OUTPUT_DIR"), required=os.getenv("DRE_TASK2_NVR_OUTPUT_DIR") is None)
    value.add_argument("--strict", action="store_true"); value.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    return value


def _balanced_take(subjects: Sequence[str], labels: dict[str, int], count: int) -> list[str]:
    values = sorted(map(str, subjects))
    if count <= 0 or len(values) <= count: return values
    buckets = {label: [p for p in values if labels[p] == label] for label in (0, 1)}; selected = []
    while len(selected) < count and any(buckets.values()):
        for label in (0, 1):
            if buckets[label] and len(selected) < count: selected.append(buckets[label].pop(0))
    return selected


def _checkpoint_fold(path: Path) -> int | None:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    for mapping in (payload, payload.get("metadata", {}) if isinstance(payload, dict) else {}):
        if isinstance(mapping, dict):
            for key in ("outer_fold", "fold", "fold_idx"):
                if key in mapping: return int(mapping[key])
    return None


def _extract_evidence(adapter: P2ExportAdapter, loader, device: str) -> list[dict[str, Any]]:
    adapter.eval(); records = []
    with torch.no_grad():
        for batch in loader:
            export = {key: value.detach().cpu() for key, value in adapter(batch).items() if torch.is_tensor(value)}
            for p, subject in enumerate(batch["subject_id"]):
                c = int(batch["channel_mask"][p].sum()); s = int(batch["seizure_mask"][p].sum())
                record: dict[str, Any] = {"patient_key": str(subject), "center": str(batch["center"][p]), "outcome_target": int(batch["outcome_target"][p]),
                    "clinical_target_mask": batch["clinical_target_mask"][p:p+1, :c].bool().cpu(), "channel_mask": batch["channel_mask"][p:p+1, :c].bool().cpu(),
                    "seizure_mask": batch["seizure_mask"][p:p+1, :s].bool().cpu(), "seizure_channel_mask": batch["seizure_channel_mask"][p:p+1, :s, :c].bool().cpu()}
                for key in ("patient_channel_embedding", "final_nez_logit", "final_score_nez", "direct_nez_logit"):
                    record[key] = export[key][p:p+1, :c]
                record["seizure_channel_embedding"] = export["seizure_channel_embedding"][p:p+1, :s, :c]
                record["seizure_nez_probability"] = export["seizure_nez_probability"][p:p+1, :s, :c]
                if "graph_adjacency" in batch:
                    record["graph_adjacency"] = batch["graph_adjacency"][p:p+1, :s, :, :c, :c].cpu()
                    record["graph_phase_channel_mask"] = batch["graph_phase_channel_mask"][p:p+1, :s, :, :c].cpu()
                    record["graph_valid"] = batch["graph_valid"][p:p+1, :s].cpu()
                records.append(record)
    return records


def _fit_prototypes(records: Sequence[dict[str, Any]], train_subjects: set[str]) -> tuple[CleanNEZPrototype, CleanNEZPrototype]:
    patient_rows, seizure_rows = [], []
    for record in records:
        if record["patient_key"] not in train_subjects: continue
        outside = record["channel_mask"][0] & ~record["clinical_target_mask"][0]
        patient_rows.append(record["patient_channel_embedding"][0, outside])
        valid = record["seizure_mask"][0].unsqueeze(-1) & record["seizure_channel_mask"][0] & outside.unsqueeze(0)
        seizure_rows.append(record["seizure_channel_embedding"][0][valid])
    return CleanNEZPrototype("patient_channel_embedding").fit(patient_rows), CleanNEZPrototype("seizure_channel_embedding").fit(seizure_rows)


def _permute_p2_evidence(evidence: dict[str, Any], seed: int) -> dict[str, Any]:
    output = dict(evidence); valid = evidence["channel_mask"]; generator = torch.Generator().manual_seed(seed)
    idx = torch.where(valid[0])[0]; order = idx[torch.randperm(idx.numel(), generator=generator)]
    for key in ("patient_channel_embedding", "final_nez_logit", "final_score_nez", "direct_nez_logit"):
        output[key] = evidence[key].clone(); output[key][0, idx] = evidence[key][0, order]
    for key in ("seizure_channel_embedding", "seizure_nez_probability"):
        output[key] = evidence[key].clone(); output[key][0, :, idx] = evidence[key][0, :, order]
    return output


def _variant(value: dict[str, Any], names: tuple[str, ...]) -> dict[str, torch.Tensor]:
    return {"scalar_values": torch.stack([value["scalars"][name][0] for name in names]),
            **{name: value[name][0].detach().cpu() for name in ("patient_channel_embedding", "channel_scalars", "target", "channel_mask", "reliable_abnormality")}}


def _make_records(raw: Sequence[dict[str, Any]], patient_proto: CleanNEZPrototype, seizure_proto: CleanNEZPrototype, profile: str, control: str, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    names = scalar_features_for_profile(profile); records, audits = [], []
    control = {"NVR_TARGET_PERMUTATION":"target_permutation", "NVR_P2_PERMUTATION":"p2_channel_permutation", "NVR_GRAPH_PERMUTATION":"graph_permutation"}.get(control, control)
    for number, source in enumerate(raw):
        evidence = dict(source)
        if control == "target_permutation": evidence["clinical_target_mask"] = target_permutation(evidence["clinical_target_mask"], evidence["channel_mask"], seed+number)
        elif control == "p2_channel_permutation": evidence = _permute_p2_evidence(evidence, seed+number)
        elif control == "graph_permutation" and "graph_adjacency" in evidence: evidence["graph_adjacency"] = graph_permutation(evidence["graph_adjacency"], evidence["channel_mask"], seed+number)
        main = build_nvr_features(evidence, patient_proto, seizure_proto)
        plus_target, minus_target, cf_valid = make_counterfactual_targets(main)
        plus = build_nvr_features(evidence, patient_proto, seizure_proto, target_override=plus_target)
        minus = build_nvr_features(evidence, patient_proto, seizure_proto, target_override=minus_target)
        drop_mask = evidence["channel_mask"].clone(); candidates = torch.where(drop_mask[0])[0]
        if candidates.numel() > 4:
            generator = torch.Generator().manual_seed(seed+number); order = candidates[torch.randperm(candidates.numel(), generator=generator)]
            required = max(1, math.ceil(candidates.numel() * .10)); removed = 0
            for candidate in order:
                trial = drop_mask.clone(); trial[0, candidate] = False; t = evidence["clinical_target_mask"] & trial
                if trial.sum() >= 4 and t.any() and (trial & ~t).any(): drop_mask = trial; removed += 1
                if removed >= required: break
        dropped = build_nvr_features(evidence, patient_proto, seizure_proto, channel_mask_override=drop_mask)
        records.append({"patient_key": source["patient_key"], "center": source["center"], "outcome_target": source["outcome_target"],
                        "main": _variant(main, names), "plus": _variant(plus, names), "minus": _variant(minus, names), "drop": _variant(dropped, names),
                        "counterfactual_valid": bool(cf_valid[0])})
        valid = main["channel_mask"][0]; old = source["seizure_nez_probability"][0][source["seizure_channel_mask"][0]]
        audits.append({"patient_key": source["patient_key"], "n_channels": int(valid.sum()), "target_count": int(main["target"][0].sum()),
                       "old_seizure_nez_probability_mean": float(old.mean()) if old.numel() else np.nan,
                       "old_seizure_nez_probability_std": float(old.std(unbiased=False)) if old.numel() else np.nan,
                       "q_consensus_nez_mean": float(main["q_consensus_nez"][0][valid].mean()), "q_proto_nez_mean": float(main["q_proto_nez"][0][valid].mean()),
                       "counterfactual_valid": bool(cf_valid[0]), **{name: float(main["scalars"][name][0]) for name in ALL_SCALAR_FEATURES}})
    return records, audits


def _write_outputs(output: Path, args: Any, predictions: pd.DataFrame, fold_metrics: list[dict[str, Any]], histories: list[pd.DataFrame], audits: list[dict[str, Any]], fold_details: list[dict[str, Any]], folds: pd.DataFrame, paper_valid: bool) -> None:
    predictions.to_csv(output/"oof_patient_predictions.csv", index=False); pd.DataFrame(fold_metrics).to_csv(output/"fold_metrics.csv", index=False)
    pd.concat(histories, ignore_index=True).to_csv(output/"training_curves.csv", index=False); pd.DataFrame(audits).to_csv(output/"nvr_patient_feature_audit.csv", index=False)
    pooled = compute_metrics(predictions.outcome_true, predictions.outcome_probability_success, prediction_success=predictions.outcome_pred_05)
    pd.DataFrame([{"scope": "pooled_oof", **pooled}]).to_csv(output/"summary_metrics.csv", index=False)
    center_rows=[{"center":center,**compute_metrics(group.outcome_true,group.outcome_probability_success,prediction_success=group.outcome_pred_05)} for center,group in predictions.groupby("center")]
    pd.DataFrame(center_rows).to_csv(output/"center_metrics.csv",index=False)
    bootstrap, skipped = bootstrap_metrics(predictions.outcome_true, predictions.outcome_probability_success, prediction_success=predictions.outcome_pred_05, repeats=args.bootstrap_repeats, seed=args.seed)
    ci=[]
    for name in ("accuracy","balanced_accuracy","macro_f1","weighted_f1","auroc","success_auprc","failure_auprc","success_precision","success_recall","failure_precision","failure_recall","brier","ece"):
        values=bootstrap[name].dropna().to_numpy() if name in bootstrap else np.asarray([])
        ci.append({"metric":name,"mean":float(values.mean()) if values.size else np.nan,"ci_low":float(np.quantile(values,.025)) if values.size else np.nan,"ci_high":float(np.quantile(values,.975)) if values.size else np.nan,"valid_repeats":int(values.size),"skipped_single_class":int(skipped)})
    pd.DataFrame(ci).to_csv(output/"bootstrap_ci.csv",index=False)
    audit_frame=pd.DataFrame(audits); audit_frame.to_csv(output/"patient_feature_table.csv",index=False)
    prototype_rows=[]
    for detail in fold_details:
        for kind in ("patient_prototype_audit","seizure_prototype_audit"):
            prototype_rows.append({"outer_fold":detail["outer_fold"],**detail[kind]})
    pd.DataFrame(prototype_rows).to_csv(output/"clean_nez_prototype_audit.csv",index=False)
    consensus_columns=[c for c in ("patient_key","outer_fold","partition","q_consensus_nez_mean","q_proto_nez_mean","view_disagreement_mean","reliability_weight_mean") if c in audit_frame]
    audit_frame[consensus_columns].to_csv(output/"nez_consensus_audit.csv",index=False)
    for filename,prefixes in (("outside_residual_audit.csv",("outside_","inside_")),("seizure_nez_stability_audit.csv",("outside_persistent","outside_worstcase","old_seizure")),("virtual_target_network_audit.csv",("residual_","persistent_residual_","hub_miss_","virtual_disruption_"))):
        columns=[c for c in audit_frame if c in ("patient_key","outer_fold","partition") or c.startswith(prefixes)]
        audit_frame[columns].to_csv(output/filename,index=False)
    monotone_cols=[c for c in predictions if c in ("patient_key","outer_fold","residual_risk_score","network_risk_score","diffuse_risk_score","support_score","structured_success_logit","set_delta")]
    predictions[monotone_cols].to_csv(output/"monotone_component_audit.csv",index=False)
    cf_cols=[c for c in ("patient_key","outer_fold","partition","counterfactual_valid") if c in audit_frame]
    audit_frame[cf_cols].to_csv(output/"counterfactual_audit.csv",index=False)
    predictions.assign(error=predictions.outcome_pred_05!=predictions.outcome_true).sort_values(["error","outcome_probability_success"],ascending=[False,True]).to_csv(output/"failure_case_analysis.csv",index=False)
    fold_audit = {"protocol": "fixed_outer_cv_no_inner", "mainline": "NVR-Outcome", "profile": args.profile, "n_outer_folds": int(predictions.outer_fold.nunique()),
                  "n_patients": len(predictions), "n_success": int(predictions.outcome_true.sum()), "n_failure": int((1-predictions.outcome_true).sum()),
                  "manifest_hash": manifest_hash(folds), "inner_cv_used": False, "threshold_tuning_used": False, "early_stopping_used": False,
                  "p2_frozen": True, "center_as_model_input": False, "negative_control": args.negative_control, "folds": fold_details, "paper_valid": paper_valid}
    (output/"fold_protocol_audit.json").write_text(json.dumps(fold_audit, indent=2), encoding="utf-8")
    (output/"config_audit.json").write_text(json.dumps({"profile": args.profile, "task2_success_label": 1, "task2_failure_label": 0, "p2_frozen": True,
        "p2_backbones_per_fold": 1, "prototype_fit_scope": "outer_train_only_patient_balanced_clean_clinical_NEZ", "clinical_target_semantics": "recorded surgical target, not ground-truth EZ",
        "seizure_nez_probability_usage": "audit_only", "center_usage": "loss_balancing_only", "coordinates_used": False, "SOZ_used": False, "resection_used": False,
        "true_EZ_claimed": False, "fixed_epochs": args.epochs, "optimizer": "AdamW", "lr": 5e-4, "weight_decay": 1e-3,
        "decision_threshold": .5, "threshold_source": FIXED_THRESHOLD_SOURCE, "negative_control": args.negative_control, "paper_valid": paper_valid}, indent=2), encoding="utf-8")
    status = "" if predictions.outer_fold.nunique() == 5 else "\nFULL_OUTER_CV_NOT_RUN\n"
    if args.protocol == "quick": status += "\nQUICK_SCREENING_NOT_PAPER_VALID\n"
    if not paper_valid: status += "\nOUTER_CV_NOT_PAPER_VALID\n"
    (output/"NVR_OUTCOME_REPORT.md").write_text(f"# NEZ-Verified Residual Network Outcome Report\n\nProfile: `{args.profile}`\n\nPatients: {len(predictions)}\n\nP2 is frozen. Clinical target is not claimed to be true EZ.\n{status}", encoding="utf-8")
    (output/"run_args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")


def main() -> int:
    args = parser().parse_args(); profile = get_nvr_profile(args.profile); args.epochs = args.epochs or (2 if args.protocol == "quick" else profile.outcome_epochs)
    output = Path(args.output_dir).expanduser(); output.mkdir(parents=True, exist_ok=True); seed_everything(args.seed)
    if args.protocol == "outer_cv" and args.outer_folds != 5: raise ValueError("Formal NVR outer_cv requires exactly five folds")
    if args.strict and args.protocol == "outer_cv" and not args.p2_training_manifest: raise ValueError("Strict NVR outer_cv requires --p2_training_manifest")
    feature = filtered_cache(load_cache(args.feature_cache), load_exclusion_manifest(args.exclusion_manifest))
    outcomes, _ = load_outcome_table(args.outcome_table, cache=feature); outcomes = outcomes[outcomes.outcome_group.isin(["success", "failure"])]
    outcomes = outcomes[outcomes.patient_key.isin(manifest_patient_keys(args.fold_manifest))].copy(); folds = load_fold_manifest(args.fold_manifest, outcomes, strict=args.strict)
    cohort = set(folds.patient_key); outcomes = outcomes[outcomes.patient_key.isin(cohort)]; labels = outcomes.set_index("patient_key").outcome_label.astype(int).to_dict()
    target_lookup, alignment, distribution = build_clinical_target_lookup(feature, sorted(cohort), strict=args.strict)
    alignment.to_csv(output/"clinical_target_alignment_audit.csv", index=False); distribution.to_csv(output/"clinical_target_distribution_audit.csv", index=False)
    graphs = load_graph_cache(args.graph_cache)
    if profile.virtual_network and not graphs: raise ValueError(f"{args.profile} requires --graph_cache")
    runtime = P23Runtime(args.p2_runtime_root); p2_args = load_p2_args(locate_p2_config(args.p2_checkpoint_root, args.p2_config), feature_cache=args.feature_cache)
    evidence_root = Path(args.evidence_cache_dir).expanduser() if args.evidence_cache_dir else output.parent.parent/"_frozen_p2_evidence"/f"seed_{args.seed}"
    prediction_frames=[]; metric_rows=[]; histories=[]; all_audits=[]; fold_details=[]
    fold_values = sorted(folds.outer_fold.unique())[:args.max_outer_folds or None]
    for fold in fold_values:
        test = set(folds.loc[folds.outer_fold == fold, "patient_key"]); train = cohort-test
        if args.max_patients:
            test = set(_balanced_take(sorted(test), labels, max(2, args.max_patients//4))); train = set(_balanced_take(sorted(train), labels, max(4, args.max_patients-len(test))))
        checkpoint = Path(args.p2_checkpoint_root)/f"fold_{int(fold)}"/"best_model.pt"
        if not checkpoint.exists(): raise FileNotFoundError(checkpoint)
        metadata_fold = _checkpoint_fold(checkpoint); metadata_match = metadata_fold in (None, int(fold))
        if args.strict and metadata_fold != int(fold): raise ValueError(f"P2 checkpoint fold mismatch: expected {fold}, found {metadata_fold}")
        leakage_verified = False
        if args.p2_training_manifest:
            assert_checkpoint_safe(checkpoint_training_subjects(args.p2_training_manifest, int(fold)), test); leakage_verified = True
        signature = hashlib.sha256(json.dumps({"fold": int(fold), "train": sorted(train), "test": sorted(test), "checkpoint": str(checkpoint.resolve()),
            "feature": str(Path(args.feature_cache).resolve()), "graph": str(Path(args.graph_cache).resolve()) if args.graph_cache else None}, sort_keys=True).encode()).hexdigest()
        evidence_path = evidence_root/f"fold_{int(fold)}.pt"; raw = None
        if args.resume and evidence_path.exists():
            cached = torch.load(evidence_path, map_location="cpu", weights_only=False)
            if cached.get("signature") == signature: raw = cached["records"]; print(f"[{args.profile}] fold {fold}: loaded frozen P2 evidence cache", flush=True)
        if raw is None:
            print(f"[{args.profile}] fold {fold}: exporting frozen P2 evidence", flush=True)
            examples, _ = prepare_examples(runtime, feature, p2_args, normalizer_subjects=sorted(train), output_subjects=sorted(train|test))
            loader = make_loader(examples, runtime, labels, graphs, batch_size=args.batch_size, shuffle=False, seed=args.seed, clinical_target_lookup=target_lookup)
            adapter = P2ExportAdapter(runtime, p2_args, checkpoint, device=args.device, limited_finetune=False)
            if any(p.requires_grad for p in adapter.parameters()): raise RuntimeError("NVR requires a completely frozen P2 adapter")
            raw = _extract_evidence(adapter, loader, args.device); evidence_path.parent.mkdir(parents=True, exist_ok=True); temporary=evidence_path.with_suffix(".tmp")
            torch.save({"signature": signature, "records": raw}, temporary); temporary.replace(evidence_path); del adapter
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        patient_proto, seizure_proto = _fit_prototypes(raw, train)
        transformed, audits = _make_records(raw, patient_proto, seizure_proto, args.profile, args.negative_control, args.seed+int(fold)*1000)
        train_records=[r for r in transformed if r["patient_key"] in train]; test_records=[r for r in transformed if r["patient_key"] in test]
        model=NVROutcomeModel(train_records[0]["main"]["patient_channel_embedding"].shape[-1], args.profile).to(args.device); fit_nvr_normalizer(model, train_records)
        train_loader=make_nvr_loader(train_records,batch_size=args.batch_size,shuffle=True,seed=args.seed+int(fold)); test_loader=make_nvr_loader(test_records,batch_size=args.batch_size,shuffle=False,seed=args.seed)
        fold_dir=output/f"fold_{int(fold)}"; fold_dir.mkdir(parents=True,exist_ok=True)
        history=train_nvr_fixed_epochs(model,train_loader,device=args.device,epochs=args.epochs,seed=args.seed+int(fold),robust=profile.robust_consistency,
            progress_prefix=f"{args.profile} fold {fold}",checkpoint_path=fold_dir/"training_progress.pt",resume=args.resume,
            resume_signature=hashlib.sha256((signature+args.profile+args.negative_control+"|".join(scalar_features_for_profile(args.profile))+"nvr_v1").encode()).hexdigest())
        torch.save({"model_state_dict":model.state_dict(),"patient_prototype":patient_proto.state_dict(),"seizure_prototype":seizure_proto.state_dict(),"outer_fold":int(fold),"profile":args.profile,"p2_frozen":True},fold_dir/"final_nvr.pt")
        prediction=predict_nvr(model,test_loader,args.device); prediction["outer_fold"]=int(fold); prediction["seed"]=args.seed; prediction["profile"]=args.profile; prediction["protocol"]=args.protocol; prediction["decision_threshold"]=.5; prediction["threshold_source"]=FIXED_THRESHOLD_SOURCE
        required_features=("outside_residual_top10_mean","outside_persistent_top10","outside_worstcase_top10","view_disagreement_mean","residual_edge_mass_spread","residual_spectral_radius_ratio_spread","hub_miss_ratio_spread","virtual_disruption_spread","global_abnormality_entropy_normalized")
        audit_lookup={row["patient_key"]:row for row in audits}
        for name in required_features: prediction[name]=prediction.patient_key.map(lambda patient:audit_lookup[patient][name])
        prediction_frames.append(prediction); metric_rows.append({"outer_fold":int(fold),**compute_metrics(prediction.outcome_true,prediction.outcome_probability_success,prediction_success=prediction.outcome_pred_05)}); histories.append(history.assign(outer_fold=int(fold)))
        for row in audits:
            if row["patient_key"] in train|test: row["outer_fold"]=int(fold); row["partition"]="train" if row["patient_key"] in train else "test"; all_audits.append(row)
        fold_details.append({"outer_fold":int(fold),"n_train":len(train),"n_test":len(test),"train_test_overlap":len(train&test),"p2_checkpoint":str(checkpoint.resolve()),"checkpoint_outer_fold":metadata_fold,"checkpoint_outer_fold_matches":metadata_match,"p2_checkpoint_training_leakage_verified":leakage_verified,"patient_prototype_audit":vars(patient_proto.audit()),"seizure_prototype_audit":vars(seizure_proto.audit())})
        print(f"[{args.profile}] fold {fold}: complete ({len(prediction)} OOF predictions)",flush=True)
    predictions=pd.concat(prediction_frames,ignore_index=True)
    if predictions.patient_key.duplicated().any(): raise ValueError("Each NVR patient must have one OOF prediction")
    completed=predictions.outer_fold.nunique()==5 and set(predictions.patient_key)==cohort
    paper_valid=bool(args.protocol=="outer_cv" and completed and args.strict and args.negative_control=="none" and all(x["p2_checkpoint_training_leakage_verified"] and x["checkpoint_outer_fold_matches"] for x in fold_details))
    _write_outputs(output,args,predictions,metric_rows,histories,all_audits,fold_details,folds,paper_valid)
    print(json.dumps({"profile":args.profile,"n_predictions":len(predictions),"outer_folds":list(map(int,fold_values)),"output_dir":str(output.resolve()),"paper_valid":paper_valid},indent=2)); return 0


if __name__ == "__main__": raise SystemExit(main())
