#!/usr/bin/env python3
"""Run P2-Q10-PAT without retraining or changing frozen channel scores."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from neuroez_c.p2_pat_decoder import logit_to_probability
from neuroez_c.p2_pat_oracle_target import compute_stable_patient_oracle_threshold_target
from neuroez_c.p2_pat_protocol import (
    PAT_PROFILES, build_patient_features, canonicalize_pat_channel_frame, evaluate_pat_ledger,
    feature_columns, run_lopo_selection, score_invariance_audit,
)


def _apply_thresholds(frame: pd.DataFrame, *, global_probability: float, patient_thresholds: dict[str, tuple[float,float]], profile: str, config: dict) -> pd.DataFrame:
    result = canonicalize_pat_channel_frame(frame, profile=profile).copy()
    global_logit = float(np.log(global_probability/(1-global_probability)))
    result["global_threshold_probability"] = global_probability
    result["global_threshold_logit"] = global_logit
    result["predicted_threshold_residual"] = result.subject_id.map(lambda subject: patient_thresholds[str(subject)][1]).astype(float)
    result["patient_threshold_logit"] = result.subject_id.map(lambda subject: patient_thresholds[str(subject)][0]).astype(float)
    result["patient_threshold_probability"] = result.patient_threshold_logit.map(logit_to_probability)
    result["score_nez"] = 1/(1+np.exp(-result.base_nez_logit.to_numpy(float)))
    result["score_ez"] = 1-result.score_nez
    result["base_global_pred_nez"] = (result.base_nez_logit >= global_logit).astype(int)
    result["pat_pred_nez"] = (result.base_nez_logit >= result.patient_threshold_logit).astype(int)
    result["base_global_pred_ez"] = 1-result.base_global_pred_nez
    result["pat_pred_ez"] = 1-result.pat_pred_nez
    result["selected_threshold"] = result.patient_threshold_probability
    result["p2_pat_profile"] = profile
    result["ridge_alpha"] = config.get("ridge_alpha", np.nan)
    result["shrinkage"] = config.get("shrinkage", 0.0)
    result["max_prediction_residual"] = config.get("max_prediction_residual", 0.0)
    result["decision_rule"] = "fold_validation_global_threshold" if profile == "PAT0_GLOBAL" else "validation_trained_patient_adaptive_threshold"
    result["threshold_source"] = "frozen_p2_q10_outer_validation" if profile == "PAT0_GLOBAL" else "outer_validation_patient_oracle_targets_with_lopo_selection"
    result["true_count_used_for_prediction"] = False
    result["center_used_as_model_input"] = False
    result["analysis_status"] = "PRIMARY_LEGAL"
    return result


def _threshold_diagnostics(patients: pd.DataFrame) -> tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    rows = patients[["subject_id","center","outer_fold","global_threshold_probability","patient_threshold_probability","predicted_threshold_residual","residual_clip_rate"]].copy()
    rows["abs_threshold_residual"] = rows.predicted_threshold_residual.abs()
    def aggregate(source: pd.DataFrame, key: str) -> pd.DataFrame:
        return source.groupby(key, as_index=False).agg(n_patients=("subject_id","nunique"), mean_global_threshold_probability=("global_threshold_probability","mean"), mean_patient_threshold_probability=("patient_threshold_probability","mean"), mean_threshold_residual=("predicted_threshold_residual","mean"), mean_abs_threshold_residual=("abs_threshold_residual","mean"), p90_abs_threshold_residual=("abs_threshold_residual",lambda x: np.quantile(x,.90)), p95_abs_threshold_residual=("abs_threshold_residual",lambda x: np.quantile(x,.95)), max_abs_threshold_residual=("abs_threshold_residual","max"), positive_residual_fraction=("predicted_threshold_residual",lambda x: (x>0).mean()), negative_residual_fraction=("predicted_threshold_residual",lambda x: (x<0).mean()), residual_clip_rate=("residual_clip_rate","mean"))
    return rows, aggregate(rows,"outer_fold"), aggregate(rows,"center")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2_q10_root", required=True)
    parser.add_argument("--base_manifest", required=True)
    parser.add_argument("--p2_pat_profile", choices=PAT_PROFILES, required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--p2_pat_oracle_epsilon_f1", type=float, default=0.002)
    parser.add_argument("--p2_pat_max_target_residual", type=float, default=1.0)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--max_outer_folds", type=int, default=0)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    random.seed(args.random_seed); np.random.seed(args.random_seed)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(Path(args.base_manifest).read_text(encoding="utf-8"))
    if manifest.get("profile") != "P2_TEMPORAL_Q10" or len(manifest.get("per_fold", [])) != 5:
        raise RuntimeError("PAT requires an audited five-fold P2_TEMPORAL_Q10 base manifest")
    run_args = vars(args).copy(); run_args.update({"p2_q10_checkpoint_retrained": False, "channel_scores_modified": False, "center_as_input": False, "true_k_as_input": False})
    (output/"run_args.json").write_text(json.dumps(run_args, indent=2), encoding="utf-8")
    (output/"p2_pat_base_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    config_audit = {"status":"passed", "profile":args.p2_pat_profile, "decoder":"StandardScaler+Ridge" if args.p2_pat_profile != "PAT0_GLOBAL" else "global_threshold", "feature_columns":feature_columns(args.p2_pat_profile), "center_used_as_model_input":False, "true_k_used_as_model_input":False, "outer_test_labels_used_for_decoder_fit":False, "p2_q10_retrained":False}
    (output/"p2_pat_config_audit.json").write_text(json.dumps(config_audit, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps({**config_audit,"mode":"dry_run"}, indent=2)); return

    all_ledgers=[]; validation_targets=[]; validation_features=[]; test_features=[]; lopo_rows=[]; selected_by_fold={}
    fold_limit = args.max_outer_folds or 5
    for fold in range(1, fold_limit+1):
        meta = next(row for row in manifest["per_fold"] if int(row["outer_fold"]) == fold)
        if meta.get("status") != "passed": raise RuntimeError(f"base fold {fold} failed audit")
        validation = canonicalize_pat_channel_frame(pd.read_csv(meta["validation_path"]), profile=args.p2_pat_profile)
        test = canonicalize_pat_channel_frame(pd.read_csv(meta["test_path"]), profile=args.p2_pat_profile)
        global_probability = float(meta["global_threshold_probability"])
        global_logit = float(meta["global_threshold_logit"])
        for frame in (validation,test): frame["outer_fold"] = fold
        if args.p2_pat_profile == "PAT0_GLOBAL":
            selected = {"profile":"PAT0_GLOBAL", "ridge_alpha":None, "shrinkage":0.0, "max_prediction_residual":0.0}
            thresholds = {str(subject):(global_logit,0.0) for subject in test.subject_id.unique()}
        else:
            val_features = build_patient_features(validation, global_logit, args.p2_pat_profile)
            val_features["split_role"] = "validation"; validation_features.append(val_features)
            targets=[]
            for subject, group in validation.groupby("subject_id", sort=True):
                targets.append(compute_stable_patient_oracle_threshold_target(group.base_nez_logit.to_numpy(), group.label_nez.to_numpy(), global_logit, subject_id=subject, outer_fold=fold, split_role="validation", epsilon_f1=args.p2_pat_oracle_epsilon_f1, max_target_residual=args.p2_pat_max_target_residual))
            target_frame=pd.DataFrame(targets); validation_targets.append(target_frame)
            selected, fold_lopo, decoder = run_lopo_selection(validation, val_features, target_frame, profile=args.p2_pat_profile)
            lopo_rows.append(fold_lopo)
            current_test_features=build_patient_features(test, global_logit, args.p2_pat_profile); current_test_features["split_role"]="outer_test"; test_features.append(current_test_features)
            columns=feature_columns(args.p2_pat_profile)
            threshold_values,residuals=decoder.predict_threshold_logit(current_test_features[columns].to_numpy(), np.full(len(current_test_features),global_logit))
            thresholds={str(subject):(float(threshold),float(residual)) for subject,threshold,residual in zip(current_test_features.subject_id,threshold_values,residuals)}
        selected_by_fold[str(fold)]={**selected,"global_threshold_probability":global_probability,"global_threshold_logit":global_logit,"n_validation_patients":int(validation.subject_id.nunique()),"n_test_patients":int(test.subject_id.nunique())}
        all_ledgers.append(_apply_thresholds(test,global_probability=global_probability,patient_thresholds=thresholds,profile=args.p2_pat_profile,config=selected))

    ledger=pd.concat(all_ledgers,ignore_index=True)
    ledger.to_csv(output/"p2_pat_oof_channel_ledger.csv",index=False)
    pd.concat(validation_targets,ignore_index=True).to_csv(output/"p2_pat_validation_oracle_targets.csv",index=False) if validation_targets else pd.DataFrame().to_csv(output/"p2_pat_validation_oracle_targets.csv",index=False)
    pd.concat(validation_features,ignore_index=True).to_csv(output/"p2_pat_patient_features_validation.csv",index=False) if validation_features else pd.DataFrame().to_csv(output/"p2_pat_patient_features_validation.csv",index=False)
    pd.concat(test_features,ignore_index=True).to_csv(output/"p2_pat_patient_features_test.csv",index=False) if test_features else pd.DataFrame().to_csv(output/"p2_pat_patient_features_test.csv",index=False)
    pd.concat(lopo_rows,ignore_index=True).to_csv(output/"p2_pat_lopo_selection.csv",index=False) if lopo_rows else pd.DataFrame().to_csv(output/"p2_pat_lopo_selection.csv",index=False)
    (output/"p2_pat_selected_decoder_by_fold.json").write_text(json.dumps(selected_by_fold,indent=2),encoding="utf-8")
    patients,summary,by_fold,by_center=evaluate_pat_ledger(ledger)
    if args.p2_pat_profile == "PAT0_GLOBAL" and fold_limit == 5:
        expected = manifest.get("formal_metrics", [{}])[0]
        tolerances = {"patient_macro_f1": 5e-4, "patient_macro_auprc_ez": 1e-8, "patient_macro_ez_mrr": 1e-8}
        for metric, tolerance in tolerances.items():
            if metric not in expected or abs(float(summary[metric].iloc[0])-float(expected[metric])) > tolerance:
                raise RuntimeError(f"PAT0 failed to reproduce frozen P2-Q10 {metric}")
    patients.to_csv(output/"p2_pat_formal_by_patient.csv",index=False); summary.to_csv(output/"p2_pat_formal_summary.csv",index=False); by_fold.to_csv(output/"p2_pat_formal_by_fold.csv",index=False); by_center.to_csv(output/"p2_pat_formal_by_center.csv",index=False)
    diag_patient,diag_fold,diag_center=_threshold_diagnostics(patients)
    diag_patient.to_csv(output/"p2_pat_threshold_diagnostics_by_patient.csv",index=False); diag_fold.to_csv(output/"p2_pat_threshold_diagnostics_by_fold.csv",index=False); diag_center.to_csv(output/"p2_pat_threshold_diagnostics_by_center.csv",index=False)
    base=pd.concat([canonicalize_pat_channel_frame(pd.read_csv(next(row for row in manifest["per_fold"] if int(row["outer_fold"])==fold)["test_path"])) for fold in range(1,fold_limit+1)],ignore_index=True)
    invariance=score_invariance_audit(base,ledger); (output/"p2_pat_score_invariance_audit.json").write_text(json.dumps(invariance,indent=2),encoding="utf-8")
    if not invariance["passed"]: raise RuntimeError("PAT changed frozen scores or ranking metrics")
    result={"profile":args.p2_pat_profile,"formal_metrics":summary.iloc[0].to_dict(),"score_invariance":invariance,"selected_decoder_by_fold":selected_by_fold,"p2_q10_retrained":False}
    (output/"P2_Q10_PAT_REPORT.md").write_text("# P2-Q10-PAT Report\n\nFormal predictions use a validation-trained patient threshold. Outer-test labels are connected only after prediction.\n\n```json\n"+json.dumps(result,indent=2)+"\n```\n",encoding="utf-8")
    print(json.dumps({"status":"passed","profile":args.p2_pat_profile,"n_patients":int(ledger.subject_id.nunique()),"patient_macro_f1":float(summary.patient_macro_f1.iloc[0]),"score_invariance":True},indent=2))


if __name__ == "__main__":
    main()
