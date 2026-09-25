from __future__ import annotations

import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT=Path(__file__).resolve().parents[2]
if str(PROJECT) not in sys.path: sys.path.insert(0,str(PROJECT))
from neuroez_c.task2.metrics import compute_metrics

C_PROFILES=("C0_TARGET_ONLY","C1_P2_ONLY","C2_P2_TARGET_CONCORDANCE","C3_CROSS_SEIZURE_CONCORDANCE","C4_TARGET_NETWORK","C5_FULL")
SEEDS=(42,43,44)


def _metrics(frame:pd.DataFrame)->dict:
    return compute_metrics(frame.outcome_true,frame.outcome_probability_success,prediction_success=frame.outcome_pred_05)


def _fixed_ensemble(frames:dict[int,pd.DataFrame],output:Path)->pd.DataFrame|None:
    if set(frames)!=set(SEEDS): return None
    base=frames[42].copy(); probabilities=[]
    for seed in SEEDS:
        local=frames[seed].sort_values("patient_key").reset_index(drop=True)
        if local.patient_key.tolist()!=base.sort_values("patient_key").patient_key.tolist(): raise ValueError("C5 seed cohorts differ")
        probabilities.append(local.outcome_probability_success.clip(1e-7,1-1e-7).map(lambda p:np.log(p/(1-p))).to_numpy())
    base=base.sort_values("patient_key").reset_index(drop=True); logits=np.mean(probabilities,axis=0); base["outcome_probability_success"]=1/(1+np.exp(-logits)); base["outcome_probability_failure"]=1-base.outcome_probability_success; base["outcome_pred_05"]=(base.outcome_probability_success>=.5).astype(int); base["profile"]="C6_FIXED_ENSEMBLE"; base["seed"]="42|43|44"
    output.mkdir(parents=True,exist_ok=True); base.to_csv(output/"oof_patient_predictions.csv",index=False); pd.DataFrame([{"scope":"pooled_oof",**_metrics(base)}]).to_csv(output/"summary_metrics.csv",index=False); return base


def main()->int:
    parser=argparse.ArgumentParser(description="Summarize fixed C0-C6 clinical-target outcome ablations"); parser.add_argument("--root_dir",required=True); parser.add_argument("--bootstrap_repeats",type=int,default=2000); parser.add_argument("--seed",type=int,default=42); args=parser.parse_args()
    root=Path(args.root_dir); output=root/"ablation"; output.mkdir(parents=True,exist_ok=True)
    predictions={}; rows=[]; folds=[]; centers=[]
    for profile in C_PROFILES:
        for seed in SEEDS:
            path=root/profile/f"seed_{seed}"/"oof_patient_predictions.csv"
            if not path.exists(): continue
            frame=pd.read_csv(path); predictions[(profile,seed)]=frame; metric=_metrics(frame); rows.append({"profile":profile,"seed":seed,"n_patients":len(frame),**metric})
            for fold,group in frame.groupby("outer_fold"): folds.append({"profile":profile,"seed":seed,"outer_fold":fold,**_metrics(group)})
            for center,group in frame.groupby("center"): centers.append({"profile":profile,"seed":seed,"center":center,**_metrics(group)})
    c5={seed:predictions[("C5_FULL",seed)] for seed in SEEDS if ("C5_FULL",seed) in predictions}; ensemble=_fixed_ensemble(c5,root/"C6_FIXED_ENSEMBLE")
    if ensemble is not None: rows.append({"profile":"C6_FIXED_ENSEMBLE","seed":"fixed_42_43_44","n_patients":len(ensemble),**_metrics(ensemble)})
    frame=pd.DataFrame(rows); frame.to_csv(output/"profile_seed_metrics.csv",index=False); pd.DataFrame(folds).to_csv(output/"fold_metrics_all_profiles_seeds.csv",index=False); pd.DataFrame(centers).to_csv(output/"center_metrics_all_profiles_seeds.csv",index=False)
    numeric=[c for c in frame.columns if c not in {"profile","seed"}]
    seed_summary=frame[frame.seed.astype(str).isin(map(str,SEEDS))].groupby("profile")[numeric].agg(["mean","std"]); seed_summary.columns=[f"{a}_{b}" for a,b in seed_summary.columns]; seed_summary.reset_index().to_csv(output/"three_seed_mean_std.csv",index=False)
    deltas=[]
    for seed in SEEDS:
        for left,right,label in (("C1_P2_ONLY","C2_P2_TARGET_CONCORDANCE","C2_minus_C1"),("C2_P2_TARGET_CONCORDANCE","C3_CROSS_SEIZURE_CONCORDANCE","C3_minus_C2"),("C3_CROSS_SEIZURE_CONCORDANCE","C4_TARGET_NETWORK","C4_minus_C3"),("C4_TARGET_NETWORK","C5_FULL","C5_minus_C4")):
            if (left,seed) in predictions and (right,seed) in predictions:
                lm,rm=_metrics(predictions[(left,seed)]),_metrics(predictions[(right,seed)]); deltas.append({"seed":seed,"comparison":label,"accuracy_delta":rm["accuracy"]-lm["accuracy"],"auroc_delta":rm["auroc"]-lm["auroc"]})
    delta_frame=pd.DataFrame(deltas); delta_frame.to_csv(output/"predefined_ablation_deltas.csv",index=False)
    flags=[]
    c0=frame[(frame.profile=="C0_TARGET_ONLY") & (frame.seed==42)]
    if not c0.empty and float(c0.iloc[0].accuracy)>=.75: flags.append("POTENTIAL_TARGET_LABEL_LEAKAGE")
    missing=[f"{p}/seed_{s}" for p in C_PROFILES for s in SEEDS if (p,s) not in predictions]
    report=["# P2 Clinical-Target Outcome Report","","Primary profile: `C5_FULL`","Fixed ensemble: `C6_FIXED_ENSEMBLE` (mean logits of seeds 42, 43, 44)",""]
    if missing: report.extend(["FULL_OUTER_CV_NOT_RUN",f"Missing/unrun: {', '.join(missing)}",""])
    report.extend(flags or ["No diagnostic flag can be declared until the required runs exist."]); (output/"P2_TARGET_OUTCOME_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(json.dumps({"runs_found":len(predictions),"ensemble_created":ensemble is not None,"missing":missing,"flags":flags,"output":str(output.resolve())},indent=2)); return 0


if __name__=="__main__": raise SystemExit(main())
