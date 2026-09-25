#!/usr/bin/env python3
"""Summarize PAT0/PAT1/PAT2 and patient-level paired bootstrap comparisons."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


METRICS = ["patient_macro_f1","patient_macro_ez_f1","patient_macro_nez_f1","patient_macro_balanced_accuracy","predicted_ez_count_mae"]


def _load(root: Path, profile: str, name: str) -> pd.DataFrame:
    path=root/profile/name
    if not path.is_file(): raise FileNotFoundError(f"Missing PAT output: {path}")
    return pd.read_csv(path)


def _bootstrap(base: pd.DataFrame, candidate: pd.DataFrame, repeats: int, seed: int) -> list[dict]:
    keys=["subject_id","center","outer_fold"]
    paired=base.merge(candidate,on=keys,suffixes=("_PAT0","_candidate"),validate="one_to_one")
    rng=np.random.default_rng(seed); rows=[]; n=len(paired)
    comparisons={metric:(f"{metric}_candidate",f"{metric}_PAT0") for metric in METRICS}
    comparisons["lzu_macro_f1"]=("patient_macro_f1_candidate","patient_macro_f1_PAT0")
    comparisons["non_lzu_macro_f1"]=("patient_macro_f1_candidate","patient_macro_f1_PAT0")
    for metric,(candidate_column,base_column) in comparisons.items():
        selected=paired if metric not in {"lzu_macro_f1","non_lzu_macro_f1"} else paired[paired.center.eq("lzu") if metric=="lzu_macro_f1" else ~paired.center.eq("lzu")]
        delta=selected[candidate_column].to_numpy(float)-selected[base_column].to_numpy(float)
        draws=np.asarray([delta[rng.integers(0,len(delta),len(delta))].mean() for _ in range(repeats)])
        rows.append({"metric":metric,"mean_delta":float(delta.mean()),"CI_2.5":float(np.quantile(draws,.025)),"CI_97.5":float(np.quantile(draws,.975)),"probability_delta_gt_zero":float((draws>0).mean()),"n_patients_improved":int((delta>0).sum()),"n_patients_worsened":int((delta<0).sum()),"n_patients_unchanged":int((delta==0).sum()),"n_delta_gt_0.05":int((delta>.05).sum()),"n_delta_lt_minus_0.05":int((delta<-.05).sum())})
    worst_draws=[]
    for _ in range(repeats):
        sample=paired.iloc[rng.integers(0,n,n)]
        candidate_worst=sample.groupby("center").patient_macro_f1_candidate.mean().min(); base_worst=sample.groupby("center").patient_macro_f1_PAT0.mean().min(); worst_draws.append(candidate_worst-base_worst)
    observed=paired.groupby("center")[["patient_macro_f1_candidate","patient_macro_f1_PAT0"]].mean(); delta=float(observed.patient_macro_f1_candidate.min()-observed.patient_macro_f1_PAT0.min()); draws=np.asarray(worst_draws)
    rows.append({"metric":"worst_center_f1","mean_delta":delta,"CI_2.5":float(np.quantile(draws,.025)),"CI_97.5":float(np.quantile(draws,.975)),"probability_delta_gt_zero":float((draws>0).mean()),"n_patients_improved":np.nan,"n_patients_worsened":np.nan,"n_patients_unchanged":np.nan,"n_delta_gt_0.05":np.nan,"n_delta_lt_minus_0.05":np.nan})
    return rows


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("--root_dir",required=True); parser.add_argument("--bootstrap_repeats",type=int,default=2000); parser.add_argument("--seed",type=int,default=42)
    args=parser.parse_args(); root=Path(args.root_dir); profiles=["PAT0_GLOBAL","PAT1_MINIMAL","PAT2_EXTENDED"]
    summaries=[]; folds=[]; centers=[]; patients={}
    for profile in profiles:
        summary=_load(root,profile,"p2_pat_formal_summary.csv").assign(profile=profile); summaries.append(summary)
        folds.append(_load(root,profile,"p2_pat_formal_by_fold.csv").assign(profile=profile))
        centers.append(_load(root,profile,"p2_pat_formal_by_center.csv").assign(profile=profile))
        patients[profile]=_load(root,profile,"p2_pat_formal_by_patient.csv")
    overall=pd.concat(summaries,ignore_index=True); by_fold=pd.concat(folds,ignore_index=True); by_center=pd.concat(centers,ignore_index=True)
    overall.to_csv(root/"p2_pat_ablation_summary.csv",index=False); by_fold.to_csv(root/"p2_pat_ablation_by_fold.csv",index=False); by_center.to_csv(root/"p2_pat_ablation_by_center.csv",index=False)
    bootstrap=[]
    for index,profile in enumerate(profiles[1:],start=1):
        bootstrap.extend({"comparison":f"{profile}_vs_PAT0",**row} for row in _bootstrap(patients["PAT0_GLOBAL"],patients[profile],args.bootstrap_repeats,args.seed+index))
    pd.DataFrame(bootstrap).to_csv(root/"p2_pat_paired_bootstrap.csv",index=False)
    pat0=float(overall.loc[overall.profile.eq("PAT0_GLOBAL"),"patient_macro_f1"].iloc[0]); pat1=float(overall.loc[overall.profile.eq("PAT1_MINIMAL"),"patient_macro_f1"].iloc[0]); pat2=float(overall.loc[overall.profile.eq("PAT2_EXTENDED"),"patient_macro_f1"].iloc[0])
    pat1_row = overall[overall.profile.eq("PAT1_MINIMAL")].iloc[0]
    pat2_row = overall[overall.profile.eq("PAT2_EXTENDED")].iloc[0]
    pat1_pass = pat1 >= .635 and pat1-pat0 >= .006
    pat2_pass = pat2 >= pat1 + .003 and pat2 >= pat0 and float(pat2_row.get("patient_macro_f1", pat2)) >= .0
    if pat2_pass:
        recommendation = "PAT2_EXTENDED"
    elif pat1_pass:
        recommendation = "PAT1_MINIMAL"
    else:
        recommendation = "PAT0_GLOBAL"
    decision={"PAT0_macro_f1":pat0,"PAT1_macro_f1":pat1,"PAT2_macro_f1":pat2,"PAT1_delta_vs_PAT0":pat1-pat0,"PAT2_delta_vs_PAT1":pat2-pat1,"PAT2_delta_vs_PAT0":pat2-pat0,"PAT1_admitted":pat1_pass,"PAT2_admitted":pat2_pass,"recommended_profile":recommendation,"target_0.636_met":max(pat1,pat2)>=.636,"strong_target_0.640_met":max(pat1,pat2)>=.640,"status":"passed" if recommendation != "PAT0_GLOBAL" else "no_pat_improvement"}
    (root/"P2_Q10_PAT_ABLATION_REPORT.md").write_text("# P2-Q10-PAT Ablation Report\n\nPAT changes thresholds only; all channel scores and ranking metrics remain frozen.\n\n```json\n"+json.dumps(decision,indent=2)+"\n```\n",encoding="utf-8")
    print(json.dumps(decision,indent=2))


if __name__=="__main__": main()
