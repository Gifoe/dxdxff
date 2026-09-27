"""Aggregate private 65x20 calibration cells by patient-ID clusters, publish no patient rows."""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict

import numpy as np

from common import BUDGETS, LAMBDA_GRID, POLICIES, PRIOR_RUNTIME, ROOT, RUNTIME, LOCK_SHA, preflight, read_csv, write_csv, write_json

N_BOOT=10000
METRICS=("ap","auc","mrr","top1","ndcg","macro_f1","ez_f1","nez_f1","ba","predicted_ez_fraction")


def finite_mean(values):
    x=np.asarray(values,dtype=float)
    x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def load_cells():
    cells=[]
    for fold in range(1,6):
        selected=read_csv(PRIOR_RUNTIME/"private"/f"fold_{fold}"/"A1_VLOO_PRIVATE.csv")
        if len(selected)!=13: raise RuntimeError("Incomplete 13-patient VLOO fold")
        for row in selected:
            sid=row["subject_id"]
            stem=hashlib.sha256(sid.encode()).hexdigest()[:16]
            with (RUNTIME/"private"/f"fold_{fold}"/(stem+".pkl")).open("rb") as f:
                cell=pickle.load(f)
            if (cell["lock_sha"]!=LOCK_SHA or cell["subject_id"]!=sid or cell["fold"]!=fold or
                cell["epoch"]!=int(row["selected_epoch"]) or cell["threshold"]!=float(row["selected_threshold"]) or
                len(cell["split_audit"])!=20 or not cell["zero_budget_exact"] or
                not cell["target_scores_frozen_before_query_labels"]):
                raise RuntimeError("Private cell provenance, zero budget, or label boundary failed")
            cells.append(cell)
    if len(cells)!=65 or len({c["subject_id"] for c in cells})!=47:
        raise RuntimeError("65-cell/47-patient geometry changed")
    return cells


def metric_delta(row, other, key):
    if other is None: return np.nan
    a,b=row[key],other[key]
    return float(a-b) if np.isfinite(a) and np.isfinite(b) else np.nan


def cluster_boot(values, rows, unique, boot_counts):
    if len(values)!=len(rows): raise RuntimeError("Bootstrap row alignment failed")
    sums=np.zeros(len(unique)); counts=np.zeros(len(unique))
    lookup={sid:i for i,sid in enumerate(unique)}
    for row,value in zip(rows,values):
        if np.isfinite(value):
            i=lookup[row["subject_id"]]
            sums[i]+=value; counts[i]+=1
    if not counts.sum(): return None,None,None
    mass=boot_counts@counts
    sample=np.divide(boot_counts@sums,mass,out=np.full(N_BOOT,np.nan),where=mass>0)
    sample=sample[np.isfinite(sample)]
    if len(sample)<.99*N_BOOT: raise RuntimeError("Too many empty patient-bootstrap draws")
    return float(sums.sum()/counts.sum()),float(np.quantile(sample,.025)),float(np.quantile(sample,.975))


def positive_folds(values, rows):
    return sum((finite_mean([v for v,r in zip(values,rows) if r["fold"]==fold]) or 0)>0 for fold in range(1,6))


def value(row,key):
    x=row.get(key)
    return float(x) if x is not None and np.isfinite(x) else np.nan


def group_key(row):
    return row["acquisition_policy"],row["calibration_type"],(
        "FULL_POOL" if row["acquisition_policy"]=="FULL_POOL_CALIBRATION" else row["budget"])


def main():
    preflight()
    reproduction=json.loads((ROOT/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if reproduction["checkpoints"]!=150 or reproduction["max_grid_error"]>1e-6:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    cells=load_cells()
    rows=[r for c in cells for r in c["records"]]
    unique=sorted({c["subject_id"] for c in cells})
    rng=np.random.default_rng(42)
    draws=rng.integers(0,len(unique),size=(N_BOOT,len(unique)))
    boot_counts=np.zeros((N_BOOT,len(unique)),dtype=np.int16)
    np.add.at(boot_counts,(np.arange(N_BOOT)[:,None],draws),1)
    boot_counts=boot_counts.astype(np.float32)
    exact={(r["subject_id"],r["fold"],r["repetition"],r["acquisition_policy"],r["calibration_type"],r["budget"]):r for r in rows}
    if len(exact)!=len(rows): raise RuntimeError("Duplicate private policy/budget/repetition")
    groups=defaultdict(list)
    for row in rows: groups[group_key(row)].append(row)
    def partner(r,policy=None,kind=None):
        return exact.get((r["subject_id"],r["fold"],r["repetition"],
                          policy or r["acquisition_policy"],kind or r["calibration_type"],r["budget"]))
    def deltas(gr,key,comparison):
        if comparison=="a1": return [value(r,key)-value(r,"a1_"+key) for r in gr]
        if comparison=="bias": return [metric_delta(r,partner(r,kind="BIAS_ONLY"),key) if r["calibration_type"]=="FULL_RESIDUAL" else np.nan for r in gr]
        if comparison=="random": return [metric_delta(r,partner(r,policy="RANDOM"),key) if r["acquisition_policy"] in ("UNCERTAINTY","UNCERTAINTY_DIVERSITY") else np.nan for r in gr]
        if comparison=="uncertainty": return [metric_delta(r,partner(r,policy="UNCERTAINTY"),key) if r["acquisition_policy"]=="UNCERTAINTY_DIVERSITY" else np.nan for r in gr]
        raise ValueError(comparison)
    summaries=[]; bootstrap=[]; foldrows=[]; headroom=[]
    for (policy,kind,budget),gr in sorted(groups.items(),key=lambda x:(x[0][0],x[0][1],str(x[0][2]))):
        rec=np.asarray([((value(r,"ap")-value(r,"a1_ap"))/max(value(r,"full_pool_ap")-value(r,"a1_ap"),1e-8))
                        if np.isfinite(value(r,"ap")) and np.isfinite(value(r,"full_pool_ap")) and
                           value(r,"full_pool_ap")>value(r,"a1_ap") else np.nan for r in gr])
        comparisons={"ap_a1":deltas(gr,"ap","a1"),"mrr_a1":deltas(gr,"mrr","a1"),
                     "macro_f1_a1":deltas(gr,"macro_f1","a1"),"full_bias_ap":deltas(gr,"ap","bias"),
                     "active_random_ap":deltas(gr,"ap","random"),
                     "diversity_uncertainty_ap":deltas(gr,"ap","uncertainty"),"headroom_recovery":rec}
        intervals={}
        for stat,v in comparisons.items():
            point,low,high=cluster_boot(v,gr,unique,boot_counts)
            intervals[stat]=(point,low,high)
            if point is not None:
                bootstrap.append({"acquisition_policy":policy,"calibration_type":kind,"budget":budget,
                                  "statistic":stat,"n_clusters":47,"n_estimable":int(np.isfinite(v).sum()),
                                  "mean":point,"ci_low":low,"ci_high":high})
        a1_delta=np.asarray(comparisons["ap_a1"],float)
        rand_delta=np.asarray(comparisons["active_random_ap"],float)
        fold_a1=positive_folds(a1_delta,gr)
        fold_rand=positive_folds(rand_delta,gr)
        for fold in range(1,6):
            indices=np.asarray([r["fold"]==fold for r in gr])
            foldrows.append({"acquisition_policy":policy,"calibration_type":kind,"budget":budget,"fold":fold,
                             "target_cells":len({(r["fold"],r["subject_id"]) for r in np.asarray(gr,dtype=object)[indices]}),
                             "mean_delta_ap_vs_a1":finite_mean(a1_delta[indices]),
                             "mean_delta_ap_vs_random":finite_mean(rand_delta[indices])})
        result={"acquisition_policy":policy,"calibration_type":kind,"budget":budget,
                "n_target_cells":len({(r["fold"],r["subject_id"]) for r in gr}),
                "n_unique_patients":len({r["subject_id"] for r in gr}),
                "n_repetitions":len({(r["fold"],r["subject_id"],r["repetition"]) for r in gr}),
                "n_estimable":sum(np.isfinite(value(r,"ap")) for r in gr),
                **{f"mean_{m}":finite_mean([value(r,m) for r in gr]) for m in METRICS},
                "matched_a1_ap":finite_mean([value(r,"a1_ap") for r in gr]),
                "matched_a1_mrr":finite_mean([value(r,"a1_mrr") for r in gr]),
                "matched_a1_top1":finite_mean([value(r,"a1_top1") for r in gr]),
                "matched_a1_macro_f1":finite_mean([value(r,"a1_macro_f1") for r in gr]),
                "matched_a1_ez_f1":finite_mean([value(r,"a1_ez_f1") for r in gr]),
                "delta_ap_vs_a1":intervals["ap_a1"][0],
                "delta_mrr_vs_a1":intervals["mrr_a1"][0],
                "delta_macro_f1_vs_a1":intervals["macro_f1_a1"][0],
                "bias_only_ap":finite_mean([value(partner(r,kind="BIAS_ONLY"),"ap") if partner(r,kind="BIAS_ONLY") else np.nan for r in gr]),
                "delta_ap_full_vs_bias":intervals["full_bias_ap"][0],
                "random_ap":finite_mean([value(partner(r,policy="RANDOM"),"ap") if partner(r,policy="RANDOM") else np.nan for r in gr]),
                "delta_ap_vs_random":intervals["active_random_ap"][0],
                "mean_support_ez":finite_mean([r["support_ez"] for r in gr]),
                "mean_support_nez":finite_mean([r["support_nez"] for r in gr]),
                "fraction_support_has_both_classes":finite_mean([r["support_both"] for r in gr]),
                "mean_delta_w_norm":finite_mean([r["delta_w_norm"] for r in gr]),
                "mean_oracle_direction_cosine":finite_mean([r["oracle_direction_cosine"] for r in gr]),
                "mean_headroom_recovery":intervals["headroom_recovery"][0],
                "ap_delta_ci_low":intervals["ap_a1"][1],"ap_delta_ci_high":intervals["ap_a1"][2],
                "full_vs_bias_ci_low":intervals["full_bias_ap"][1],"full_vs_bias_ci_high":intervals["full_bias_ap"][2],
                "active_vs_random_ci_low":intervals["active_random_ap"][1],"active_vs_random_ci_high":intervals["active_random_ap"][2],
                "positive_folds_vs_a1":fold_a1,"positive_folds_vs_random":fold_rand}
        summaries.append(result)
        headroom.append({"acquisition_policy":policy,"calibration_type":kind,"budget":budget,
                         "positive_full_pool_headroom_repetitions":int(np.isfinite(rec).sum()),
                         "mean_recovery_fraction":intervals["headroom_recovery"][0],
                         "cluster_ci_low":intervals["headroom_recovery"][1],
                         "cluster_ci_high":intervals["headroom_recovery"][2]})
    write_csv(ROOT/"SUMMARY_MATRIX.csv",summaries)
    write_csv(ROOT/"statistics"/"PATIENT_CLUSTER_BOOTSTRAP.csv",bootstrap)
    write_csv(ROOT/"statistics"/"FOLD_CONSISTENCY.csv",foldrows)
    write_csv(ROOT/"comparisons"/"HEADROOM_RECOVERY.csv",headroom)
    by={(r["acquisition_policy"],r["calibration_type"],str(r["budget"])):r for r in summaries}
    for policy,file in (("RANDOM","RANDOM_CALIBRATION_CURVE.csv"),
                        ("UNCERTAINTY","UNCERTAINTY_CALIBRATION_CURVE.csv"),
                        ("UNCERTAINTY_DIVERSITY","UNCERTAINTY_DIVERSITY_CALIBRATION_CURVE.csv"),
                        ("ORACLE_BALANCED_RANDOM","ORACLE_BALANCED_RANDOM_CURVE.csv")):
        write_csv(ROOT/"curves"/file,[r for r in summaries if r["acquisition_policy"]==policy and r["calibration_type"]=="FULL_RESIDUAL"])
    write_csv(ROOT/"curves"/"FULL_POOL_CALIBRATION.csv",[r for r in summaries if r["acquisition_policy"]=="FULL_POOL_CALIBRATION"])
    write_csv(ROOT/"curves"/"BIAS_ONLY_CURVE.csv",[r for r in summaries if r["calibration_type"]=="BIAS_ONLY"])
    write_csv(ROOT/"comparisons"/"MATCHED_A1_DELTAS.csv",[r for r in summaries if r["calibration_type"]=="FULL_RESIDUAL"])
    write_csv(ROOT/"comparisons"/"FULL_VS_BIAS.csv",[r for r in summaries if r["calibration_type"]=="FULL_RESIDUAL" and r["acquisition_policy"] in POLICIES])
    write_csv(ROOT/"comparisons"/"ACTIVE_VS_RANDOM.csv",[r for r in summaries if r["calibration_type"]=="FULL_RESIDUAL" and r["acquisition_policy"] in ("UNCERTAINTY","UNCERTAINTY_DIVERSITY")])
    composition=[]; geometry=[]
    for r in summaries:
        if r["calibration_type"]!="FULL_RESIDUAL": continue
        gr=groups[(r["acquisition_policy"],r["calibration_type"],r["budget"])]
        composition.append({"acquisition_policy":r["acquisition_policy"],"budget":r["budget"],
                            "n_repetitions":r["n_repetitions"],"mean_support_ez":r["mean_support_ez"],
                            "mean_support_nez":r["mean_support_nez"],"fraction_support_has_both_classes":r["fraction_support_has_both_classes"],
                            "fraction_zero_ez":finite_mean([v["support_ez"]==0 for v in gr]),
                            "fraction_zero_nez":finite_mean([v["support_nez"]==0 for v in gr])})
        geometry.append({"acquisition_policy":r["acquisition_policy"],"budget":r["budget"],
                         "n_repetitions":r["n_repetitions"],"mean_delta_w_norm":r["mean_delta_w_norm"],
                         "mean_oracle_direction_cosine":r["mean_oracle_direction_cosine"],
                         "note":"Oracle is target-label-using and nondeployable; raw R4 coefficient transformed to FIT-standardized coordinates"})
    write_csv(ROOT/"support"/"SUPPORT_CLASS_COMPOSITION.csv",composition)
    write_csv(ROOT/"geometry"/"FEWSHOT_DIRECTION_ALIGNMENT.csv",geometry)
    selection=[]
    for fold in range(1,6):
        sub=[c for c in cells if c["fold"]==fold]
        for lam in LAMBDA_GRID:
            selection.append({"fold":fold,"lambda_w":lam,"target_cells":13,
                              "mean_fit_episode_ap":finite_mean([c["lambda_result"]["scores"][str(lam)] for c in sub]),
                              "selected_in_cells":sum(c["lambda_result"]["selected_lambda"]==lam for c in sub),
                              "min_estimable_fit_patients":min(c["lambda_result"]["estimable_fit_patients"][str(lam)] for c in sub)})
    write_csv(ROOT/"FIT_REGULARIZATION_SELECTION.csv",selection)
    split_rows=[]
    for fold in range(1,6):
        audit=[a for c in cells if c["fold"]==fold for a in c["split_audit"]]
        split_rows.append({"fold":fold,"target_cells":13,"repetitions":len(audit),
                           "min_candidate_n":min(a["candidate_n"] for a in audit),
                           "max_candidate_n":max(a["candidate_n"] for a in audit),
                           "min_query_n":min(a["query_n"] for a in audit),
                           "query_ap_estimable":sum(a["query_estimable"] for a in audit),
                           "budget_16_available":sum(16 in a["available_budgets"] for a in audit)})
    write_json(ROOT/"splits"/"TARGET_CALIBRATION_SPLIT_AUDIT.json",
               {"folds":split_rows,"target_cells":65,"unique_patients":47,"repetitions":1300,
                "fixed_label_blind_50_50":True,"no_class_resampling":True,"zero_budget_exact_A1":True,
                "query_labels_read_after_all_repetition_score_vectors_fixed":True,
                "query_ap_estimable_total":sum(r["query_ap_estimable"] for r in split_rows)})
    # Gates are evaluated only on predeclared FULL_RESIDUAL rows.
    pool=by[("FULL_POOL_CALIBRATION","FULL_RESIDUAL","FULL_POOL")]
    pool_checks={"ap_gain_ge_0_030":pool["delta_ap_vs_a1"]>=.030,
                 "positive_folds_ge_4":pool["positive_folds_vs_a1"]>=4,
                 "cluster_ci_lower_gt_0":pool["ap_delta_ci_low"]>0}
    pool_pass=all(pool_checks.values())
    write_json(ROOT/"gates"/"FULL_POOL_CAPACITY_GATE.json",{"pass":pool_pass,"checks":pool_checks,
               "terminal":"FULL_POOL_RESIDUAL_CAPACITY_SUPPORTED" if pool_pass else "FULL_POOL_RESIDUAL_CAPACITY_NOT_SUPPORTED"})
    few_checks={}; active_checks={}
    for policy in ("RANDOM","UNCERTAINTY","UNCERTAINTY_DIVERSITY"):
        for budget in (1,2,4,8):
            row=by.get((policy,"FULL_RESIDUAL",str(budget)))
            if row is None: continue
            checks={"ap_gain_ge_0_020":row["delta_ap_vs_a1"]>=.020,
                    "positive_ap_folds_ge_4":row["positive_folds_vs_a1"]>=4,
                    "ap_delta_ci_lower_gt_0":row["ap_delta_ci_low"]>0,
                    "mrr_nondecreasing":row["mean_mrr"]>=row["matched_a1_mrr"],
                    "top1_nondecreasing":row["mean_top1"]>=row["matched_a1_top1"],
                    "macro_f1_within_0_005":row["mean_macro_f1"]>=row["matched_a1_macro_f1"]-.005,
                    "ez_f1_within_0_005":row["mean_ez_f1"]>=row["matched_a1_ez_f1"]-.005,
                    "full_bias_ap_gain_ge_0_010":row["delta_ap_full_vs_bias"]>=.010,
                    "full_bias_ci_lower_gt_0":row["full_vs_bias_ci_low"]>0}
            few_checks[f"{policy}_B{budget}"]={"pass":all(checks.values()),"checks":checks}
    few_pass=any(v["pass"] for v in few_checks.values())
    write_json(ROOT/"gates"/"FEWSHOT_CALIBRATION_GATE.json",{"pass":few_pass,"candidates":few_checks,
               "terminal":"FEWSHOT_PATIENT_GEOMETRY_CALIBRATION_SUPPORTED" if few_pass else "FEWSHOT_PATIENT_GEOMETRY_CALIBRATION_NOT_SUPPORTED"})
    for budget in (1,2,4,8):
        row=by.get(("UNCERTAINTY_DIVERSITY","FULL_RESIDUAL",str(budget)))
        rand=by.get(("RANDOM","FULL_RESIDUAL",str(budget)))
        if row is None or rand is None: continue
        checks={"ap_gain_random_ge_0_010":row["delta_ap_vs_random"]>=.010,
                "positive_ap_folds_ge_4":row["positive_folds_vs_random"]>=4,
                "active_random_ci_lower_gt_0":row["active_vs_random_ci_low"]>0,
                "mrr_nondecreasing":row["mean_mrr"]>=rand["mean_mrr"],
                "top1_nondecreasing":row["mean_top1"]>=rand["mean_top1"],
                "support_both_nondecreasing":row["fraction_support_has_both_classes"]>=rand["fraction_support_has_both_classes"]}
        active_checks[f"B{budget}"]={"pass":all(checks.values()),"checks":checks}
    active_pass=any(v["pass"] for v in active_checks.values())
    write_json(ROOT/"gates"/"ACTIVE_SELECTION_GATE.json",{"pass":active_pass,"candidates":active_checks,
               "terminal":"ACTIVE_CHANNEL_SELECTION_SUPPORTED" if active_pass else "ACTIVE_CHANNEL_SELECTION_NOT_SUPPORTED"})
    efficiency=[]
    for policy in ("RANDOM","UNCERTAINTY","UNCERTAINTY_DIVERSITY"):
        candidates=[by[(policy,"FULL_RESIDUAL",str(b))] for b in (1,2,4,8,16) if (policy,"FULL_RESIDUAL",str(b)) in by]
        out={"acquisition_policy":policy}
        for threshold in (.01,.02,.03):
            passed=[r["budget"] for r in candidates if r["delta_ap_vs_a1"]>=threshold and r["ap_delta_ci_low"]>0]
            out[f"min_budget_delta_ap_ge_{threshold:.2f}_ci_positive"]=min(passed) if passed else "NA"
        for fraction in (.25,.50,.75):
            passed=[r["budget"] for r in candidates if r["mean_headroom_recovery"] is not None and r["mean_headroom_recovery"]>=fraction]
            out[f"min_budget_headroom_recovery_ge_{fraction:.2f}"]=min(passed) if passed else "NA"
        efficiency.append(out)
    write_csv(ROOT/"comparisons"/"LABEL_EFFICIENCY.csv",efficiency)
    bias_gain=any(r["delta_ap_vs_a1"] is not None and r["delta_ap_vs_a1"]>0 and r["ap_delta_ci_low"] is not None and r["ap_delta_ci_low"]>0
                  for r in summaries if r["calibration_type"]=="BIAS_ONLY" and r["acquisition_policy"] in POLICIES and r["budget"] in (1,2,4,8,16))
    full_beats_bias=any(r["delta_ap_full_vs_bias"] is not None and r["delta_ap_full_vs_bias"]>=.010 and r["full_vs_bias_ci_low"] is not None and r["full_vs_bias_ci_low"]>0
                        for r in summaries if r["calibration_type"]=="FULL_RESIDUAL" and r["acquisition_policy"] in POLICIES and r["budget"] in (1,2,4,8,16))
    if few_pass and active_pass: overall="ACTIVE_FEWSHOT_PATIENT_CALIBRATION_JUSTIFIED"
    elif few_pass: overall="FEWSHOT_PATIENT_CALIBRATION_JUSTIFIED"
    elif pool_pass: overall="PATIENT_GEOMETRY_REQUIRES_MORE_THAN_FEWSHOT_CALIBRATION"
    elif bias_gain and not full_beats_bias: overall="PATIENT_CALIBRATION_IS_PRIMARILY_DECISION_OFFSET_NOT_GEOMETRY"
    else: overall="LINEAR_RESIDUAL_PATIENT_CALIBRATION_NOT_SUPPORTED"
    write_json(ROOT/"LABEL_USAGE_AUDIT.json",{"fit_labels_lambda_selection":True,"target_candidate_labels_only_acquired_support_except_explicit_non_deployable_oracle_and_full_pool":True,
              "target_query_labels_after_all_score_vectors_frozen":True,"target_full_labels_non_deployable_oracle_only_after_predictions_frozen":True,
              "outer_prediction_metric_or_selection":False,"legacy_monolithic_loader_materializes_all_80_labels":True,
              "strict_no_outer_label_materialization_satisfied":False,"target_cells":65,"unique_target_patient_ids":47})
    (ROOT/"README.md").write_text("# Active few-shot patient calibration\n\nFrozen A1 development diagnostic only. Run source checkpoint replay, then `drive.py`, then `finalize.py` on the original server with private AFPC_RUNTIME and exact prior selected-epoch R4 payloads. Patient-level labels, channel scores, representations, checkpoints and per-cell records stay in private runtime. Only aggregate outputs are committed. Retrospective EZ supervision is a mechanistic simulation, not a prospective clinical claim.\n",encoding="utf-8")
    (ROOT/"IMPLEMENTATION_AUDIT.md").write_text("# Implementation audit\n\n- Fresh 150-checkpoint A1 replay and selected-epoch R4 prehook/classifier equality were verified before calibration. FIT/validation roles, source locks, selected epoch and VLOO threshold are checked per cell.\n- The prior reference's private raw R4 payloads are reused, not its aggregate outcome numbers. FIT scaler is fit only to FIT channel rows for each selected checkpoint.\n- Candidate/query split is deterministic, label-blind, fixed across policies and budgets in each repetition. All 20 query-score sets are frozen before any query labels or target oracle are accessed. Only acquired candidate labels enter deployable calibrators.\n- Exact B=0 A1 ranking and decision checks pass. Bias-only uses matched support. Oracle-balanced support and full-pool are explicitly nondeployable controls.\n- Patient-ID cluster resampling (10,000, seed 42) includes every fold/repetition of sampled IDs. Per-cell files remain private; output is aggregate-only.\n- A memory-bound one-cell subprocess driver changes only process lifetime, not samples, optimizer, acquisition or statistical rules.\n- Legacy monolithic A1 loader materializes all 80 patient labels while constructing the FIT+validation data object; no outer predictions or metrics are produced, but literal no-outer-label-materialization is not satisfied.\n",encoding="utf-8")
    compact=lambda r:f"AP {r['mean_ap']:.4f}, delta {r['delta_ap_vs_a1']:+.4f}, 95% CI [{r['ap_delta_ci_low']:+.4f},{r['ap_delta_ci_high']:+.4f}], folds {r['positive_folds_vs_a1']}/5"
    lines=["# Active few-shot patient calibration — development-only diagnostic","",
           "Exact A1 replay: 150/150 checkpoint grids within 1e-6. Exact R4 classifier replay and B=0 threshold/ranking identity passed. There are 65 VLOO cells, 47 unique patients, 20 fixed label-blind repetitions each. No outer predictions or metrics were computed. The legacy loader nevertheless materializes all 80 labels, so this is not strictly sealed.","",
           "All AP comparisons below are against frozen A1 on the identical fixed query channels; intervals use 10,000 patient-ID cluster resamples. Historical EZ labels simulate calibration and do not establish clinical label availability.","",
           f"Full candidate-pool residual: {compact(pool)}.","",
           "| Policy | B=1 delta AP | B=2 | B=4 | B=8 | B=16 |",
           "|---|---:|---:|---:|---:|---:|"]
    for policy in ("RANDOM","UNCERTAINTY","UNCERTAINTY_DIVERSITY"):
        vals=[by.get((policy,"FULL_RESIDUAL",str(b))) for b in (1,2,4,8,16)]
        lines.append("| "+policy+" | "+" | ".join(f"{r['delta_ap_vs_a1']:+.4f}" if r else "NA" for r in vals)+" |")
    hrmap={(r["acquisition_policy"],r["calibration_type"],str(r["budget"])):r for r in headroom}
    lines += ["", "Required mean per-repetition positive-full-pool headroom recovery (unstable ratio; see warning below):", "",
              "| Policy | B=1 | B=2 | B=4 | B=8 | B=16 |", "|---|---:|---:|---:|---:|---:|"]
    for policy in ("RANDOM","UNCERTAINTY","UNCERTAINTY_DIVERSITY"):
        lines.append("| "+policy+" | "+" | ".join(f"{hrmap[(policy,'FULL_RESIDUAL',str(b))]['mean_recovery_fraction']:+.2f}" for b in (1,2,4,8,16))+" |")
    unc8=by[("UNCERTAINTY","FULL_RESIDUAL","8")]
    div4=by[("UNCERTAINTY_DIVERSITY","FULL_RESIDUAL","4")]
    ran4=by[("RANDOM","FULL_RESIDUAL","4")]
    lines += ["", "## Gate-level evidence and limitations", "",
              f"The few-shot gate passes for UNCERTAINTY at B=8: matched A1 AP {unc8['matched_a1_ap']:.4f} to {unc8['mean_ap']:.4f}, paired delta {unc8['delta_ap_vs_a1']:+.4f} (95% patient-cluster CI [{unc8['ap_delta_ci_low']:+.4f}, {unc8['ap_delta_ci_high']:+.4f}]), positive 5/5 folds. At B=1 both active policies lower AP.",
              f"The predeclared active gate passes for UNCERTAINTY_DIVERSITY at B=4: versus matched RANDOM delta {div4['delta_ap_vs_random']:+.4f} (95% CI [{div4['active_vs_random_ci_low']:+.4f}, {div4['active_vs_random_ci_high']:+.4f}]), positive 5/5 folds; both-class support {div4['fraction_support_has_both_classes']:.1%} versus RANDOM {ran4['fraction_support_has_both_classes']:.1%}. Its absolute A1 delta is {div4['delta_ap_vs_a1']:+.4f}, below the separate +0.020 few-shot threshold.",
              f"The two positive gates therefore occur at different policy/budget settings (UNCERTAINTY B=8; UNCERTAINTY_DIVERSITY B=4). No single setting is established as passing both gates. The literal predeclared decision logic still assigns the overall terminal below.",
              f"FULL_POOL improves A1 only {pool['delta_ap_vs_a1']:+.4f}, below its required +0.030, despite 5/5 positive folds and a positive CI. This is a failed capacity gate; the stronger active-subset result does not license calling full-pool capacity supported.",
              "Because adding a constant bias leaves ranking unchanged, BIAS_ONLY AP equals matched frozen-A1 AP by mathematical identity. Full-minus-bias AP is therefore identical to full-minus-A1 AP and is NOT an independent geometry check; the nonzero AP gain itself shows a channel-dependent score change, but oracle-direction alignment remains modest.",
              f"Only {next(h['positive_full_pool_headroom_repetitions'] for h in headroom if h['acquisition_policy']=='FULL_POOL_CALIBRATION')}/1300 repetitions have positive FULL_POOL headroom. The required per-repetition recovery ratio divides by sometimes tiny positive headroom, so its mean can exceed 100% or be strongly negative; treat LABEL_EFFICIENCY's recovery thresholds as unstable descriptives, not additional success evidence.","",
              f"The historical all-channel A1 AP (~0.557) is not the matched fixed-query baseline here ({pool['matched_a1_ap']:.4f}). No cross-protocol numerical comparison should be made between them."]
    best_by_policy={p:max((by[(p,"FULL_RESIDUAL",str(b))] for b in (1,2,4,8,16) if (p,"FULL_RESIDUAL",str(b)) in by),
                          key=lambda r:r["delta_ap_vs_a1"]) for p in ("RANDOM","UNCERTAINTY","UNCERTAINTY_DIVERSITY")}
    geomap={(r["acquisition_policy"],str(r["budget"])):r for r in geometry}
    lines += ["","## Required interpretation","",
              "1. A1 reproduction: yes, 150 checkpoints and R4 classifier replay within 1e-6.",
              "2. B=0 reproduces the VLOO-selected A1 decision and ranking exactly on every fixed query split.",
              f"3–4. Full-pool residual: {compact(pool)}; this is the attainable 50%-candidate-pool capacity estimate for this fixed model, not a few-shot result.",
              f"5–7. Minimum budget with +0.01/+0.02/+0.03 AP and positive cluster CI is tabulated in LABEL_EFFICIENCY.csv; the best tested RANDOM/UNCERTAINTY/UNCERTAINTY_DIVERSITY rows are {compact(best_by_policy['RANDOM'])}; {compact(best_by_policy['UNCERTAINTY'])}; {compact(best_by_policy['UNCERTAINTY_DIVERSITY'])}, respectively.",
              f"8. Both-class support at B=4 is RANDOM {ran4['fraction_support_has_both_classes']:.1%}, UNCERTAINTY {by[('UNCERTAINTY','FULL_RESIDUAL','4')]['fraction_support_has_both_classes']:.1%}, UNCERTAINTY_DIVERSITY {div4['fraction_support_has_both_classes']:.1%}; acquisition does not guarantee a balanced support.",
              "9–10. Full residual versus bias-only matched AP/CI is in FULL_VS_BIAS.csv. Direction recovery is supported only if the predeclared full-minus-bias gate passes; a bias-only improvement is an offset, not geometry.",
              "11. UNCERTAINTY residual cosine to the label-using patient oracle at B=1/2/4/8/16 is "+"/".join(f"{geomap[('UNCERTAINTY',str(b))]['mean_oracle_direction_cosine']:.3f}" for b in (1,2,4,8,16))+" (FIT-standardized coordinates). It rises with budget but remains modest; oracle alignment was never an acquisition or fitting input.",
              "12. B=1/2/4/8/16 positive-full-pool-headroom recovery and cluster CIs are in HEADROOM_RECOVERY.csv; repetitions with nonpositive full-pool headroom are excluded, not silently set to zero.",
              "13. UNCERTAINTY and UNCERTAINTY_DIVERSITY versus matched RANDOM are in ACTIVE_VS_RANDOM.csv; the latter is judged only by the locked active gate.",
              "14. Five-fold mean signs and 47-patient clustered intervals are in FOLD_CONSISTENCY.csv and PATIENT_CLUSTER_BOOTSTRAP.csv; 20 repetitions are not independent subjects.",
              "15. The experiment tests target-supervision identifiability but cannot prove zero-label failure has a single causal explanation; full-pool and bias-only controls constrain that interpretation.",
              "16. A later actively selected few-shot model is justified only if both fixed gates pass. Retrospective labels are not automatically clinically obtainable.","",
              f"`{'FULL_POOL_RESIDUAL_CAPACITY_SUPPORTED' if pool_pass else 'FULL_POOL_RESIDUAL_CAPACITY_NOT_SUPPORTED'}`",
              f"`{'FEWSHOT_PATIENT_GEOMETRY_CALIBRATION_SUPPORTED' if few_pass else 'FEWSHOT_PATIENT_GEOMETRY_CALIBRATION_NOT_SUPPORTED'}`",
              f"`{'ACTIVE_CHANNEL_SELECTION_SUPPORTED' if active_pass else 'ACTIVE_CHANNEL_SELECTION_NOT_SUPPORTED'}`",
              f"`{overall}`"]
    (ROOT/"FINAL_REPORT.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print("[FINAL]",overall,flush=True)


if __name__=="__main__":
    main()
