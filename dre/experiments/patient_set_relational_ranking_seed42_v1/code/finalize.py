"""Aggregate compact outputs; never export patient/channel records."""
from __future__ import annotations

import json
import pickle
from collections import defaultdict

import numpy as np
from scipy.stats import kendalltau,spearmanr

import protocol as p
import evaluate as ev

N_BOOT=10000
METRICS=("ap","auc","mrr","top1","ndcg","macro_f1","ez_f1","ba","predicted_ez_fraction")


def mean(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def bootstrap(rows,values,ids,counts):
    index={sid:i for i,sid in enumerate(ids)}
    numer=np.zeros(len(ids));denom=np.zeros(len(ids))
    for row,value in zip(rows,values):
        if np.isfinite(value):
            at=index[row["sid"]];numer[at]+=value;denom[at]+=1
    if not denom.sum():return dict(mean=None,low=None,high=None,n=0)
    dn=counts@denom;draws=np.divide(counts@numer,dn,out=np.full(len(counts),np.nan),where=dn>0)
    if np.isfinite(draws).sum()<.99*len(draws):raise RuntimeError("Degenerate patient-ID bootstrap")
    good=draws[np.isfinite(draws)]
    return dict(mean=float(numer.sum()/denom.sum()),low=float(np.quantile(good,.025)),
                high=float(np.quantile(good,.975)),n=int(denom.sum()))


def aggregate():
    manifest=ev.frozen()
    status=json.loads((p.RUNTIME/"EVALUATION_STATUS.json").read_text(encoding="utf-8"))
    if status["score_manifest_sha"]!=p.sha(p.RUNTIME/"ALL_SCORES_FROZEN_BEFORE_NEW_TARGET_LABEL_USE.json"):
        raise RuntimeError("Metric evaluation not tied to frozen scores")
    with (p.RUNTIME/"private"/"EVALUATED_METRICS.pkl").open("rb") as f:rows=pickle.load(f)
    with (p.RUNTIME/"private"/"RANK_CHANGE_INPUTS.pkl").open("rb") as f:ranks=pickle.load(f)
    active=manifest["active_architectures"]
    expected=65*len(active)*2*20*5
    if len(rows)!=expected or len(ranks)!=65*len(active)*2:
        raise RuntimeError("Metric/rank coverage incomplete")
    by={(r["fold"],r["sid"],r["rep"],r["arch"],r["arm"],r["context"]):r for r in rows}
    if len(by)!=len(rows):raise RuntimeError("Duplicate context metric record")
    ids=sorted({r["sid"] for r in rows})
    if len(ids)!=47:raise RuntimeError("Target patient-ID count changed")
    rng=np.random.default_rng(42)
    draw=rng.integers(0,len(ids),size=(N_BOOT,len(ids)))
    counts=np.zeros((N_BOOT,len(ids)),dtype=np.int16)
    np.add.at(counts,(np.arange(N_BOOT)[:,None],draw),1)
    counts=counts.astype(np.float32)
    def ref(r,context):return by[(r["fold"],r["sid"],r["rep"],r["arch"],r["arm"],context)]
    matrix=[];boot=[];folds=[];comparisons={name:[] for name in
        ("FULL_VS_QUERY_CONTEXT","CORRECT_VS_WRONG_CONTEXT","CORRECT_CONTEXT_VS_SHUFFLED_RELATION")}
    comparison_pair={"FULL_VS_QUERY_CONTEXT":("full","query_only"),
                     "CORRECT_VS_WRONG_CONTEXT":("full","wrong"),
                     "CORRECT_CONTEXT_VS_SHUFFLED_RELATION":("full","shuffled")}
    for arch in active:
        for arm in p.ARMS:
            full=[r for r in rows if r["arch"]==arch and r["arm"]==arm and r["context"]=="full"]
            if len(full)!=1300:raise RuntimeError("Expected 65 x 20 full-context rows")
            for context in ("full","query_only"):
                part=[ref(r,context) for r in full]
                delta=[r["ap"]-ref(r,"a1")["ap"] for r in part]
                stat=bootstrap(part,delta,ids,counts)
                line=dict(variant=p.arm_name(arch,arm),arch=arch,arm=arm,context=context,
                          n_cells=65,n_patient_ids=47,n_repetitions=1300,
                          **{metric:mean([r[metric] for r in part]) for metric in METRICS},
                          delta_ap_vs_a1=stat["mean"],delta_ap_ci_low=stat["low"],delta_ap_ci_high=stat["high"])
                matrix.append(line)
                boot.append(dict(variant=line["variant"],context=context,comparison="vs_A1",metric="ap",
                                 mean=stat["mean"],ci_low=stat["low"],ci_high=stat["high"],
                                 n_estimable=stat["n"],patient_clusters=47,resamples=N_BOOT))
            for name,(left,right) in comparison_pair.items():
                lhs=[ref(r,left) for r in full]
                delta=[a["ap"]-ref(r,right)["ap"] for r,a in zip(full,lhs)]
                stat=bootstrap(lhs,delta,ids,counts)
                comparisons[name].append(dict(variant=p.arm_name(arch,arm),arch=arch,arm=arm,
                                              delta_ap=stat["mean"],ci_low=stat["low"],ci_high=stat["high"],
                                              n_estimable=stat["n"],patient_clusters=47))
                boot.append(dict(variant=p.arm_name(arch,arm),context=name,comparison=left+"_minus_"+right,
                                 metric="ap",mean=stat["mean"],ci_low=stat["low"],ci_high=stat["high"],
                                 n_estimable=stat["n"],patient_clusters=47,resamples=N_BOOT))
            for fold in range(1,6):
                sub=[r for r in full if r["fold"]==fold]
                folds.append(dict(variant=p.arm_name(arch,arm),fold=fold,n_cells=13,n_repetitions=260,
                                  ap=mean([r["ap"] for r in sub]),
                                  delta_ap_vs_a1=mean([r["ap"]-ref(r,"a1")["ap"] for r in sub]),
                                  full_minus_query=mean([r["ap"]-ref(r,"query_only")["ap"] for r in sub]),
                                  correct_minus_wrong=mean([r["ap"]-ref(r,"wrong")["ap"] for r in sub]),
                                  correct_minus_shuffled=mean([r["ap"]-ref(r,"shuffled")["ap"] for r in sub])))
    baseline=[r for r in rows if r["arch"]==active[0] and r["arm"]==p.ARMS[0] and r["context"]=="a1"]
    source_ap=mean([r["ap"] for r in baseline])
    if abs(source_ap-p.SOURCE_AP)>1e-6:
        raise RuntimeError(f"Exact A1 fixed-query AP mismatch: {source_ap} != {p.SOURCE_AP}")
    matrix.insert(0,dict(variant="R0_A1",arch="R0_A1",arm="SOURCE",context="full",n_cells=65,
                         n_patient_ids=47,n_repetitions=1300,
                         **{metric:mean([r[metric] for r in baseline]) for metric in METRICS},
                         delta_ap_vs_a1=0.,delta_ap_ci_low=0.,delta_ap_ci_high=0.))
    p.afc.write_csv(p.ROOT/"RELATIONAL_VARIANT_MATRIX.csv",matrix)
    p.afc.write_csv(p.ROOT/"PATIENT_CLUSTER_BOOTSTRAP.csv",boot)
    p.afc.write_csv(p.ROOT/"FOLD_CONSISTENCY.csv",folds)
    for name,items in comparisons.items():p.afc.write_csv(p.ROOT/(name+".csv"),items)
    return rows,ranks,matrix,folds,comparisons,ids,source_ap


def rank_audit(ranks):
    output=[]
    for row in ranks:
        y=np.asarray(row["y"],dtype=np.int8);base=np.asarray(row["a1_all"]);score=np.asarray(row["full_all"])
        if len(y)!=len(base) or len(y)!=len(score):raise RuntimeError("Rank-change input mismatch")
        k=min(5,len(y));before=set(np.argsort(-base,kind="stable")[:k]);after=set(np.argsort(-score,kind="stable")[:k])
        # Kendall/Spearman compare channel rankings, not label ordering.
        tau=kendalltau(base,score).statistic;rho=spearmanr(base,score).statistic
        output.append(dict(variant=p.arm_name(row["arch"],row["arm"]),fold=row["fold"],
                           kendall_tau=float(tau),spearman_rho=float(rho),
                           changed_top5_channels=len(before^after)//2,
                           ez_promoted_top5=sum(y[i] for i in after-before),
                           ez_demoted_top5=sum(y[i] for i in before-after)))
    summary=[]
    for variant in sorted({r["variant"] for r in output}):
        part=[r for r in output if r["variant"]==variant]
        summary.append(dict(variant=variant,n_cells=len(part),
                            **{k:mean([r[k] for r in part]) for k in
                               ("kendall_tau","spearman_rho","changed_top5_channels",
                                "ez_promoted_top5","ez_demoted_top5")},
                            total_ez_promoted_top5=sum(r["ez_promoted_top5"] for r in part),
                            total_ez_demoted_top5=sum(r["ez_demoted_top5"] for r in part)))
    p.afc.write_csv(p.ROOT/"RANK_CHANGE_AUDIT.csv",summary)
    return summary


def failure_strata(rows,ids):
    # The strata are based exclusively on frozen A1 AP, never relational outcome.
    base=defaultdict(list)
    for r in rows:
        if r["context"]=="a1" and r["arm"]==p.ARMS[0] and r["arch"]==p.R1:
            base[r["sid"]].append(r["ap"])
    by_id={sid:mean(values) for sid,values in base.items()}
    finite=sorted((value,sid) for sid,value in by_id.items() if value is not None)
    if len(finite)!=47:raise RuntimeError("Failure strata require all 47 source patients")
    cut=(finite[15][0],finite[31][0]);ranked={sid:i for i,(_,sid) in enumerate(finite)}
    strata={sid:("poor_A1" if ranked[sid]<16 else "medium_A1" if ranked[sid]<32 else "strong_A1") for sid in ids}
    output=[]
    for arch in sorted({r["arch"] for r in rows}):
        for arm in p.ARMS:
            full=[r for r in rows if r["arch"]==arch and r["arm"]==arm and r["context"]=="full"]
            by={(r["fold"],r["sid"],r["rep"],r["arch"],r["arm"],r["context"]):r for r in rows}
            for name in ("poor_A1","medium_A1","strong_A1"):
                part=[r for r in full if strata[r["sid"]]==name]
                delta=[r["ap"]-by[(r["fold"],r["sid"],r["rep"],arch,arm,"a1")]["ap"] for r in part]
                output.append(dict(variant=p.arm_name(arch,arm),stratum=name,
                                   n_patient_ids=len({r["sid"] for r in part}),n_repetitions=len(part),
                                   a1_ap=mean([by[(r["fold"],r["sid"],r["rep"],arch,arm,"a1")]["ap"] for r in part]),
                                   relational_ap=mean([r["ap"] for r in part]),delta_ap=mean(delta),
                                   a1_tercile_cut_1=cut[0],a1_tercile_cut_2=cut[1]))
    p.afc.write_csv(p.ROOT/"FAILURE_STRATIFIED_ANALYSIS.csv",output)
    return output


def gates(matrix,folds,comparisons):
    by_comp={name:{r["variant"]:r for r in values} for name,values in comparisons.items()}
    results={}
    for r in matrix:
        if r["variant"]=="R0_A1" or r["context"]!="full":continue
        variant=r["variant"];fs=[f for f in folds if f["variant"]==variant]
        positive=sum(f["delta_ap_vs_a1"]>0 for f in fs)
        noninferior=(r["mrr"]>=matrix[0]["mrr"]-.005 and r["top1"]>=matrix[0]["top1"]-.005)
        wrong=by_comp["CORRECT_VS_WRONG_CONTEXT"][variant]
        shuffle=by_comp["CORRECT_CONTEXT_VS_SHUFFLED_RELATION"][variant]
        context_fold=max(sum(f["correct_minus_wrong"]>0 for f in fs),
                         sum(f["correct_minus_shuffled"]>0 for f in fs))
        mechanism=bool(wrong["delta_ap"]>0 and wrong["ci_low"]>0 and
                       shuffle["delta_ap"]>0 and context_fold>=4)
        stat=bool(r["delta_ap_ci_low"]>0 and positive>=4 and noninferior)
        results[variant]=dict(ap=r["ap"],delta_ap=r["delta_ap_vs_a1"],delta_ci_low=r["delta_ap_ci_low"],
                              positive_folds=positive,ranking_noninferior=noninferior,
                              correct_minus_wrong=wrong["delta_ap"],wrong_ci_low=wrong["ci_low"],
                              correct_minus_shuffled=shuffle["delta_ap"],context_positive_folds=context_fold,
                              GATE1_RELATIONAL_ZEROSHOT_IMPROVEMENT_SUPPORTED=bool(stat and r["delta_ap_vs_a1"]>=.015),
                              GATE2_RELATIONAL_ZEROSHOT_REACHES_CURRENT_B8=bool(stat and r["ap"]>=p.B8_CURRENT),
                              GATE3_RELATIONAL_ZEROSHOT_REACHES_BEST_B8=bool(stat and r["ap"]>=p.B8_BEST),
                              GATE4_PATIENT_RELATIONAL_CONTEXT_SUPPORTED=mechanism,
                              frozen_strong_positive=bool(stat and r["ap"]>=p.B8_CURRENT and wrong["delta_ap"]>0),
                              frozen_promising=bool(r["ap"]>=.590 and r["delta_ap_ci_low"]>0 and
                                                    positive>=4 and wrong["delta_ap"]>0))
    primary=json.loads((p.ROOT/"FIT_SELECTION_LOCK.json").read_text(encoding="utf-8"))["primary_candidate"]
    name=p.arm_name(primary["arch"],primary["arm"])
    selected=results[name]
    stage2=bool(selected["frozen_strong_positive"] or selected["frozen_promising"])
    if any(r["GATE3_RELATIONAL_ZEROSHOT_REACHES_BEST_B8"] for r in results.values()):
        terminal="PATIENT_RELATIONAL_ZEROSHOT_RECOVERY_JUSTIFIED"
    elif any(r["GATE2_RELATIONAL_ZEROSHOT_REACHES_CURRENT_B8"] for r in results.values()):
        terminal="RELATIONAL_ZEROSHOT_APPROACHES_B8"
    elif any(r["GATE1_RELATIONAL_ZEROSHOT_IMPROVEMENT_SUPPORTED"] and not r["GATE4_PATIENT_RELATIONAL_CONTEXT_SUPPORTED"] for r in results.values()):
        terminal="RELATIONAL_GAIN_NOT_ATTRIBUTABLE_TO_PATIENT_CONTEXT"
    elif any(r["delta_ci_low"]>0 and .585<=r["ap"]<.590 for r in results.values()):
        terminal="RELATIONAL_SIGNAL_WEAK_BUT_PRESENT"
    else:terminal="R4_RELATIONAL_ZEROSHOT_NOT_SUPPORTED"
    row=dict(lock_sha=p.LOCK_SHA,source_ap=p.SOURCE_AP,b8_current=p.B8_CURRENT,b8_best=p.B8_BEST,
             fit_frozen_primary=name,stage2_allowed=stage2,
             stage2_not_launched_by_phase1_finalizer=True,terminal=terminal,variants=results)
    p.atomic_json(p.ROOT/"RELATIONAL_GATES.json",row)
    return row


def report(matrix,comparisons,ranks,strata,g):
    lines=["# Patient-set Relational Ranking Zero-shot Study", "",
           f"Exact A1/B0 fixed-query AP: {p.SOURCE_AP:.9f}; 150 source checkpoints, 65 cells, 47 IDs, 20 fixed repetitions.",
           "All readout selection used FIT patients. All new target/control scores were hash-frozen before target-label metrics.",
           "Inference is transductive B=0: full context uses all patient channels' unlabeled R4/A1 margins, with no target labels or updates.",
           "Strict target-label sequencing is false: legacy VLOO source selection used cross-patient labels, and its loader materializes target labels. No target labels were indexed for the new readout selection or score freeze.",
           "", "## Frozen R4 results", "",
           "| Variant | Context | AP | Delta AP vs A1 | Patient-ID CI | MRR | Top1 |", "|---|---|---:|---:|---|---:|---:|"]
    for r in matrix:
        ci=f"[{r['delta_ap_ci_low']:+.6f}, {r['delta_ap_ci_high']:+.6f}]"
        lines.append(f"| {r['variant']} | {r['context']} | {r['ap']:.6f} | {r['delta_ap_vs_a1']:+.6f} | {ci} | {r['mrr']:.6f} | {r['top1']:.6f} |")
    lines.extend(["", "## Context controls", ""])
    for name,items in comparisons.items():
        lines.append("### "+name)
        for r in items:lines.append(f"- {r['variant']}: delta AP {r['delta_ap']:+.6f}, 95% CI [{r['ci_low']:+.6f}, {r['ci_high']:+.6f}].")
        lines.append("")
    lines.extend(["## Decision", "",f"- Terminal: `{g['terminal']}`.",
                  f"- FIT-frozen primary: `{g['fit_frozen_primary']}`; Stage 2 allowed: `{g['stage2_allowed']}`.",
                  "- R3 was run only if the FIT-only R2-versus-R1 gate passed; see `FIT_SELECTION_LOCK.json`.",
                  "- Channel-rank changes and frozen-A1 failure strata are in the compact audit CSVs.",
                  "- Current B8 AP 0.599632; best B8 AP 0.605291. No B8 target support was used.",
                  ""])
    (p.ROOT/"FINAL_REPORT.md").write_text("\n".join(lines),encoding="utf-8")


def main():
    rows,ranks,matrix,folds,comparisons,ids,_=aggregate()
    rank_summary=rank_audit(ranks)
    strata=failure_strata(rows,ids)
    g=gates(matrix,folds,comparisons)
    source=p.PROJECT/"zeroshot_geometry_canonicalization_metadg_seed42_v1"
    (p.ROOT/"SOURCE_REPRODUCTION.json").write_bytes((source/"SOURCE_REPRODUCTION.json").read_bytes())
    (p.ROOT/"B0_IDENTITY_AUDIT.json").write_bytes((source/"B0_IDENTITY_AUDIT.json").read_bytes())
    p.atomic_json(p.ROOT/"LABEL_USAGE_AUDIT.json",
                  dict(lock_sha=p.LOCK_SHA,final_inference_B=0,target_gradient_updates=0,
                       target_labels_used_for_FIT_selection=False,target_labels_indexed_before_new_score_freeze=False,
                       legacy_loader_materializes_target_labels=True,
                       strict_target_label_sequencing=False,
                       reason="Exact A1 VLOO uses other-target labels in source checkpoint selection; legacy representation loader materializes y.",
                       score_manifest_sha=p.sha(p.RUNTIME/"ALL_SCORES_FROZEN_BEFORE_NEW_TARGET_LABEL_USE.json")))
    (p.ROOT/"IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\nExact A1 selected-epoch R4 and source margins; 17 source contexts across 65 cells. "
        "FIT-only 80/20 patient split, full all-pair relation training/inference, zero-initialized residual, "
        "FIT-selected lambda and epochs, full-FIT retraining. The two arms are BCE-only and FIT-selected positive ranking loss. "
        "Every target/control score was hash-frozen before new outcome metrics; 20 deterministic 50/50 fixed queries, "
        "47-ID cluster bootstrap with 10,000 draws. Full context is transductive B=0. "
        "Wrong donor is FIT-only and closest in channel count; shuffled control permutes reference A1-score/R4 pairing. "
        "R1's shuffled control is a designed identity because its summary consumes only R4. "
        "Strict target-label sequencing is false under inherited A1 VLOO and loader; see label audit.\n",encoding="utf-8")
    report(matrix,comparisons,rank_summary,strata,g)
    print(f"[PHASE1_FINALIZED] terminal={g['terminal']} stage2={g['stage2_allowed']}",flush=True)


if __name__=="__main__":main()
