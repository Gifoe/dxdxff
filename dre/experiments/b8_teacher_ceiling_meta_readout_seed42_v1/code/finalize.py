"""Aggregate-only publication of the complete locked B=8 ceiling study."""
from __future__ import annotations

import hashlib
import json
import pickle
from collections import defaultdict

import numpy as np

import teacher_core as tc
import freeze
from fit_select import context_key

METRICS=("ap","auc","mrr","top1","ndcg","macro_f1","ez_f1","ba","predicted_ez_fraction")
B8=("CURRENT_64D_B8","RETUNED_64D_B8","PCA_D4_B8","PCA_D8_B8","PCA_D16_B8",
    "META_D4_B8","META_D8_B8","META_D16_B8","PROTOTYPE_B8")
VARIANTS=("FROZEN_A1",)+B8+("CURRENT_64D_FULLPOOL","RETUNED_64D_FULLPOOL",
          "PCA_D4_FULLPOOL","PCA_D8_FULLPOOL","PCA_D16_FULLPOOL",
          "META_D4_FULLPOOL","META_D8_FULLPOOL","META_D16_FULLPOOL","ORACLE_BALANCED_B8")
N_BOOT=10000


def mean(xs):
    a=np.asarray(xs,dtype=float);a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def load():
    rows=[];cells=[]
    for fold in range(1,6):
        for src in tc.selected_rows(fold):
            sid=src["subject_id"];stem=hashlib.sha256(sid.encode()).hexdigest()[:16]
            with (tc.RUNTIME/"private"/"target"/f"fold_{fold}"/(stem+".pkl")).open("rb") as f:cell=pickle.load(f)
            if cell["lock_sha"]!=tc.LOCK_SHA or cell["sid"]!=sid or cell["fold"]!=fold or not cell["score_vectors_frozen_before_query_labels"] or cell["prior_replay_max_ap_error"]>1e-8:
                raise RuntimeError("Private target cell provenance/replay failed")
            cells.append(cell);rows.extend(cell["records"])
    if len(cells)!=65 or len({c["sid"] for c in cells})!=47:raise RuntimeError("65 cells / 47 patient IDs not complete")
    exact={(r["fold"],r["sid"],r["repetition"],r["variant"]):r for r in rows}
    if len(exact)!=len(rows):raise RuntimeError("Duplicate target record")
    for c in cells:
        for rep in range(20):
            found={r["variant"] for r in c["records"] if r["repetition"]==rep}
            if found!=set(VARIANTS):raise RuntimeError(f"Variant coverage not complete: {found^set(VARIANTS)}")
    return cells,rows,exact


def boot(values,record_rows,ids,counts):
    by={sid:i for i,sid in enumerate(ids)}
    sums=np.zeros(len(ids));n=np.zeros(len(ids))
    for v,r in zip(values,record_rows):
        if v is not None and np.isfinite(v):
            i=by[r["sid"]];sums[i]+=v;n[i]+=1
    if n.sum()==0:return None,None,None,0
    den=counts@n
    draws=np.divide(counts@sums,den,out=np.full(len(counts),np.nan),where=den>0)
    draws=draws[np.isfinite(draws)]
    if len(draws)<.99*len(counts):raise RuntimeError("Patient bootstrap degeneracy")
    return float(sums.sum()/n.sum()),float(np.quantile(draws,.025)),float(np.quantile(draws,.975)),int(n.sum())


def finite_delta(a,b):
    x=a["ap"];y=b["ap"]
    return float(x-y) if np.isfinite(x) and np.isfinite(y) else np.nan


def fit_selection_outputs():
    grid=defaultdict(list);chosen=defaultdict(int);count=defaultdict(int)
    for fold in range(1,6):
        for ctx in tc.fold_contexts(fold):
            with (tc.RUNTIME/"private"/tc.FIT_SELECTION_FOLDER/(context_key(ctx)+".pkl")).open("rb") as f:sel=pickle.load(f)
            if sel["lock_sha"]!=tc.LOCK_SHA:raise RuntimeError("FIT selection provenance failed")
            for variant,item in sel["variants"].items():
                for kind in ("b8","fullpool"):
                    key=(fold,variant,kind)
                    count[key]+=1
                    best=item[kind]
                    chosen[(key,best["lambda_w"],best["lambda_b"])]+=1
                    for g in item[kind+"_grid"]:
                        grid[(key,g["lambda_w"],g["lambda_b"])].append(g["fit_patient_equal_ap"])
    outs={"b8":[],"fullpool":[]}
    for (key,lw,lb),vals in sorted(grid.items()):
        fold,variant,kind=key
        outs[kind].append({"fold":fold,"variant":variant,"lambda_w":lw,"lambda_b":lb,
                           "mean_fit_patient_equal_query_ap_across_contexts":mean(vals),
                           "selected_contexts":chosen[(key,lw,lb)],"context_count":count[key]})
    tc.afc.write_csv(tc.ROOT/"FIT_B8_HYPERPARAM_SELECTION.csv",outs["b8"])
    tc.afc.write_csv(tc.ROOT/"FULLPOOL_HYPERPARAM_SELECTION.csv",outs["fullpool"])


def main():
    tc.preflight()
    freeze.verify()
    cells,rows,exact=load()
    fit_selection_outputs()
    ids=sorted({r["sid"] for r in rows})
    rng=np.random.default_rng(42)
    draws=rng.integers(0,len(ids),size=(N_BOOT,len(ids)))
    counts=np.zeros((N_BOOT,len(ids)),dtype=np.int16)
    np.add.at(counts,(np.arange(N_BOOT)[:,None],draws),1)
    counts=counts.astype(np.float32)
    groups=defaultdict(list)
    for r in rows:groups[r["variant"]].append(r)
    matrix=[];pair=[];fold_rows=[];boot_rows=[];support=[];alignment=[]
    summaries={}
    for variant in VARIANTS:
        gr=groups[variant]
        if len(gr)!=1300:raise RuntimeError(f"Incomplete variant {variant}")
        def partner(r,v):return exact[(r["fold"],r["sid"],r["repetition"],v)]
        a1=np.asarray([finite_delta(r,partner(r,"FROZEN_A1")) for r in gr])
        b8=np.asarray([finite_delta(r,partner(r,"CURRENT_64D_B8")) for r in gr])
        pa=boot(a1,gr,ids,counts);pb=boot(b8,gr,ids,counts)
        row={"variant":variant,"n_cells":65,"n_unique_patients":47,"n_repetitions":1300,
             "n_estimable_ap":int(sum(np.isfinite(r["ap"]) for r in gr)),
             **{f"mean_{m}":mean([r[m] for r in gr]) for m in METRICS},
             "delta_ap_vs_a1":pa[0],"delta_ap_vs_current_b8":pb[0],
             "delta_ap_vs_current_b8_ci_low":pb[1],"delta_ap_vs_current_b8_ci_high":pb[2]}
        summaries[variant]=row;matrix.append(row)
        pair.append({"variant":variant,"ap":row["mean_ap"],"delta_ap_vs_a1":pa[0],
                     "delta_ap_vs_current_b8":pb[0],"ci_low":pb[1],"ci_high":pb[2],
                     "positive_folds_vs_current_b8":sum((mean([v for v,r in zip(b8,gr) if r["fold"]==f]) or 0)>0 for f in range(1,6))})
        for label,st in (("delta_ap_vs_a1",pa),("delta_ap_vs_current_b8",pb)):
            boot_rows.append({"variant":variant,"statistic":label,"mean":st[0],"ci_low":st[1],"ci_high":st[2],
                              "n_estimable":st[3],"n_patient_clusters":47,"resamples":N_BOOT,"seed":42})
        for fold in range(1,6):
            fr=[r for r in gr if r["fold"]==fold]
            fold_rows.append({"variant":variant,"fold":fold,"n_cells":13,"n_repetitions":260,
                              "mean_ap":mean([r["ap"] for r in fr]),
                              "delta_ap_vs_a1":mean([v for v,r in zip(a1,gr) if r["fold"]==fold]),
                              "delta_ap_vs_current_b8":mean([v for v,r in zip(b8,gr) if r["fold"]==fold])})
        support.append({"variant":variant,"mean_support_n":mean([r["support_n"] for r in gr]),
                        "mean_support_ez":mean([r["support_ez"] for r in gr]),
                        "mean_support_nez":mean([r["support_nez"] for r in gr]),
                        "fraction_one_class_support":mean([r["one_class_support"] for r in gr]),
                        "fraction_prototype_fallback":mean([r["prototype_fallback"] for r in gr])})
        alignment.append({"variant":variant,"mean_oracle_direction_cosine":mean([r["direction_cosine"] for r in gr]),
                          "mean_direction_norm":mean([r["direction_norm"] for r in gr]),
                          "oracle_used_for_training_or_selection":False})
    tc.afc.write_csv(tc.ROOT/"TEACHER_VARIANT_MATRIX.csv",matrix)
    tc.afc.write_csv(tc.ROOT/"B8_TEACHER_COMPARISONS.csv",pair)
    tc.afc.write_csv(tc.ROOT/"FOLD_CONSISTENCY.csv",fold_rows)
    tc.afc.write_csv(tc.ROOT/"PATIENT_CLUSTER_BOOTSTRAP.csv",boot_rows)
    tc.afc.write_csv(tc.ROOT/"SUPPORT_CLASS_COMPOSITION.csv",support)
    tc.afc.write_csv(tc.ROOT/"TEACHER_DIRECTION_ALIGNMENT.csv",alignment)
    full_pairs=[("CURRENT_64D_B8","CURRENT_64D_FULLPOOL"),
                ("RETUNED_64D_B8","RETUNED_64D_FULLPOOL")]+[(f"{kind}_D{d}_B8",f"{kind}_D{d}_FULLPOOL") for kind in ("PCA","META") for d in tc.DIMS]
    diagnostic=[]
    for b8_name,pool_name in full_pairs:
        gr=groups[b8_name]
        vals=np.asarray([finite_delta(r,exact[(r["fold"],r["sid"],r["repetition"],pool_name)]) for r in gr])
        stat=boot(vals,gr,ids,counts)
        diagnostic.append({"b8_variant":b8_name,"fullpool_variant":pool_name,
                           "b8_ap":summaries[b8_name]["mean_ap"],"fullpool_ap":summaries[pool_name]["mean_ap"],
                           "delta_b8_minus_fullpool":stat[0],"ci_low":stat[1],"ci_high":stat[2],
                           "positive_folds_b8_minus_fullpool":sum((mean([v for v,r in zip(vals,gr) if r["fold"]==f]) or 0)>0 for f in range(1,6))})
    tc.afc.write_csv(tc.ROOT/"FULLPOOL_DIAGNOSTIC.csv",diagnostic)
    deploy=[x for x in B8 if x!="CURRENT_64D_B8"]
    best=max(deploy,key=lambda v:summaries[v]["mean_ap"])
    bestrow=summaries[best]
    bestpair=next(x for x in pair if x["variant"]==best)
    foldok=bestpair["positive_folds_vs_current_b8"]>=4
    ciok=bestpair["ci_low"]>0
    rankok=bestrow["mean_mrr"]>=summaries["CURRENT_64D_B8"]["mean_mrr"]-.005 and bestrow["mean_top1"]>=summaries["CURRENT_64D_B8"]["mean_top1"]-.005
    milestones={f"TEACHER_AP_0{int(t*100):02d}_REACHED":bool(bestrow["mean_ap"]>=t and ciok and foldok and rankok) for t in (.62,.65,.68,.70)}
    impro=bool(bestpair["delta_ap_vs_current_b8"]>=.020 and ciok and foldok)
    anomaly=diagnostic[1]
    if anomaly["delta_b8_minus_fullpool"]<0:terminal="PREVIOUS_B8_FULLPOOL_REVERSAL_EXPLAINED_BY_REGULARIZATION"
    elif anomaly["ci_low"]>0:terminal="BOUNDARY_FOCUSED_SUPPORT_OUTPERFORMS_DENSE_PATIENT_LABELING"
    else:terminal="B8_CAPTURES_MOST_USEFUL_PATIENT_SUPERVISION"
    gates={"best_b8_teacher":best,"best_b8_ap":bestrow["mean_ap"],"best_delta_ap_vs_current_b8":bestpair["delta_ap_vs_current_b8"],
           "best_delta_ci_low":bestpair["ci_low"],"best_positive_folds":bestpair["positive_folds_vs_current_b8"],
           "rank_not_materially_worse":rankok,"B8_TEACHER_IMPROVEMENT_SUPPORTED":impro,
           **milestones,"fullpool_anomaly_terminal":terminal,"candidate_selection_caveat":"Best variant selected descriptively on same 65 target cells; no multiplicity correction or independent confirmation"}
    tc.afc.write_json(tc.ROOT/"TEACHER_CEILING_GATES.json",gates)
    prior=json.loads((tc.ACTIVE/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    tc.afc.write_json(tc.ROOT/"SOURCE_REPRODUCTION.json",{**prior,"b8_source_replay":"fresh 150-checkpoint source extractor completed on 2026-09-27 with empty stderr; exact R4 private selected-epoch payloads verified", "current_b8_and_fullpool_max_ap_replay_error":max(c["prior_replay_max_ap_error"] for c in cells)})
    tc.afc.write_json(tc.ROOT/"LABEL_USAGE_AUDIT.json",{"target_cells":65,"unique_patient_ids":47,"repetitions":1300,
        "exact_B0_identity":True,"FIT_only_hyperparameter_selection":True,"FIT_only_meta_training_and_validation":True,
        "FIT_meta_validation_patients_also_contributed_to_prior_global_lambda_selection":True,
        "deployable_target_support_labels_only_after_acquisition":True,
        "nondeployable_FULL_POOL_and_oracle_balanced_controls_use_additional_target_candidate_labels":True,
        "target_query_labels_only_after_all_20_repetitions_all_variant_scores_frozen":True,
        "target_label_oracle_direction_computed_only_after_score_freeze":True,"outer_predictions_metrics_selection":False,
        "legacy_monolithic_loader_materializes_all_80_labels":True,"strict_no_outer_label_materialization":False})
    (tc.ROOT/"IMPLEMENTATION_AUDIT.md").write_text("# Implementation audit\n\n- New branch and SHA-256 protocol lock were pushed before new target outcomes. Exact A1 150-checkpoint grid and R4 classifier replay were rechecked; B=0 ranking/decision identity was checked for every target cell.\n- Fixed 50/50 candidate/query split and prior 64D UNCERTAINTY support trajectory were reused for all B8 readouts. This isolates readout estimation but does **not** test new acquisition policies.\n- FIT-only standardized R4, FIT-only PCA, and FIT-patient-disjoint projection-optimizer train/validation. Stage-1 global lambda selection uses all FIT patients, including the later meta-validation patients, so that meta-validation is not independent of prior lambda tuning. Lambda/gamma/dimension selection never uses target query labels.\n- All 20 repetitions and all variant query scores were frozen in a target cell before any query label or target oracle was read. Oracle-balanced B8 and FULL_POOL are nondeployable controls.\n- Original current B8 and FULL_POOL AP were replayed against private prior per-repetition records to <1e-8.\n- Patient-ID cluster bootstrap used 10,000 seed-42 draws; 20 repetitions were not treated as independent patients. Variant winner and threshold gates are retrospective on the same 65 target cells, with no family-wise multiplicity control; independent confirmation remains necessary. The 37 FIT selection/model files were hash-frozen before the first target evaluation.\n- Legacy loader materializes all 80 labels. No outer predictions, metrics, or selection were computed, but strict no-outer-label-materialization is false.\n",encoding="utf-8")
    report=["# B=8 Teacher Ceiling Study — development-only", "",
            "65 VLOO target cells, 47 unique patient IDs, 20 fixed-query repetitions each. All values are exploratory matched-query patient-level analyses, not outer-test or prospective results.","",
            f"1. Original B8 replay: {summaries['CURRENT_64D_B8']['mean_ap']:.6f} EZ-AP; previous 0.599632, max private per-repetition AP error {max(c['prior_replay_max_ap_error'] for c in cells):.2g}.",
            f"2. Retuned B8: {summaries['RETUNED_64D_B8']['mean_ap']:.6f}, delta {summaries['RETUNED_64D_B8']['mean_ap']-summaries['CURRENT_64D_B8']['mean_ap']:+.6f} AP; FIT-only regularization selection.",
            f"3. FULL_POOL anomaly: current B8/FULL {diagnostic[0]['b8_ap']:.6f}/{diagnostic[0]['fullpool_ap']:.6f}; retuned {diagnostic[1]['b8_ap']:.6f}/{diagnostic[1]['fullpool_ap']:.6f}, paired delta {diagnostic[1]['delta_b8_minus_fullpool']:+.6f} [{diagnostic[1]['ci_low']:+.6f},{diagnostic[1]['ci_high']:+.6f}]. Terminal `{terminal}`.",
            "4. PCA B8: "+", ".join(f"d{d} {summaries[f'PCA_D{d}_B8']['mean_ap']:.6f}" for d in tc.DIMS)+"; all dimensions and lambda values were FIT-only selected.",
            "5. Episodic meta B8: "+", ".join(f"d{d} {summaries[f'META_D{d}_B8']['mean_ap']:.6f}" for d in tc.DIMS)+". Paired intervals/fold signs in accompanying CSVs.",
            f"6. Highest observed deployable B8 Teacher: `{best}`. This is descriptive winner selection across predeclared variants, not independent validation.",
            f"7. Best Teacher: AP {bestrow['mean_ap']:.6f}, MRR {bestrow['mean_mrr']:.6f}, Top1 {bestrow['mean_top1']:.6f}, Macro-F1 {bestrow['mean_macro_f1']:.6f}, EZ-F1 {bestrow['mean_ez_f1']:.6f}, BA {bestrow['mean_ba']:.6f}; delta AP vs current B8 {bestpair['delta_ap_vs_current_b8']:+.6f} [{bestpair['ci_low']:+.6f},{bestpair['ci_high']:+.6f}], positive folds {bestpair['positive_folds_vs_current_b8']}/5. Its Macro-F1 and EZ-F1 are lower than current B8 ({summaries['CURRENT_64D_B8']['mean_macro_f1']:.6f} and {summaries['CURRENT_64D_B8']['mean_ez_f1']:.6f}), despite slightly higher AP.",
            "8. Milestones: "+", ".join(f"{k}={v}" for k,v in milestones.items())+f"; B8_TEACHER_IMPROVEMENT_SUPPORTED={impro}.",
            f"9. Oracle-direction cosine increases from {next(x['mean_oracle_direction_cosine'] for x in alignment if x['variant']=='CURRENT_64D_B8'):.6f} (current B8) to {next(x['mean_oracle_direction_cosine'] for x in alignment if x['variant']==best):.6f} (best B8), alongside the small AP increase. Retuned FULL_POOL rises to cosine {next(x['mean_oracle_direction_cosine'] for x in alignment if x['variant']=='RETUNED_64D_FULLPOOL'):.6f} and AP {summaries['RETUNED_64D_FULLPOOL']['mean_ap']:.6f}. The broad direction is concordant, but variants are not strictly monotone; cosine was target-label diagnostic only, not a selection input. See the aggregate plot.",
            f"10. B=8 bottleneck: simply reducing dimension or episodically learning a low-dimensional projection did **not** improve the current B8 teacher; the largest B8 gain is only {bestpair['delta_ap_vs_current_b8']:+.6f} AP with a CI crossing zero. Retuned FULL_POOL, using on average {next(x['mean_support_n'] for x in support if x['variant']=='RETUNED_64D_FULLPOOL'):.2f} candidate labels, reaches {summaries['RETUNED_64D_FULLPOOL']['mean_ap']:.6f} AP; B8 support is one-class in {next(x['fraction_one_class_support'] for x in support if x['variant']=='CURRENT_64D_B8'):.2%} of repetitions. This weighs against the claim that the main bottleneck was merely 8 labels fitting a 64D residual, and points to supervision amount/acquisition as plausible constraints. It does not isolate a causal eight-label information ceiling: these retrospective controls differ in both label count and selection mechanism.","",
            "No Student distillation. No new outer-test metrics. The legacy loader still materializes all 80 labels; literal strict sealing is therefore false. The retrospective best-variant comparison is uncorrected for multiple comparisons. Projection-optimizer meta training and validation are patient-disjoint, but the earlier global FIT lambda search included those meta-validation patients."]
    (tc.ROOT/"FINAL_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig,ax=plt.subplots(figsize=(7,5))
        for item in alignment:
            v=item["variant"];x=item["mean_oracle_direction_cosine"];y=summaries[v]["mean_ap"]
            if x is not None and y is not None:
                ax.scatter(x,y,s=36);ax.annotate(v,(x,y),fontsize=6,xytext=(3,2),textcoords="offset points")
        ax.set_xlabel("Mean cosine to post-score target-label oracle direction")
        ax.set_ylabel("Matched-query EZ-AP")
        fig.tight_layout();fig.savefig(tc.ROOT/"TEACHER_AP_VS_ORACLE_COSINE.png",dpi=160);plt.close(fig)
    except ImportError:
        raise RuntimeError("matplotlib required for locked aggregate plot")
    print(f"[FINAL] best={best} AP={bestrow['mean_ap']:.6f} anomaly={terminal} improvement={impro}",flush=True)


if __name__=="__main__":main()
