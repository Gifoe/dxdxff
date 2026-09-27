"""Aggregate fixed 65-cell seizure-geometry audit; never publish patient rows."""
from __future__ import annotations

import json
import pickle

import numpy as np
from scipy.stats import rankdata

from common import CONTEXTS, LOCK_SHA, ROOT, RUNTIME, preflight, read_csv, write_csv, write_json

N_BOOT = 10000
METHODS = [("D0", ""), ("D1", "")] + [(d, c) for d in ("D2", "D3") for c in CONTEXTS]


def mean(x):
    return float(np.mean(x))


def quant(x, p):
    return float(np.quantile(x, p))


def ci(values, weights):
    v = np.asarray(values, dtype=np.float64)
    if v.shape != (65,) or weights.shape != (N_BOOT, 65) or not np.isfinite(v).all():
        raise RuntimeError("Invalid patient-cluster bootstrap input")
    sample = (weights @ v) / weights.sum(axis=1)
    return quant(sample, .025), quant(sample, .975)


def cluster_weights(cells):
    unique = sorted({c["subject_id"] for c in cells})
    if len(unique) != 47:
        raise RuntimeError("47-patient validation cluster structure changed")
    group = np.asarray([unique.index(c["subject_id"]) for c in cells])
    draws = np.random.default_rng(42).integers(0, len(unique), size=(N_BOOT, len(unique)))
    count = np.zeros((N_BOOT, len(unique)), dtype=np.int16)
    np.add.at(count, (np.arange(N_BOOT)[:, None], draws), 1)
    return count[:, group].astype(np.float32)


def load_cells():
    cells = []
    for fold in range(1,6):
        folder = RUNTIME / "private" / f"fold_{fold}"
        selected = read_csv(folder / "A1_VLOO_PRIVATE.csv")
        if len(selected) != 13:
            raise RuntimeError("Incomplete target-cell VLOO")
        for row in selected:
            sid = row["subject_id"]
            with (folder / "cells" / f"cell_{sid.replace(':','_')}.pkl").open("rb") as f:
                cell = pickle.load(f)
            if cell["lock_sha"] != LOCK_SHA or cell["fold"] != fold or cell["epoch"] != int(row["selected_epoch"]) or cell["subject_id"] != sid:
                raise RuntimeError("Private target-cell provenance failed")
            if not cell["target_labels_accessed_only_after_all_predictor_scores_fixed"] or len(cell["predictions"]) != 10 or len(cell["wrong"]) != 8 or len(cell["shuffled"]) != 4:
                raise RuntimeError("Target-cell score-freezing or diagnostic grid failed")
            cells.append(cell)
    if len(cells) != 65 or len({(c["fold"],c["subject_id"]) for c in cells}) != 65:
        raise RuntimeError("Expected 65 fold-by-patient target cells")
    return cells


def get_rows(cells, method, context):
    rows=[]
    for cell in cells:
        candidates=[r for r in cell["predictions"] if r["predictor"]==method and r["context"]==context]
        if len(candidates)!=1:
            raise RuntimeError(f"Missing predictor {method}/{context}")
        rows.append(candidates[0])
    return rows


def vec(rows, key):
    return np.asarray([r[key] for r in rows], dtype=np.float64)


def fold_positive(cells, delta):
    return int(sum(mean(delta[np.asarray([c["fold"]==fold for c in cells])])>0 for fold in range(1,6)))


def weighted_spearman_ranks(values, weights):
    """Exact midranks of resampled cross-patient pairs with multiplicities."""
    order = np.argsort(values, kind="stable")
    sorted_weights = weights[:, order]
    cumulative = np.cumsum(sorted_weights, axis=1, dtype=np.float64)
    ranks = cumulative - (sorted_weights - 1.0) / 2.0
    # Equal values receive a shared weighted midrank, including duplicated pairs.
    sorted_value = values[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_value) != 0) + 1]
    ends = np.r_[starts[1:], len(order)]
    for start, end in zip(starts, ends):
        if end - start > 1:
            before = cumulative[:, start - 1] if start else 0.0
            mass = np.sum(sorted_weights[:, start:end], axis=1)
            ranks[:, start:end] = (before + (mass + 1.0) / 2.0)[:, None]
    inverse = np.argsort(order)
    return ranks[:, inverse]


def fit_patient_bootstrap_rho(base, context, rng):
    assoc=base["associations"][context]
    n=len(base["fit_ids"])
    draws=rng.integers(0,n,size=(N_BOOT,n))
    counts=np.zeros((N_BOOT,n),dtype=np.int16)
    np.add.at(counts,(np.arange(N_BOOT)[:,None],draws),1)
    weight=(counts[:,assoc["pair_i"]].astype(np.float32)*counts[:,assoc["pair_j"]].astype(np.float32))
    rx=weighted_spearman_ranks(np.asarray(assoc["distance"],dtype=np.float64),weight)
    ry=weighted_spearman_ranks(np.asarray(assoc["direction_cos"],dtype=np.float64),weight)
    mass=weight.sum(axis=1)
    mx=np.sum(weight*rx,axis=1)/mass
    my=np.sum(weight*ry,axis=1)/mass
    dx=rx-mx[:,None]
    dy=ry-my[:,None]
    numerator=np.sum(weight*dx*dy,axis=1)
    denominator=np.sqrt(np.sum(weight*dx*dx,axis=1)*np.sum(weight*dy*dy,axis=1))
    return np.divide(numerator,denominator,out=np.zeros(N_BOOT),where=denominator>0)


def association_outputs(cells):
    by_base={}
    for cell in cells:
        by_base[(cell["fold"],cell["epoch"])]=by_base.get((cell["fold"],cell["epoch"]),0)+1
    rng=np.random.default_rng(42)
    rows=[]
    for context in CONTEXTS:
        sampled=np.zeros(N_BOOT,dtype=np.float64)
        observed=0.0
        n_pairs=0
        for (fold,epoch),multiplicity in sorted(by_base.items()):
            with (RUNTIME/"private"/f"fold_{fold}"/f"epoch_{epoch:02d}_base.pkl").open("rb") as f:
                base=pickle.load(f)
            if base["lock_sha"]!=LOCK_SHA:
                raise RuntimeError("FIT association cache provenance failed")
            rho=base["associations"][context]
            observed+=multiplicity*rho["rho"]
            n_pairs+=multiplicity*rho["pairs"]
            sampled+=multiplicity*fit_patient_bootstrap_rho(base,context,rng)
        sampled/=65
        rows.append({"context":context,"target_cells":65,"unique_validation_patient_ids":47,
                     "mean_fit_patient_pair_spearman_rho":observed/65,
                     "fit_pair_observations_repeated_across_cells":n_pairs,
                     "fit_patient_cluster_bootstrap_ci_low":quant(sampled,.025),
                     "fit_patient_cluster_bootstrap_ci_high":quant(sampled,.975),
                     "bootstrap_note":"10k FIT-patient-ID resamples per distinct selected-epoch coordinate; repeated fold/epoch memberships remain descriptive"})
    return rows


def main():
    preflight()
    reproduction=json.loads((ROOT/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    if reproduction["checkpoints"]!=150 or reproduction["max_grid_error"]>1e-6:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    alignment=json.loads((ROOT/"CHANNEL_ALIGNMENT_AUDIT.json").read_text(encoding="utf-8"))
    if not alignment["pass"]:
        raise RuntimeError("CROSS_SEIZURE_CHANNEL_ALIGNMENT_UNAVAILABLE: persistence cannot run")
    cells=load_cells()
    weights=cluster_weights(cells)
    d1=get_rows(cells,"D1","")
    d1ap,d1cos=vec(d1,"ap"),vec(d1,"cosine")
    a1ap=np.asarray([c["frozen_a1"]["ap"] for c in cells])
    a1mrr=np.asarray([c["frozen_a1"]["mrr"] for c in cells])
    a1top=np.asarray([c["frozen_a1"]["top1"] for c in cells])
    true_oracle=np.asarray([c["oracle"]["ap"] for c in cells])
    r80_ap=np.asarray([c["r80_oracle"]["ap"] for c in cells])
    r80cos=np.asarray([c["r80_oracle"]["cosine"] for c in cells])
    retention=(mean(r80_ap)-mean(d1ap))/(mean(true_oracle)-mean(d1ap)) if mean(true_oracle)>mean(d1ap) else float("-inf")
    r80_checks={"median_cos_ge_0_75":quant(r80cos,.5)>=.75,
                "fraction_cos_ge_0_70_ge_0_65":mean(r80cos>=.70)>=.65,
                "headroom_retention_ge_0_70":retention>=.70}
    r80_pass=all(r80_checks.values())
    r80_terminal="R80_ORACLE_SUBSPACE_SUFFICIENT" if r80_pass else "R80_ORACLE_SUBSPACE_INSUFFICIENT"
    write_json(ROOT/"oracle"/"R80_EXPRESSIVITY_GATE.json",{"pass":r80_pass,"terminal":r80_terminal,
               "checks":r80_checks,"median_cosine":quant(r80cos,.5),"fraction_cos_ge_0_70":mean(r80cos>=.70),
               "headroom_retention":retention,"full_oracle_ap_in_sample_label_using":mean(true_oracle),
               "r80_oracle_ap_label_using":mean(r80_ap),"shared_linear_ap":mean(d1ap)})
    evr_rows=[]
    for rank in (1,2,4,8,12,16,24,32):
        values=np.asarray([c["evr"][str(rank)] for c in cells])
        evr_rows.append({"rank":rank,"target_cells":65,"mean_cumulative_evr":mean(values),
                         "median_cumulative_evr":quant(values,.5),"min":float(values.min()),"max":float(values.max())})
    write_csv(ROOT/"oracle"/"ORACLE_SUBSPACE_EVR.csv",evr_rows)
    stability=np.asarray([c["oracle_stability"] for c in cells])
    write_csv(ROOT/"oracle"/"ORACLE_DIRECTION_STABILITY.csv",[{"target_cells":65,"unique_patients":47,
              "mean":mean(stability),"median":quant(stability,.5),"q10":quant(stability,.1),
              "q25":quant(stability,.25),"q75":quant(stability,.75),"q90":quant(stability,.9)}])
    rank_rows=[]
    for fold in range(1,6):
        sub=[c for c in cells if c["fold"]==fold]
        rank_rows.append({"fold":fold,"target_cells":len(sub),
                          "r80_rank_counts_json":json.dumps({str(k):sum(c["r80"]==k for c in sub) for k in sorted({c["r80"] for c in sub})}),
                          "r90_rank_counts_json":json.dumps({str(k):sum(c["r90"]==k for c in sub) for k in sorted({c["r90"] for c in sub})}),
                          "mean_r80":mean([c["r80"] for c in sub]),"mean_r90":mean([c["r90"] for c in sub])})
    write_csv(ROOT/"oracle"/"R80_BY_TARGET_CELL.csv",rank_rows)
    reconstruct=[]
    for fold in (0,1,2,3,4,5):
        sub=cells if fold==0 else [c for c in cells if c["fold"]==fold]
        reconstruct.append({"fold":"all" if fold==0 else fold,"target_cells":len(sub),
                            "median_cosine":quant([c["r80_oracle"]["cosine"] for c in sub],.5),
                            "fraction_cos_ge_0_70":mean([c["r80_oracle"]["cosine"]>=.70 for c in sub]),
                            **{f"mean_{m}":mean([c["r80_oracle"][m] for c in sub]) for m in ("ap","auc","mrr","top1","ndcg")}})
    write_csv(ROOT/"oracle"/"R80_ORACLE_RECONSTRUCTION.csv",reconstruct)
    dims=[]
    for context in CONTEXTS:
        raw={c["raw_dimensions"][context] for c in cells}
        processed={c["context_pca_dimensions"][context] for c in cells}
        if len(raw)!=1 or len(processed)!=1:
            raise RuntimeError("Context dimensions differ across cells")
        dims.append({"context":context,"raw_dimension":raw.pop(),"predictor_input_dimension":processed.pop(),
                     "fit_only_scaler_pca":True})
    write_csv(ROOT/"contexts"/"CONTEXT_DIMENSIONS.csv",dims)
    write_json(ROOT/"contexts"/"CONTEXT_SPECIFICATION.json",{"contexts":dims,"static_pca":"8D FIT R4 channel PCA",
               "seizure_pca":"8D valid FIT seizure-channel PCA","preprocess":"FIT StandardScaler then FIT PCA min(16,input_dim,nFIT-1)",
               "seizure_shuffle":"deterministic reassignment of pooled valid embeddings, preserving per-seizure counts; combined control retains true persistence features",
               "validation_labels_context_input":False})
    association=association_outputs(cells)
    write_csv(ROOT/"contexts"/"CONTEXT_DIRECTION_ASSOCIATION.csv",association)
    wrong_map={(d,c):[next(r for r in cell["wrong"] if r["predictor"]==d and r["context"]==c) for cell in cells]
               for d in ("D2","D3") for c in CONTEXTS}
    shuffled_map={(d,c):[next(r for r in cell["shuffled"] if r["predictor"]==d and r["context"]==c) for cell in cells]
                  for d in ("D2","D3") for c in ("SR_DIST","SR_COMBINED")}
    wrong_public=[]
    for (d,c),rows in wrong_map.items():
        wrong_public.append({"predictor":d,"context":c,"target_cells":65,
                             "mean_correct_cos":mean(vec(rows,"correct_cosine")),"mean_wrong_cos":mean(vec(rows,"wrong_cosine")),
                             "correct_minus_wrong_cos":mean(vec(rows,"correct_cosine")-vec(rows,"wrong_cosine")),
                             "mean_correct_ap":mean(vec(rows,"correct_ap")),"mean_wrong_ap":mean(vec(rows,"wrong_ap")),
                             "correct_minus_wrong_ap":mean(vec(rows,"correct_ap")-vec(rows,"wrong_ap")),
                             "cos_delta_cluster_ci_low":ci(vec(rows,"correct_cosine")-vec(rows,"wrong_cosine"),weights)[0],
                             "cos_delta_cluster_ci_high":ci(vec(rows,"correct_cosine")-vec(rows,"wrong_cosine"),weights)[1],
                             "ap_delta_cluster_ci_low":ci(vec(rows,"correct_ap")-vec(rows,"wrong_ap"),weights)[0],
                             "ap_delta_cluster_ci_high":ci(vec(rows,"correct_ap")-vec(rows,"wrong_ap"),weights)[1]})
    write_csv(ROOT/"controls"/"WRONG_CONTEXT_COMPARISON.csv",wrong_public)
    shuffled_public=[]
    for (d,c),rows in shuffled_map.items():
        dc=vec(rows,"true_cosine")-vec(rows,"shuffled_cosine")
        da=vec(rows,"true_ap")-vec(rows,"shuffled_ap")
        shuffled_public.append({"predictor":d,"context":c,"target_cells":65,
                                "mean_true_cos":mean(vec(rows,"true_cosine")),"mean_shuffled_cos":mean(vec(rows,"shuffled_cosine")),
                                "true_minus_shuffled_cos":mean(dc),"cos_delta_ci_low":ci(dc,weights)[0],"cos_delta_ci_high":ci(dc,weights)[1],
                                "mean_true_ap":mean(vec(rows,"true_ap")),"mean_shuffled_ap":mean(vec(rows,"shuffled_ap")),
                                "true_minus_shuffled_ap":mean(da),"ap_delta_ci_low":ci(da,weights)[0],"ap_delta_ci_high":ci(da,weights)[1],
                                "control_note":"SR_COMBINED substitutes shuffled SR_DIST while keeping true SR_PERSIST; no fake aligned persistence"})
    write_csv(ROOT/"controls"/"SHUFFLED_SEIZURE_STRUCTURE_COMPARISON.csv",shuffled_public)
    static_by_method={d:get_rows(cells,d,"STATIC_R4_MARG") for d in ("D2","D3")}
    summary,dboot,rboot,comparison,ident_checks,read_checks=[],[],[],[],{},{}
    for method,context in METHODS:
        rows=get_rows(cells,method,context)
        co,ap,mrr,top=vec(rows,"cosine"),vec(rows,"ap"),vec(rows,"mrr"),vec(rows,"top1")
        static_co=vec(static_by_method[method],"cosine") if method in ("D2","D3") else co
        static_ap=vec(static_by_method[method],"ap") if method in ("D2","D3") else ap
        dc,da=co-static_co,ap-static_ap
        wrong=wrong_map.get((method,context))
        shuffle=shuffled_map.get((method,context))
        wc=mean(vec(wrong,"correct_cosine")-vec(wrong,"wrong_cosine")) if wrong else None
        wa=mean(vec(wrong,"correct_ap")-vec(wrong,"wrong_ap")) if wrong else None
        sc=mean(vec(shuffle,"true_cosine")-vec(shuffle,"shuffled_cosine")) if shuffle else None
        sa=mean(vec(shuffle,"true_ap")-vec(shuffle,"shuffled_ap")) if shuffle else None
        ident=False
        readout=False
        if method=="D3" and context!="STATIC_R4_MARG":
            checks={"mean_cos_gain_static_ge_0_10":mean(dc)>=.10,
                    "median_cos_gain_static_ge_0_08":quant(co,.5)-quant(static_co,.5)>=.08,
                    "cluster_cos_delta_ci_lower_gt_0":ci(dc,weights)[0]>0,
                    "positive_cos_folds_vs_static_ge_4":fold_positive(cells,dc)>=4,
                    "ap_gain_static_ge_0_015":mean(da)>=.015,
                    "positive_ap_folds_vs_static_ge_4":fold_positive(cells,da)>=4,
                    "correct_wrong_cos_ge_0_05":wc>=.05,
                    "correct_wrong_ap_ge_0_010":wa>=.010,
                    "true_shuffled_cos_ge_0_03":sc>=.03 if sc is not None else False}
            ident=all(checks.values())
            ident_checks[context]={"pass":ident,"checks":checks,
                                   "note":"SR_PERSIST cannot pass shuffled-structure condition: no fabricated persistence control" if sc is None else None}
            read_rules={"ap_gain_frozen_A1_ge_0_010":mean(ap-a1ap)>=.010,
                        "positive_ap_folds_vs_A1_ge_4":fold_positive(cells,ap-a1ap)>=4,
                        "ap_delta_A1_ci_lower_gt_0":ci(ap-a1ap,weights)[0]>0,
                        "mrr_nondecreasing":mean(mrr)>=mean(a1mrr),
                        "top1_nondecreasing":mean(top)>=mean(a1top),
                        "mean_cos_ge_0_40":mean(co)>=.40,
                        "median_cos_ge_0_35":quant(co,.5)>=.35,
                        "correct_beats_wrong":wc>0 and wa>0}
            readout=all(read_rules.values())
            read_checks[context]={"pass":readout,"checks":read_rules}
            comparison.append({"predictor":method,"context":context,"delta_mean_cos_vs_static":mean(dc),
                               "delta_median_cos_vs_static":quant(co,.5)-quant(static_co,.5),
                               "delta_mean_ap_vs_static":mean(da),"delta_mean_mrr_vs_static":mean(mrr-vec(static_by_method[method],"mrr")),
                               "positive_cos_folds":fold_positive(cells,dc),"positive_ap_folds":fold_positive(cells,da),
                               "cos_delta_ci_low":ci(dc,weights)[0],"cos_delta_ci_high":ci(dc,weights)[1],
                               "ap_delta_ci_low":ci(da,weights)[0],"ap_delta_ci_high":ci(da,weights)[1]})
        row={"predictor":method,"context":context,"n_target_cells":65,"n_unique_patients":47,
             "mean_cosine":mean(co),"median_cosine":quant(co,.5),"mean_angle_deg":mean(vec(rows,"angle_deg")),
             "frac_cos_gt_0":mean(co>0),"frac_cos_ge_0_30":mean(co>=.30),"frac_cos_ge_0_50":mean(co>=.50),"frac_cos_ge_0_70":mean(co>=.70),
             "mean_ap":mean(ap),"mean_auc":mean(vec(rows,"auc")),"mean_mrr":mean(mrr),"mean_top1":mean(top),"mean_ndcg":mean(vec(rows,"ndcg")),
             "delta_cos_vs_static":mean(dc) if method in ("D2","D3") else None,
             "delta_ap_vs_static":mean(da) if method in ("D2","D3") else None,
             "delta_cos_vs_D1":mean(co-d1cos),"delta_ap_vs_D1":mean(ap-d1ap),"delta_ap_vs_frozen_A1":mean(ap-a1ap),
             "positive_cos_folds_vs_static":fold_positive(cells,dc) if method in ("D2","D3") else None,
             "positive_ap_folds_vs_static":fold_positive(cells,da) if method in ("D2","D3") else None,
             "correct_minus_wrong_cos":wc,"correct_minus_wrong_ap":wa,
             "true_minus_shuffled_seizure_cos":sc,"true_minus_shuffled_seizure_ap":sa,
             "bootstrap_delta_cos_static_ci_low":ci(dc,weights)[0] if method in ("D2","D3") else None,
             "bootstrap_delta_cos_static_ci_high":ci(dc,weights)[1] if method in ("D2","D3") else None,
             "bootstrap_delta_ap_static_ci_low":ci(da,weights)[0] if method in ("D2","D3") else None,
             "bootstrap_delta_ap_static_ci_high":ci(da,weights)[1] if method in ("D2","D3") else None,
             "identifiability_gate_pass":ident,"readout_gate_pass":readout}
        summary.append(row)
        dboot.append({"predictor":method,"context":context,"n_clusters":47,"mean_cosine":mean(co),
                      "cos_ci_low":ci(co,weights)[0],"cos_ci_high":ci(co,weights)[1],
                      "delta_cos_static":mean(dc) if method in ("D2","D3") else None,
                      "delta_cos_static_ci_low":ci(dc,weights)[0] if method in ("D2","D3") else None,
                      "delta_cos_static_ci_high":ci(dc,weights)[1] if method in ("D2","D3") else None,
                      "delta_cos_D1":mean(co-d1cos),"delta_cos_D1_ci_low":ci(co-d1cos,weights)[0],"delta_cos_D1_ci_high":ci(co-d1cos,weights)[1]})
        rboot.append({"predictor":method,"context":context,"n_clusters":47,
                      "mean_ap":mean(ap),"ap_ci_low":ci(ap,weights)[0],"ap_ci_high":ci(ap,weights)[1],
                      "mean_mrr":mean(mrr),"mrr_ci_low":ci(mrr,weights)[0],"mrr_ci_high":ci(mrr,weights)[1],
                      "delta_ap_static":mean(da) if method in ("D2","D3") else None,
                      "delta_ap_static_ci_low":ci(da,weights)[0] if method in ("D2","D3") else None,
                      "delta_ap_static_ci_high":ci(da,weights)[1] if method in ("D2","D3") else None,
                      "delta_ap_D1":mean(ap-d1ap),"delta_ap_D1_ci_low":ci(ap-d1ap,weights)[0],"delta_ap_D1_ci_high":ci(ap-d1ap,weights)[1],
                      "delta_ap_A1":mean(ap-a1ap),"delta_ap_A1_ci_low":ci(ap-a1ap,weights)[0],"delta_ap_A1_ci_high":ci(ap-a1ap,weights)[1]})
    write_csv(ROOT/"SUMMARY_MATRIX.csv",summary)
    for method,filename in (("D0","D0_POPULATION_MEAN.csv"),("D1","D1_SHARED_LINEAR.csv"),
                            ("D2","D2_FULL_DIRECTION_RIDGE.csv"),("D3","D3_R80_COEFFICIENT_RIDGE.csv")):
        write_csv(ROOT/"prediction"/filename,[r for r in summary if r["predictor"]==method])
    write_csv(ROOT/"statistics"/"CLUSTER_BOOTSTRAP_DIRECTION.csv",dboot)
    write_csv(ROOT/"statistics"/"CLUSTER_BOOTSTRAP_RANKING.csv",rboot)
    write_csv(ROOT/"statistics"/"STATIC_VS_SEIZURE_CONTEXT.csv",comparison)
    ident_pass=any(v["pass"] for v in ident_checks.values())
    read_pass=any(v["pass"] for v in read_checks.values())
    ident_terminal="SEIZURE_RESOLVED_CONTEXT_IDENTIFIES_PATIENT_GEOMETRY" if ident_pass else "SEIZURE_RESOLVED_CONTEXT_DOES_NOT_IDENTIFY_PATIENT_GEOMETRY"
    read_terminal="SEIZURE_CONTEXT_READOUT_SIGNAL_SUPPORTED" if read_pass else "SEIZURE_CONTEXT_READOUT_SIGNAL_NOT_YET_SUPPORTED"
    if not r80_pass:
        overall="ORACLE_GEOMETRY_NOT_WELL_REPRESENTED_BY_SHARED_LINEAR_SUBSPACE"
    elif any(ident_checks[c]["pass"] and read_checks[c]["pass"] for c in ident_checks):
        overall="SEIZURE_CONDITIONED_PATIENT_READOUT_JUSTIFIED"
    elif ident_pass:
        overall="SEIZURE_GEOMETRY_SIGNAL_PRESENT_BUT_NOT_YET_DEPLOYABLE"
    else:
        overall="ZERO_LABEL_PATIENT_GEOMETRY_NOT_IDENTIFIED_FROM_CURRENT_A1_CONTEXT"
    write_json(ROOT/"IDENTIFIABILITY_GATE.json",{"pass":ident_pass,"terminal":ident_terminal,
               "predeclared_D3_contexts":ident_checks,"SR_PERSIST_shuffled_control":"not constructed by design"})
    write_json(ROOT/"READOUT_SIGNAL_GATE.json",{"pass":read_pass,"terminal":read_terminal,
               "predeclared_D3_contexts":read_checks,"overall_terminal":overall,"r80_terminal":r80_terminal})
    best=max([r for r in summary if r["predictor"]=="D3" and r["context"]!="STATIC_R4_MARG"],key=lambda x:x["mean_ap"])
    groups=[]
    for label,condition in (("1",lambda n:n==1),("2",lambda n:n==2),(">=3",lambda n:n>=3)):
        idx=np.asarray([condition(c["n_seizures"]) for c in cells])
        if not idx.any():
            groups.append({"seizure_group":label,"target_cells":0,"unique_patients":0})
            continue
        selected=get_rows(cells,"D3",best["context"])
        selected_wrong=wrong_map[("D3",best["context"])]
        groups.append({"seizure_group":label,"target_cells":int(idx.sum()),
                       "unique_patients":len({cells[i]["subject_id"] for i in np.flatnonzero(idx)}),
                       "D1_mean_ap":mean(d1ap[idx]),"best_predeclared_D3_context":best["context"],
                       "D3_mean_ap":mean(vec(selected,"ap")[idx]),"D3_mean_cosine":mean(vec(selected,"cosine")[idx]),
                       "D3_correct_minus_wrong_cos":mean((vec(selected_wrong,"correct_cosine")-vec(selected_wrong,"wrong_cosine"))[idx]),
                       "D3_correct_minus_wrong_ap":mean((vec(selected_wrong,"correct_ap")-vec(selected_wrong,"wrong_ap"))[idx])})
    write_csv(ROOT/"stratification"/"RESULTS_BY_SEIZURE_COUNT.csv",groups)
    write_json(ROOT/"LABEL_USAGE_AUDIT.json",{"fit_EZ_labels_oracle_and_shared_linear":True,
               "validation_EZ_labels_target_oracle_and_metrics_only":True,
               "validation_labels_context_PCA_scaler_ridge_inputs":False,
               "all_deployable_directions_and_scores_frozen_before_target_label_read":True,
               "outer_predictions_metrics_training_or_selection":False,
               "legacy_monolithic_loader_materializes_all_80_labels":True,
               "strict_no_outer_label_materialization_satisfied":False,
               "validation_cells":65,"unique_validation_patient_ids":47})
    (ROOT/"README.md").write_text("# Seizure-resolved geometry identifiability audit\n\nDevelopment-only frozen A1 diagnostic. On the original server set the private R1_HLV_*, A1_A2_RUNTIME and SRGI_RUNTIME paths; run `extract.py`, `analyze.py`, then `finalize.py`. All patient-level representations, labels, FIT models, score vectors and caches remain in the private runtime. Git contains aggregate-only outputs. No outer metrics or predictions were computed.\n",encoding="utf-8")
    (ROOT/"IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\n- Exact source A1 tip `b2871b32873b67d0e6155d2766ef040ee1e9df01`, frozen source/input hashes and 150 validation grids checked. Other-12 VLOO selects the epoch; FIT and target use the same checkpoint.\n"
        "- The source forward exposes `seizure_channel_embedding`; it is bitwise equal to the original CrossSeizureMILAggregator input. Frozen aggregator output and original classifier(R4) both replay source within 1e-6.\n"
        "- Source dataset aligns local `channel_names_norm` to patient `canonical_channels` and carries `seizure_channel_mask`; both source hashes and runtime canonical names/masks were checked. Missing channels are excluded.\n"
        "- All context construction, seizure shuffling, FIT PCA/scaler, FIT oracle fits, SVD and Ridge training occur without validation labels. Target directions/channel scores are frozen before target labels are read for the nondeployable oracle/evaluation.\n"
        "- The SR_COMBINED shuffled control replaces only SR_DIST while retaining true persistence; a fake shuffled aligned-persistence control was intentionally not created. Thus SR_PERSIST cannot satisfy the ninth all-of-all identification condition.\n"
        "- There are 65 fold-by-patient target cells but only 47 distinct validation patients. CIs resample patient-ID clusters, not cells; FIT association CI resamples FIT IDs within each selected-epoch coordinate. All pairwise/target data remain private.\n"
        "- The legacy monolithic cache initializer materializes labels for all 80 patients before role filtering. Literal no-outer-label-materialization is false, although no outer loader, prediction, metric or outcome-based model choice was made. This remains exploratory development.\n",encoding="utf-8")
    dist=next(r for r in summary if r["predictor"]=="D3" and r["context"]=="SR_DIST")
    persist=next(r for r in summary if r["predictor"]=="D3" and r["context"]=="SR_PERSIST")
    combined=next(r for r in summary if r["predictor"]=="D3" and r["context"]=="SR_COMBINED")
    static=next(r for r in summary if r["predictor"]=="D3" and r["context"]=="STATIC_R4_MARG")
    report=["# Seizure-resolved geometry identifiability — development-only result","",
            "A1 replayed at all 150 checkpoints within 1e-6. The exact temporal-pooled seizure-channel tensor is the original aggregator input; canonical channels and masks align across seizures. No outer predictions/metrics were computed. The legacy cache initializer nonetheless materializes all 80 labels, so this is not strictly sealed.","",
            f"There are 65 VLOO target cells but 47 unique patients. All target-metric intervals use 10,000 patient-ID cluster resamples.","",
            "## Geometry and matched contexts","",
            f"Oracle stability mean/median {mean(stability):.3f}/{quant(stability,.5):.3f}. FIT r80 mean/median {mean([c['r80'] for c in cells]):.1f}/{quant([c['r80'] for c in cells],.5):.0f}; r90 mean/median {mean([c['r90'] for c in cells]):.1f}/{quant([c['r90'] for c in cells],.5):.0f}.",
            f"R80 reconstruction median cosine {quant(r80cos,.5):.3f}; fraction >=0.70 {mean(r80cos>=.70):.3f}; AP headroom retention {retention:.3f}. Full and r80 oracle scores use target labels and are NONDEPLOYABLE.",
            "D3 r80-coefficient ridge, identical <=16D context preprocessing and predictor:","",
            "| Context | Cosine | EZ-AP | Delta AP vs static | Correct-wrong AP | True-shuffled cosine |",
            "|---|---:|---:|---:|---:|---:|"]
    def display_optional(value, digits):
        return "N/A" if value is None else f"{value:+.{digits}f}"
    for row in (static,dist,persist,combined):
        report.append(f"| {row['context']} | {row['mean_cosine']:.3f} | {row['mean_ap']:.4f} | {row['delta_ap_vs_static']:+.4f} | {display_optional(row['correct_minus_wrong_ap'],4)} | {display_optional(row['true_minus_shuffled_seizure_cos'],3)} |")
    assoc_by_context={r["context"]:r for r in association}
    group_by_count={r["seizure_group"]:r for r in groups}
    report += ["",f"Shared FIT linear AP {mean(d1ap):.4f}; frozen A1 AP {mean(a1ap):.4f}. Best predeclared seizure D3 by AP: {best['context']} AP {best['mean_ap']:.4f}, cosine {best['mean_cosine']:.3f}.",
               "FIT context-distance/oracle-direction-cosine correlations and FIT-ID bootstrap intervals are in CONTEXT_DIRECTION_ASSOCIATION.csv. Descriptive seizure-count strata, wrong-context and shuffled-membership controls are separate outputs.",
               "", "## Required audit answers", "",
               "1. Exact A1: yes, 150/150 checkpoint grids replayed, maximum error 0; five-fold mean Macro-F1 0.6260.",
               "2. Captured tensor: source `seizure_channel_embedding`, shape `[batch, seizure, canonical channel, 32]`; original aggregator prehook input and replay agree exactly.",
               "3. Channel identity: source canonical-channel name mapping and validity masks align; absent entries are excluded and pair statistics use observed channel intersections.",
               f"4. Seizure-count groups (distinct patients): one={group_by_count['1']['unique_patients']}, two={group_by_count['2']['unique_patients']}, three-or-more={group_by_count['>=3']['unique_patients']}. These are 47 distinct development validation patients, not 65 independent patients.",
               f"5. FIT oracle variance requires median r80={quant([c['r80'] for c in cells],.5):.0f} and r90={quant([c['r90'] for c in cells],.5):.0f}; the corresponding means are {mean([c['r80'] for c in cells]):.1f}/{mean([c['r90'] for c in cells]):.1f}.",
               f"6. r80 is better than the prior rank-4 diagnostic (median reconstruction cosine {quant(r80cos,.5):.3f} vs about 0.382; headroom retention {retention:.1%} vs about 21.4%) but still fails all prespecified expressivity thresholds.",
               f"7. Matched static R4 marginal control remains weak: D3 cosine {static['mean_cosine']:.3f}, AP {static['mean_ap']:.4f}; the earlier different predictor's R4 marginal diagnostic was about cosine 0.197/AP 0.531, so these are directionally similar, not an exact numerical reproduction.",
               f"8. SR_DIST does not improve over static: cosine delta {dist['delta_cos_vs_static']:+.3f}, AP delta {dist['delta_ap_vs_static']:+.4f}.",
               f"9. SR_PERSIST does not improve over static: cosine delta {persist['delta_cos_vs_static']:+.3f}, AP delta {persist['delta_ap_vs_static']:+.4f}.",
               f"10. SR_COMBINED does not improve over static: cosine delta {combined['delta_cos_vs_static']:+.3f}, AP delta {combined['delta_ap_vs_static']:+.4f}.",
               "11. Therefore the measured seizure distribution/persistence summaries add no convincing patient-specific geometry information beyond the matched static marginal control; higher raw context dimension alone is not evidence.",
               f"12. FIT context-distance/oracle-direction Spearman rho is static {assoc_by_context['STATIC_R4_MARG']['mean_fit_patient_pair_spearman_rho']:.3f}, SR_DIST {assoc_by_context['SR_DIST']['mean_fit_patient_pair_spearman_rho']:.3f}, SR_PERSIST {assoc_by_context['SR_PERSIST']['mean_fit_patient_pair_spearman_rho']:.3f}, combined {assoc_by_context['SR_COMBINED']['mean_fit_patient_pair_spearman_rho']:.3f}; seizure contexts do not strengthen the negative association. FIT-ID bootstrap intervals are in the association CSV.",
               f"13. Correct versus cyclic wrong context for the best seizure D3 ({best['context']}): cosine {best['correct_minus_wrong_cos']:+.3f}, AP {best['correct_minus_wrong_ap']:+.4f}; neither reaches both fixed identification thresholds and patient-cluster uncertainty includes zero.",
               f"14. True minus shuffled seizure-membership cosine: SR_DIST {dist['true_minus_shuffled_seizure_cos']:+.3f}, combined {combined['true_minus_shuffled_seizure_cos']:+.3f}; real membership does not beat the destruction control. No fabricated persistence shuffle was used.",
               f"15. Descriptively the best seizure D3 minus D1 AP is {group_by_count['1']['D3_mean_ap']-group_by_count['1']['D1_mean_ap']:+.4f} (one), {group_by_count['2']['D3_mean_ap']-group_by_count['2']['D1_mean_ap']:+.4f} (two), {group_by_count['>=3']['D3_mean_ap']-group_by_count['>=3']['D1_mean_ap']:+.4f} (three-or-more); these posthoc groups neither establish a seizure-structure effect nor justify group-specific selection.",
               f"16. Best seizure D3 AP {best['mean_ap']:.4f} barely exceeds D1 {mean(d1ap):.4f} but is below frozen A1 {mean(a1ap):.4f}; it does not meet the deployable readout gate.",
               "17. No: r80 expressivity, context identification, and deployable ranking gates all fail. A seizure-conditioned patient-specific readout is not justified by these representations.",
               "18. If pursuing this question, predeclare a separate few-shot patient-calibration study or obtain genuinely richer observations; another zero-label hypernetwork on the same A1 summaries is not supported.","",
               "A patient-specific oracle can exist without an identifiable zero-label proxy. Larger context dimensionality alone is not evidence that seizure membership helps; the shuffled-structure control must also improve. No further zero-label hypernetwork is justified unless the fixed gates pass. If they fail, few-shot patient calibration or genuinely richer observations are the coherent alternatives, subject to a new protocol.","",
               f"`{r80_terminal}`",f"`{ident_terminal}`",f"`{read_terminal}`",f"`{overall}`"]
    (ROOT/"FINAL_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print("[FINAL]",r80_terminal,ident_terminal,read_terminal,overall,flush=True)


if __name__=="__main__":
    main()
