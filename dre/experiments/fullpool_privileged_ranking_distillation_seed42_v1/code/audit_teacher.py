"""Aggregate-only FIT OOF Teacher audit and prespecified Gate 0."""
from __future__ import annotations

import json
import pickle

import numpy as np

import protocol as p
from teacher import grid_path,oof_path

METRICS=("ap","auc","mrr","top1","macro_f1","ez_f1","ba")


def avg(values):
    a=np.asarray(values,dtype=float);a=a[np.isfinite(a)]
    return float(a.mean()) if len(a) else None


def patient_row(ctx,sid):
    with oof_path(ctx,sid).open("rb") as f:row=pickle.load(f)
    if (row["lock_sha"]!=p.LOCK_SHA or row["sid"]!=sid or
            row["context_id"]!=ctx["context_id"] or not row["no_scored_channel_label_in_teacher_fit"] or
            not row["no_scored_patient_label_in_lambda_selection"] or sum(row["heldout_fold_sizes"])!=row["n_channels"]):
        raise RuntimeError("Teacher OOF provenance/crossfit invalid")
    base=p.afc.query_metrics(row["y"],row["a1_score"])
    teacher=p.afc.query_metrics(row["y"],row["teacher_score"])
    return dict(base=base,teacher=teacher,n_channels=row["n_channels"],lambda_index=row["lambda_index"])


def run():
    p.preflight()
    contexts=p.all_contexts()
    detail=[];grids=0;crossfits=0
    for ctx in contexts:
        fit_ids=sorted(p.payload(ctx["fold"],ctx["epoch"])["fit"])
        patients=[]
        for sid in fit_ids:
            if not grid_path(ctx,sid).is_file():raise RuntimeError("FIT lambda grid incomplete")
            grids+=1;patients.append(patient_row(ctx,sid));crossfits+=5
        detail.append(dict(fold=ctx["fold"],context_id=ctx["context_id"],
                           source_epoch=ctx["epoch"],target_cell_weight=len(ctx["target_ids"]),
                           n_fit_patients=len(patients),patients=patients))
    if grids!=sum(x["n_fit_patients"] for x in detail) or len(detail)!=17 or crossfits<4000:
        raise RuntimeError("Teacher grid/OOF coverage incomplete")
    context_rows=[];gain_rows=[]
    for block in detail:
        patients=block["patients"]
        base={metric:avg([x["base"][metric] for x in patients]) for metric in METRICS}
        teacher={metric:avg([x["teacher"][metric] for x in patients]) for metric in METRICS}
        delta=np.asarray([x["teacher"]["ap"]-x["base"]["ap"] for x in patients],dtype=float)
        common=dict(fold=block["fold"],context_id=block["context_id"],
                    source_epoch=block["source_epoch"],target_cell_weight=block["target_cell_weight"],
                    n_fit_patients=block["n_fit_patients"])
        context_rows.append(dict(**common,**{f"a1_{m}":base[m] for m in METRICS},
                                 **{f"teacher_{m}":teacher[m] for m in METRICS},
                                 **{f"delta_{m}":teacher[m]-base[m] for m in METRICS}))
        finite=delta[np.isfinite(delta)]
        gain_rows.append(dict(**common,n_estimable=len(finite),fraction_teacher_ap_better=float(np.mean(finite>0)),
                              mean_patient_delta_ap=float(finite.mean()),median_patient_delta_ap=float(np.median(finite)),
                              p25_patient_delta_ap=float(np.quantile(finite,.25)),
                              p75_patient_delta_ap=float(np.quantile(finite,.75))))
    folds=[]
    for fold in range(1,6):
        subset=[r for r in context_rows if r["fold"]==fold]
        weights=np.asarray([r["target_cell_weight"] for r in subset],dtype=float)
        if int(weights.sum())!=13:raise RuntimeError("Context weights do not represent 13 target cells")
        folds.append(dict(fold=fold,n_contexts=len(subset),n_target_cells=13,
                          **{f"{prefix}_{m}":float(np.average([r[prefix+"_"+m] for r in subset],weights=weights))
                             for prefix in ("a1","teacher","delta") for m in METRICS}))
    overall=dict(fold="ALL",n_contexts=17,n_target_cells=65,
                 **{f"{prefix}_{m}":avg([r[prefix+"_"+m] for r in folds])
                    for prefix in ("a1","teacher","delta") for m in METRICS})
    p.afc.write_csv(p.ROOT/"FIT_OOF_TEACHER_METRICS.csv",context_rows+folds+[overall])
    p.afc.write_csv(p.ROOT/"FIT_OOF_TEACHER_PATIENT_GAINS.csv",gain_rows)
    positive=sum(r["delta_ap"]>0 for r in folds)
    signal=bool(overall["delta_ap"]>.03 and positive>=4 and overall["teacher_ap"]>overall["a1_ap"] and
                overall["delta_mrr"]>-.005 and overall["delta_top1"]>-.005)
    audit=dict(lock_sha=p.LOCK_SHA,stage="FIT_ONLY_TEACHER_OOF",n_source_contexts=17,
               n_fit_patient_contexts=grids,n_channel_crossfit_heads=crossfits,
               channel_folds_per_patient=5,lambda_selected_from_other_fit_patients=True,
               own_label_excluded_from_teacher_fit_and_lambda_selection=True,
               fit_oof_a1_ap=overall["a1_ap"],fit_oof_teacher_ap=overall["teacher_ap"],
               mean_delta_ap=overall["delta_ap"],mean_delta_mrr=overall["delta_mrr"],
               mean_delta_top1=overall["delta_top1"],positive_outer_folds=positive,
               PRIVILEGED_TEACHER_SIGNAL_VALID=signal,
               target_patient_outcomes_not_accessed_by_teacher_audit=True)
    p.afc.write_json(p.ROOT/"TEACHER_CROSSFIT_AUDIT.json",audit)
    source=p.PROJECT/"zeroshot_geometry_canonicalization_metadg_seed42_v1"
    (p.ROOT/"SOURCE_REPRODUCTION.json").write_bytes((source/"SOURCE_REPRODUCTION.json").read_bytes())
    (p.ROOT/"B0_IDENTITY_AUDIT.json").write_bytes((source/"B0_IDENTITY_AUDIT.json").read_bytes())
    ref=p.PROJECT/"b8_teacher_ceiling_meta_readout_seed42_v1"
    prior=p.afc.read_csv(ref/"TEACHER_VARIANT_MATRIX.csv")
    item=next(r for r in prior if r["variant"]=="RETUNED_64D_FULLPOOL")
    p.afc.write_json(p.ROOT/"FULLPOOL_REFERENCE_REPRODUCTION.json",
                     dict(source="existing exact B8 teacher-ceiling matched fixed-query replay",
                          variant=item["variant"],mean_ap=float(item["mean_ap"]),
                          n_cells=int(item["n_cells"]),n_repetitions=int(item["n_repetitions"]),
                          absolute_error_vs_protocol=abs(float(item["mean_ap"])-p.FULLPOOL_AP),
                          pass_reference=abs(float(item["mean_ap"])-p.FULLPOOL_AP)<1e-6,
                          nondeployable=True))
    print(f"[TEACHER_AUDIT] AP_A1={overall['a1_ap']:.6f} AP_T={overall['teacher_ap']:.6f} delta={overall['delta_ap']:+.6f} folds={positive}/5 gate={signal}",flush=True)


if __name__=="__main__":run()
