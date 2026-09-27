"""Aggregate-only finalization; patient/channel records remain private."""
from __future__ import annotations

import hashlib
import json
import math
import pickle
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

import evaluate as ev
from train import LOCK_SHA, PROJECT, ROOT, RUNTIME, VARIANTS, preflight, selected_config

afc=ev.afc
afr=ev.afr
ALL=("Z0_A1",)+VARIANTS
N_BOOT=10000
METRICS=("ap","auc","mrr","top1","macro_f1","ez_f1","ba")


def finite_mean(values):
    x=np.asarray(values,dtype=float);x=x[np.isfinite(x)]
    return float(x.mean()) if len(x) else None


def _stem(sid): return hashlib.sha256(sid.encode()).hexdigest()[:16]


def new_cell(fold,sid,variant):
    path=RUNTIME/"private"/"cells"/variant/f"fold_{fold}"/(_stem(sid)+".pkl")
    with path.open("rb") as f: row=pickle.load(f)
    if row["lock_sha"]!=LOCK_SHA or row["fold"]!=fold or row["sid"]!=sid or len(row["metrics"])!=20:
        raise RuntimeError("New target cell incomplete")
    return row


def baseline_cell(fold,source):
    sid=source["subject_id"]; epoch=int(source["selected_epoch"]);tau=float(source["selected_threshold"])
    payload=ev._source_payload(fold,epoch)["val"][sid]
    score=afr.margin_for(payload,tau)
    afr.verify_b0(payload,tau,score)
    y=np.asarray(payload["y"],dtype=np.int8)
    records=[]
    for rep in range(20):
        candidate,query=afc.split_indices(len(y),42,fold,sid,rep)
        if set(candidate)&set(query):raise RuntimeError("B0 fixed split overlap")
        records.append(dict(rep=rep,**afc.query_metrics(y[query],score[query])))
    return dict(lock_sha=LOCK_SHA,fold=fold,sid=sid,variant="Z0_A1",epoch=epoch,
                threshold=tau,metrics=records,R4=np.asarray(payload["R4"],dtype=np.float64),y=y)


def load():
    freeze_path=RUNTIME/"SCORE_FREEZE_BEFORE_TARGET_LABELS.json"
    if not freeze_path.is_file():
        raise RuntimeError("Model scores were not hash-frozen before target evaluation")
    frozen=json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen["source_common_files"]!=5 or frozen["diagnostic_amendment_sha"]!="765107a265c48d139430c97d95da2aeb513dbadff6a639984ff2b04afa69e0bf":
        raise RuntimeError("Common-checkpoint diagnostic provenance incomplete")
    cells=[];rows=[];representations={}
    for fold in range(1,6):
        common_path=RUNTIME/"private"/"a1_common_checkpoint"/f"fold_{fold}_epoch30.pkl"
        if afc.sha(common_path)!=frozen["source_common"][str(common_path.relative_to(RUNTIME))]:
            raise RuntimeError("A1 shared-checkpoint R4 changed after freeze")
        with common_path.open("rb") as f:baseline_common=pickle.load(f)
        source=afc.read_csv(afc.PRIOR_RUNTIME/"private"/f"fold_{fold}"/"A1_VLOO_PRIVATE.csv")
        if len(source)!=13:raise RuntimeError("Source A1 VLOO grid incomplete")
        for src in source:
            sid=src["subject_id"]
            base=baseline_cell(fold,src)
            current={"Z0_A1":base}
            for variant in VARIANTS: current[variant]=new_cell(fold,sid,variant)
            cells.append((fold,sid,current))
            for variant,cell in current.items():
                for m in cell["metrics"]:
                    rows.append(dict(fold=fold,sid=sid,rep=m["rep"],variant=variant,
                                     **{k:m.get(k,np.nan) for k in METRICS}))
                y=ev.reference_y(fold,30,sid)
                if variant=="Z0_A1":
                    representations[(fold,sid,variant)]=(np.asarray(baseline_common[sid]["R4"],dtype=float),y,30)
                else:
                    path=RUNTIME/"private"/"pending"/variant/f"fold_{fold}"/(_stem(sid)+".pkl")
                    if afc.sha(path)!=cell["pending_sha"]:raise RuntimeError("Target score/R4 pending hash mismatch")
                    common=ev.private_snapshot(fold,variant,30)[sid]
                    representations[(fold,sid,variant)]=(np.asarray(common["R4"],dtype=float),y,30)
    if len(cells)!=65 or len({sid for _,sid,_ in cells})!=47 or len(rows)!=65*20*len(ALL):
        raise RuntimeError("65/47/20 target coverage changed")
    exact={(r["fold"],r["sid"],r["rep"],r["variant"]):r for r in rows}
    if len(exact)!=len(rows):raise RuntimeError("Duplicate target metric record")
    return cells,rows,exact,representations


def bootstrap(values,rows,ids,counts):
    index={sid:i for i,sid in enumerate(ids)}
    numer=np.zeros(len(ids));denom=np.zeros(len(ids))
    for v,r in zip(values,rows):
        if np.isfinite(v):
            i=index[r["sid"]]; numer[i]+=v;denom[i]+=1
    if denom.sum()<1:return dict(mean=None,low=None,high=None,n=0)
    dn=counts@denom
    draws=np.divide(counts@numer,dn,out=np.full(len(counts),np.nan),where=dn>0)
    good=draws[np.isfinite(draws)]
    if len(good)<.99*len(counts):raise RuntimeError("Degenerate patient bootstrap")
    return dict(mean=float(numer.sum()/denom.sum()),low=float(np.quantile(good,.025)),
                high=float(np.quantile(good,.975)),n=int(denom.sum()))


def outcome_tables(rows,exact):
    ids=sorted({r["sid"] for r in rows})
    rng=np.random.default_rng(42)
    draws=rng.integers(0,len(ids),size=(N_BOOT,len(ids)))
    counts=np.zeros((N_BOOT,len(ids)),dtype=np.int16)
    np.add.at(counts,(np.arange(N_BOOT)[:,None],draws),1)
    counts=counts.astype(np.float32)
    matrix=[];boot_rows=[];fold_rows=[];summary={}
    for variant in ALL:
        part=[r for r in rows if r["variant"]==variant]
        if len(part)!=1300:raise RuntimeError(f"Missing 20 fixed reps for {variant}")
        diffs=np.asarray([r["ap"]-exact[(r["fold"],r["sid"],r["rep"],"Z0_A1")]["ap"] for r in part],dtype=float)
        stat=bootstrap(diffs,part,ids,counts)
        row=dict(variant=variant,n_cells=65,n_unique_patient_ids=47,n_repetitions=1300,
                 n_estimable_ap=sum(np.isfinite(r["ap"]) for r in part),
                 **{k:finite_mean([r[k] for r in part]) for k in METRICS},
                 delta_ap_vs_a1=stat["mean"],delta_ap_ci_low=stat["low"],delta_ap_ci_high=stat["high"])
        matrix.append(row);summary[variant]=row
        boot_rows.append(dict(variant=variant,statistic="paired_delta_ap_vs_a1",mean=stat["mean"],
                              ci_low=stat["low"],ci_high=stat["high"],n_estimable=stat["n"],
                              n_patient_clusters=47,resamples=N_BOOT,seed=42))
        for fold in range(1,6):
            sub=[r for r in part if r["fold"]==fold]
            delta=finite_mean([r["ap"]-exact[(fold,r["sid"],r["rep"],"Z0_A1")]["ap"] for r in sub])
            fold_rows.append(dict(variant=variant,fold=fold,n_cells=13,n_repetitions=260,
                                  ap=finite_mean([r["ap"] for r in sub]),delta_ap_vs_a1=delta))
    afc.write_csv(ROOT/"ZEROSHOT_VARIANT_MATRIX.csv",matrix)
    afc.write_csv(ROOT/"PATIENT_CLUSTER_BOOTSTRAP.csv",boot_rows)
    afc.write_csv(ROOT/"FOLD_CONSISTENCY.csv",fold_rows)
    return summary,fold_rows


def fit_selection_table():
    rows=[]
    for fold in range(1,6):
        for variant in VARIANTS:
            sel=selected_config(fold,variant)
            for index in range(sel["candidates"]):
                path=RUNTIME/"fit"/f"fold_{fold}"/variant/f"config_{index:02d}"/"summary.json"
                item=json.loads(path.read_text(encoding="utf-8"))
                if item["lock_sha"]!=LOCK_SHA:raise RuntimeError("FIT selection provenance changed")
                rows.append(dict(fold=fold,variant=variant,config_index=index,
                                 config=json.dumps(item["config"],sort_keys=True),
                                 fit_meta_val_ap=item["best_ap"],fit_meta_val_mrr=item["best_mrr"],
                                 fit_meta_val_top1=item["best_top1"],best_epoch=item["best_epoch"],
                                 selected=(index==sel["config_index"])))
    afc.write_csv(ROOT/"FIT_HYPERPARAM_SELECTION.csv",rows)


def unit_direction(x,y):
    x=np.asarray(x,dtype=float);y=np.asarray(y,dtype=np.int8)
    if len(x)!=len(y) or set(np.unique(y))!={0,1}:return None
    d=x[y==1].mean(0)-x[y==0].mean(0)
    norm=np.linalg.norm(d)
    return d/norm if norm>1e-10 else None


def fit_geometry(fold,variant,epoch):
    """Model R4 on FIT subjects only; no target labels in this extraction."""
    import torch
    import train as tr
    from exp_ez_hybrid import _move_tensors_to_device
    from run_matched import install_interleaved_hlv_view,make_args
    import exp_ez_hybrid as core
    if not getattr(fit_geometry,"_installed",False):
        install_interleaved_hlv_view();fit_geometry._installed=True
    args=make_args("R0",RUNTIME/"diagnostic_scratch")
    exp=core.Exp_EZHybridLocalization(args)
    split=next(s for s in exp.outer_splits if int(s["fold_idx"])==fold)
    fit_set,_,_,_=exp._build_datasets(list(split["fit_subjects"]),list(split["validation_subjects"]),[])
    loader=exp._make_loader(fit_set,shuffle=False,batch_size=2)
    model=exp.runtime["model_cls"](args).to(exp.device)
    exp._dry_initialize_lazy_layers(model,loader)
    if variant=="Z0_A1":
        ckpt=tr.A1_RUNTIME/"A1"/f"fold_{fold}"/f"epoch_{epoch:02d}.pt"
    else:
        index=selected_config(fold,variant)["config_index"]
        ckpt=RUNTIME/"full"/f"fold_{fold}"/variant/f"config_{index:02d}"/f"epoch_{epoch:02d}.pt"
    state=torch.load(ckpt,map_location=exp.device,weights_only=False)
    model.load_state_dict(state["model_state_dict"] if variant=="Z0_A1" else state["model"],strict=True)
    cap=tr.CaptureR4(model);model.eval();out=[]
    with torch.no_grad():
        for raw in loader:
            batch=_move_tensors_to_device(raw,exp.device)
            model(batch)
            r4=cap.h.detach().cpu().numpy()
            mask=batch["channel_mask"].detach().cpu().numpy().astype(bool)
            y=batch["labels_ez"].detach().cpu().numpy()
            for i,sid in enumerate(batch["subject_id"]):
                out.append((sid,r4[i][mask[i]].astype(float),y[i][mask[i]].astype(np.int8)))
    cap.close()
    if len(out)!=len(split["fit_subjects"]):raise RuntimeError("FIT R4 extraction incomplete")
    return out


def diagnostics(cells,reps):
    dispersion=[];headroom=[];reversal=[]
    byvariant=defaultdict(list)
    for fold,sid,_ in cells:
        for variant in ALL:
            x,y,epoch=reps[(fold,sid,variant)]
            byvariant[variant].append((fold,sid,epoch,x,y))
    fit_cache={}
    for variant,items in byvariant.items():
        directions=[]
        for fold,sid,epoch,x,y in items:
            d=unit_direction(x,y)
            if d is not None:directions.append((fold,sid,d))
            key=(fold,variant,epoch)
            if key not in fit_cache: fit_cache[key]=fit_geometry(*key)
            fit=fit_cache[key]
            fdirs=[unit_direction(z,labels) for _,z,labels in fit]
            fdirs=[a for a in fdirs if a is not None]
            shared_direction=unit_direction(np.asarray([v for d0 in fdirs for v in (d0,-d0)]),
                                            np.asarray([v for _ in fdirs for v in (1,0)])) if fdirs else None
            if shared_direction is None and fdirs:
                v=np.mean(fdirs,axis=0);shared_direction=v/max(np.linalg.norm(v),1e-12)
            auc=roc_auc_score(y,x@shared_direction) if shared_direction is not None and set(np.unique(y))=={0,1} else np.nan
            reversal.append(dict(variant=variant,fold=fold,transfer_auroc=float(auc),
                                 reversed=int(auc<.5) if np.isfinite(auc) else None))
            # Post-score diagnostic heads. A shared FIT-trained head and an
            # out-of-fold patient-specific head both use target labels only here.
            fit_x=np.concatenate([z for _,z,_ in fit]);fit_y=np.concatenate([lab for _,_,lab in fit])
            weights=np.concatenate([np.full(len(lab),1/len(lab)) for _,_,lab in fit])
            scaler=StandardScaler().fit(fit_x);sx=scaler.transform(fit_x);tx=scaler.transform(x)
            common=LogisticRegression(C=1.,max_iter=1000,solver="lbfgs",random_state=42)
            common.fit(sx,fit_y,sample_weight=weights)
            shared_ap=float(average_precision_score(y,common.predict_proba(tx)[:,1]))
            k=min(5,int(y.sum()),int((1-y).sum()))
            specific_ap=np.nan
            if k>=2:
                pred=np.zeros(len(y),dtype=float)
                cv=StratifiedKFold(n_splits=k,shuffle=True,random_state=42)
                for a,b in cv.split(tx,y):
                    own=LogisticRegression(C=1.,max_iter=1000,solver="lbfgs",random_state=42)
                    own.fit(tx[a],y[a]);pred[b]=own.predict_proba(tx[b])[:,1]
                specific_ap=float(average_precision_score(y,pred))
            headroom.append(dict(variant=variant,fold=fold,shared_ap=shared_ap,
                                 patient_specific_oof_ap=specific_ap,gap=specific_ap-shared_ap))
        cos=[float(a@b) for i,(fa,sa,a) in enumerate(directions)
             for fb,sb,b in directions[i+1:] if fa==fb and sa!=sb]
        if not cos:raise RuntimeError("No cross-patient R4 direction pairs")
        evrs=[]
        for fold in range(1,6):
            dmat=np.stack([d for f,_,d in directions if f==fold])
            singular=np.linalg.svd(dmat-dmat.mean(0),compute_uv=False)
            evrs.append(float(singular[0]**2/max(float(np.square(singular).sum()),1e-12)))
        evr=float(np.mean(evrs))
        dispersion.append(dict(variant=variant,n_directions=len(directions),n_cross_patient_pairs=len(cos),
                               mean_pairwise_cosine=float(np.mean(cos)),median_pairwise_cosine=float(np.median(cos)),
                               q10_cosine=float(np.quantile(cos,.1)),fraction_cosine_negative=float(np.mean(np.asarray(cos)<0)),
                               leading_direction_evr=evr))
    # Published diagnostics are variant/fold aggregate only, never patient rows.
    hr=[dict(variant=v,shared_ap=finite_mean([r["shared_ap"] for r in headroom if r["variant"]==v]),
             patient_specific_oof_ap=finite_mean([r["patient_specific_oof_ap"] for r in headroom if r["variant"]==v]),
             gap=finite_mean([r["gap"] for r in headroom if r["variant"]==v])) for v in ALL]
    rv=[dict(variant=v,mean_transfer_auroc=finite_mean([r["transfer_auroc"] for r in reversal if r["variant"]==v]),
             reversal_rate=finite_mean([r["reversed"] for r in reversal if r["variant"]==v]),
             n_cells=sum(r["variant"]==v for r in reversal)) for v in ALL]
    afc.write_csv(ROOT/"PATIENT_DIRECTION_DISPERSION.csv",dispersion)
    afc.write_csv(ROOT/"SHARED_VS_PATIENT_SPECIFIC_HEADROOM.csv",hr)
    afc.write_csv(ROOT/"DIRECTION_REVERSAL_AUDIT.csv",rv)
    return {r["variant"]:r for r in dispersion},{r["variant"]:r for r in hr},{r["variant"]:r for r in rv}


def decisions(summary,folds,disp,head,rev):
    base=summary["Z0_A1"]
    foldmap={(r["variant"],r["fold"]):r for r in folds}
    gates={};mechanisms={}
    for variant in VARIANTS:
        r=summary[variant]
        pos=sum(foldmap[(variant,f)]["delta_ap_vs_a1"]>0 for f in range(1,6))
        rank_ok=r["mrr"]>=base["mrr"]-.005 and r["top1"]>=base["top1"]-.005
        ci_ok=r["delta_ap_ci_low"]>0
        significant=ci_ok and pos>=4 and rank_ok
        geometry=(significant and head[variant]["gap"]<head["Z0_A1"]["gap"] and
                  disp[variant]["mean_pairwise_cosine"]>disp["Z0_A1"]["mean_pairwise_cosine"] and
                  rev[variant]["reversal_rate"]<rev["Z0_A1"]["reversal_rate"])
        gates[variant]=dict(ap=r["ap"],delta_ap=r["delta_ap_vs_a1"],ci_low=r["delta_ap_ci_low"],
                            ci_high=r["delta_ap_ci_high"],positive_folds=pos,
                            mrr_delta=r["mrr"]-base["mrr"],top1_delta=r["top1"]-base["top1"],
                            ZEROSHOT_IMPROVEMENT_SUPPORTED=bool(significant and r["delta_ap_vs_a1"]>=.015),
                            ZERO_SHOT_REACHES_CURRENT_B8=bool(significant and r["delta_ap_vs_a1"]>=.015 and r["ap"]>=.5996322681940147),
                            ZERO_SHOT_REACHES_BEST_B8=bool(significant and r["ap"]>=.6052912572363056),
                            ZERO_SHOT_AP_062_REACHED=bool(r["ap"]>=.62),
                            PATIENT_GEOMETRY_CANONICALIZATION_SUPPORTED=bool(geometry))
        mechanisms[variant]=dict(gap_delta=head[variant]["gap"]-head["Z0_A1"]["gap"],
                                 cosine_delta=disp[variant]["mean_pairwise_cosine"]-disp["Z0_A1"]["mean_pairwise_cosine"],
                                 reversal_rate_delta=rev[variant]["reversal_rate"]-rev["Z0_A1"]["reversal_rate"],
                                 supported=bool(geometry))
    best=max(VARIANTS,key=lambda v:summary[v]["ap"])
    bg=gates[best]
    if bg["ZERO_SHOT_REACHES_BEST_B8"]:terminal="ZEROSHOT_PATIENT_GEOMETRY_RECOVERY_JUSTIFIED"
    elif bg["ZERO_SHOT_REACHES_CURRENT_B8"]:terminal="ZEROSHOT_APPROACHES_FEWSHOT_PERFORMANCE"
    elif gates["Z3_PATIENT_HELDOUT_MLDG"]["ZEROSHOT_IMPROVEMENT_SUPPORTED"] and not any(gates[v]["ZEROSHOT_IMPROVEMENT_SUPPORTED"] for v in VARIANTS[:2]):
        terminal="PATIENT_HELDOUT_META_GENERALIZATION_SUPPORTED"
    elif bg["ZEROSHOT_IMPROVEMENT_SUPPORTED"] and not bg["PATIENT_GEOMETRY_CANONICALIZATION_SUPPORTED"]:
        terminal="ZEROSHOT_GAIN_WITHOUT_GEOMETRY_CANONICALIZATION"
    elif bg["ZEROSHOT_IMPROVEMENT_SUPPORTED"]:
        terminal="ZEROSHOT_GEOMETRY_GAIN_BELOW_B8"
    else:terminal="R4_LEVEL_ZEROSHOT_CANONICALIZATION_NOT_SUPPORTED"
    afc.write_json(ROOT/"ZEROSHOT_GATES.json",dict(best_zero_shot_variant=best,terminal=terminal,
                   retrospective_winner_no_multiplicity_correction=True,variants=gates))
    afc.write_json(ROOT/"GEOMETRY_MECHANISM_AUDIT.json",dict(reference="Z0_A1",variants=mechanisms,
                   post_score_target_labels_only=True,patient_specific_head_non_deployable=True))
    return best,terminal,gates


def main():
    preflight()
    cells,rows,exact,reps=load()
    fit_selection_table()
    summary,folds=outcome_tables(rows,exact)
    if abs(summary["Z0_A1"]["ap"]-.5767434626151353)>1e-6:
        raise RuntimeError("SOURCE_A1_MATCHED_QUERY_REPRODUCTION_FAILED")
    disp,head,rev=diagnostics(cells,reps)
    best,terminal,gates=decisions(summary,folds,disp,head,rev)
    afc.write_csv(ROOT/"MATCHED_B8_REFERENCE.csv",[
        dict(reference="CURRENT_64D_B8",ap=.5996322681940147,role="reference_only_not_zero_shot"),
        dict(reference="BEST_OBSERVED_B8_TEACHER",ap=.6052912572363056,role="reference_only_not_zero_shot")])
    prior=json.loads((PROJECT/"b8_teacher_ceiling_meta_readout_seed42_v1"/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    afc.write_json(ROOT/"SOURCE_REPRODUCTION.json",{**prior,"exact_a1_matched_query_ap":summary["Z0_A1"]["ap"],
                   "B0_IDENTITY_AUDIT_PASS":True})
    afc.write_json(ROOT/"B0_IDENTITY_AUDIT.json",{"pass":True,"source_ap":summary["Z0_A1"]["ap"],
                   "expected_ap":.5767434626151353,"absolute_error":abs(summary["Z0_A1"]["ap"]-.5767434626151353),
                   "cells":65,"repetitions":1300,"budget":0})
    afc.write_json(ROOT/"LABEL_USAGE_AUDIT.json",dict(target_cells=65,unique_patient_ids=47,repetitions=1300,
                   deployment_budget=0,candidate_features_or_labels_for_new_model=False,
                   fit_only_hyperparameter_selection=True,all_new_variant_score_grids_frozen_before_vloo_labels=True,
                   five_a1_epoch30_diagnostic_r4_snapshots_frozen_before_vloo_labels=True,
                   diagnostic_amendment_sha="765107a265c48d139430c97d95da2aeb513dbadff6a639984ff2b04afa69e0bf",
                   own_target_label_excluded_from_own_vloo_selection=True,
                   strict_target_label_sequencing=False,
                   note="Exact cross-patient A1 VLOO reads each target label to select other targets; all score grids were frozen first, but final decisions cannot all be frozen before any target label is accessed.",
                   legacy_loader_materializes_all_80_labels=True,strict_no_outer_label_materialization=False,
                   outer_predictions_metrics_selection=False))
    (ROOT/"IMPLEMENTATION_AUDIT.md").write_text(
        "# Implementation audit\n\n- Protocol lock was pushed before new target outcomes. A1 150-checkpoint/R4 replay and matched B0 AP were rechecked.\n"
        "- Identical source A1 architecture, per-fold initial state, patient-equal BCE, optimizer, 30 epochs, and 19-threshold VLOO. Geometry heads/adversary are training-only.\n"
        "- FIT-only patient-ID-disjoint 80/20 hyperparameter selection; full-FIT retraining after selection. Z4 geometry and meta parameters derive solely from FIT selections.\n"
        "- All 900 new-model epoch/fold/variant score and R4 snapshots, 30 FIT-selection files, and five exact A1 epoch-30 R4 snapshots were hash-frozen before the new target-label evaluation pass. Candidate pools were ignored for B=0 fixed-query prediction.\n"
        "- A pre-outcome diagnostic amendment fixes epoch 30 as the common checkpoint within each fold for every variant. Cross-patient R4 cosine uses only within-fold pairs from that same model state; VLOO-selected states are never mixed for geometry diagnostics. Primary VLOO performance is unchanged.\n"
        "- Literal strict target-label sequencing is false: exact A1 VLOO uses a patient's label to select another patient's checkpoint. The patient's own label never enters its own selection; all candidate score grids were frozen first. The legacy loader also materializes all 80 labels, making strict outer label non-materialization false. No outer predictions, metrics or selection were computed.\n"
        "- Geometry diagnostics, target-specific OOF headroom and FIT-direction transfer are post-score exploratory at common epoch 30; patient-specific heads are nondeployable. Winner/gates are retrospective across multiple variants without family-wise correction.\n",
        encoding="utf-8")
    s=summary;g=gates
    report=["# Zero-shot Patient Geometry Recovery Study — development-only","",
        "65 fixed VLOO cells, 47 unique patients, 20 fixed 50/50 query repetitions. B=0 inference; no candidate pool used.","",
        f"1. Exact A1 reproduction: yes, 150/150 checkpoint/R4 replay within 1e-6; fixed-query AP {s['Z0_A1']['ap']:.6f}.",
        f"2. Z1 direction canonicalization AP {s['Z1_DIRECTION_CANONICALIZATION']['ap']:.6f}, delta {s['Z1_DIRECTION_CANONICALIZATION']['delta_ap_vs_a1']:+.6f}.",
        f"3. Z2 class-conditional alignment AP {s['Z2_CLASS_CONDITIONAL_ALIGNMENT']['ap']:.6f}, delta {s['Z2_CLASS_CONDITIONAL_ALIGNMENT']['delta_ap_vs_a1']:+.6f}. Z2B cross-patient centroid-SupCon AP {s['Z2B_CROSSPATIENT_SUPCON']['ap']:.6f}.",
        f"4. Z3 patient-heldout first-order MLDG AP {s['Z3_PATIENT_HELDOUT_MLDG']['ap']:.6f}, delta {s['Z3_PATIENT_HELDOUT_MLDG']['delta_ap_vs_a1']:+.6f}.",
        f"5. Z4 FIT-selected geometry plus MLDG AP {s['Z4_GEOMETRY_PLUS_MLDG']['ap']:.6f}, delta {s['Z4_GEOMETRY_PLUS_MLDG']['delta_ap_vs_a1']:+.6f}.",
        f"6. Best descriptive zero-shot variant `{best}` AP {s[best]['ap']:.6f}, paired delta {s[best]['delta_ap_vs_a1']:+.6f} [{s[best]['delta_ap_ci_low']:+.6f},{s[best]['delta_ap_ci_high']:+.6f}], positive folds {g[best]['positive_folds']}/5.",
        f"7. Reaches original B8 0.599632 under predeclared gate: {g[best]['ZERO_SHOT_REACHES_CURRENT_B8']}.",
        f"8. Reaches best B8 0.605291 under predeclared gate: {g[best]['ZERO_SHOT_REACHES_BEST_B8']}.",
        f"9. Patient-specific-minus-shared headroom at common epoch 30: A1 {head['Z0_A1']['gap']:.6f}; best {head[best]['gap']:.6f}. Smaller supports, but does not prove, canonicalization.",
        f"10. Mean within-fold cross-patient direction cosine at common epoch 30: A1 {disp['Z0_A1']['mean_pairwise_cosine']:.6f}; best {disp[best]['mean_pairwise_cosine']:.6f}.",
        f"11. FIT-direction reversal rate at common epoch 30: A1 {rev['Z0_A1']['reversal_rate']:.6f}; best {rev[best]['reversal_rate']:.6f}.",
        f"12. Interpretation: `{terminal}`. Any richer-physiology switch is a next-study hypothesis, not a causal conclusion from these retrospective data.","",
        "No B8 training, target adaptation, Student distillation or outer evaluation. The legacy loader materializes all 80 labels. Exact A1 VLOO creates cross-patient label dependencies, so strict target-label sequencing is false even though all candidate score/R4 grids were frozen before the new label-using pass. Descriptive best-variant selection is uncorrected for multiplicity."]
    (ROOT/"FINAL_REPORT.md").write_text("\n".join(report)+"\n",encoding="utf-8")
    print(f"[FINAL] best={best} AP={s[best]['ap']:.6f} terminal={terminal}",flush=True)


if __name__=="__main__":main()
