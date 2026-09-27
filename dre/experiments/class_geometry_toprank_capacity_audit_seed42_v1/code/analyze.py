"""Predeclared VLOO development analyses; public fold/aggregate rows only."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

from common import (EXPERIMENT, RUNTIME, assert_no_outer_loader, build_fold, ensure_source,
                    finalize_fold, make_experiment, mean, read_csv, source_checkpoint, source_grid, write_csv, write_json)

METRICS = ("patient_macro_f1", "patient_ez_f1", "patient_nez_f1", "patient_balanced_accuracy",
           "patient_ez_auprc", "patient_ez_auroc", "patient_ez_mrr", "top1_is_ez", "predicted_ez_fraction")


def grid_path(variant, fold, epoch):
    if variant in ("B00", "CAP0_ORIGINAL"):
        return None
    if variant == "CAP0_BALANCED":
        variant = "B10"
    return RUNTIME / "private" / "training" / variant / f"fold_{fold}" / f"epoch_{epoch:02d}_validation_grid.json"


def grids(variant, fold):
    if variant in ("B00", "CAP0_ORIGINAL"):
        return [source_grid(fold, epoch) for epoch in range(1, 31)]
    source_variant = "B10" if variant == "CAP0_BALANCED" else variant
    folder = RUNTIME / "private" / "training" / source_variant / f"fold_{fold}"
    complete = json.loads((folder / "complete.json").read_text(encoding="utf-8"))
    if complete["epochs"] != 30 or complete["outer_test_accessed"]:
        raise RuntimeError(f"Incomplete training variant {source_variant} fold {fold}")
    return [json.loads(grid_path(variant, fold, epoch).read_text(encoding="utf-8")) for epoch in range(1, 31)]


def selected(variant, output, public_name):
    rows, patients = [], []
    for fold in range(1, 6):
        private = RUNTIME / "private" / "selected" / f"{variant}_fold_{fold}.csv"
        result, _ = finalize_fold(grids(variant, fold), variant, fold, private)
        rows.append(result)
        patients.extend(read_csv(private))
    write_csv(output / public_name, rows)
    if len(patients) != 65:
        raise RuntimeError("Expected 65 excluded validation cases")
    return rows, patients


def aggregate(rows):
    return {metric: mean(rows, metric) for metric in METRICS}


def comparison(base, candidate, name):
    a, b = aggregate(base), aggregate(candidate)
    row = {"candidate": name}
    for metric in METRICS:
        row[f"baseline_{metric}"] = a[metric]
        row[f"candidate_{metric}"] = b[metric]
        row[f"delta_{metric}"] = b[metric] - a[metric]
    row["positive_auprc_folds"] = sum(float(x["patient_ez_auprc"]) > float(y["patient_ez_auprc"]) for x, y in zip(candidate, base, strict=True))
    row["positive_mrr_folds"] = sum(float(x["patient_ez_mrr"]) > float(y["patient_ez_mrr"]) for x, y in zip(candidate, base, strict=True))
    return row


def aligned(base, candidate):
    a = {(int(row["fold"]), row["subject_id"]): row for row in base}
    b = {(int(row["fold"]), row["subject_id"]): row for row in candidate}
    if len(a) != 65 or set(a) != set(b):
        raise RuntimeError("Paired VLOO patient membership changed")
    return [(key, a[key], b[key]) for key in sorted(a)]


def validation_prevalence():
    exp = make_experiment()
    meta = {}
    for split in exp.outer_splits:
        fold, _, _, val_loader, test_loader, _ = build_fold(exp, split, "validation")
        assert_no_outer_loader(test_loader)
        for example in val_loader.dataset.patient_examples:
            valid = np.asarray(example["channel_mask"], dtype=bool)
            yez = np.asarray(example["labels_ez"], dtype=float)[valid]
            n_ez, n_nez = int(sum(yez > 0.5)), int(sum(yez < 0.5))
            if not n_ez or not n_nez:
                raise RuntimeError("Single-class validation patient")
            meta[(fold, str(example["subject_id"]))] = {"n_ez": n_ez, "n_nez": n_nez,
                   "true_ez_fraction": n_ez / (n_ez + n_nez), "original_ez_loss_mass": 2*n_ez/(2*n_ez+n_nez)}
    if len(meta) != 65:
        raise RuntimeError("Validation metadata case count changed")
    order = sorted(meta, key=lambda key: (meta[key]["true_ez_fraction"], key))
    for quartile, keys in enumerate(np.array_split(np.asarray(order, dtype=object), 4), 1):
        for fold, subject in keys:
            meta[(fold, subject)]["quartile"] = f"Q{quartile}"
    return meta


def prevalence_summary(variants, meta, deltas=()):
    output = []
    for name, patient_rows in variants.items():
        by_key = {(int(row["fold"]), row["subject_id"]): row for row in patient_rows}
        if set(by_key) != set(meta):
            raise RuntimeError("Prevalence membership mismatch")
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            items = [row for key, row in by_key.items() if meta[key]["quartile"] == quartile]
            record = {"variant": name, "quartile": quartile, "n_cases": len(items)}
            for metric in ("patient_macro_f1", "patient_ez_f1", "patient_ez_auprc", "patient_ez_mrr", "top1_is_ez"):
                record[metric] = mean(items, metric)
            output.append(record)
    for name, base_name, candidate_name in deltas:
        base = {(int(r["fold"]), r["subject_id"]): r for r in variants[base_name]}
        new = {(int(r["fold"]), r["subject_id"]): r for r in variants[candidate_name]}
        for quartile in ("Q1", "Q2", "Q3", "Q4"):
            keys = [key for key in meta if meta[key]["quartile"] == quartile]
            record = {"variant": name, "quartile": quartile, "n_cases": len(keys)}
            for metric in ("patient_macro_f1", "patient_ez_f1", "patient_ez_auprc", "patient_ez_mrr", "top1_is_ez"):
                record[metric] = float(np.mean([float(new[key][metric])-float(base[key][metric]) for key in keys]))
            output.append(record)
    return output


def audit_a():
    output = EXPERIMENT / "audit_a_class_geometry"
    a0, p0 = selected("B00", output, "A0_VLOO_BY_FOLD.csv")
    a1, p1 = selected("B10", output, "A1_VLOO_BY_FOLD.csv")
    c = comparison(a0, a1, "A1")
    write_csv(output / "CLASS_GEOMETRY_COMPARISON.csv", [c])
    meta = validation_prevalence()
    pairs = aligned(p0, p1)
    table = []
    imbalance, gains = [], defaultdict(list)
    for key, base, new in pairs:
        mass = meta[key]["original_ez_loss_mass"]
        record = {"fold": key[0], "true_ez_fraction_quartile": meta[key]["quartile"],
                  "original_ez_loss_mass": mass, "imbalance": abs(mass-0.5)}
        for short, metric in (("auprc", "patient_ez_auprc"), ("mrr", "patient_ez_mrr"), ("macro_f1", "patient_macro_f1")):
            delta = float(new[metric])-float(base[metric])
            record[f"delta_{short}"] = delta
            gains[short].append(delta)
        imbalance.append(record["imbalance"])
        table.append(record)
    # Private patient-level diagnostic; public correlations/quartiles only.
    write_csv(RUNTIME / "private" / "loss_mass_patient_cases.csv", table)
    correlations = []
    for short in ("auprc", "mrr", "macro_f1"):
        x, y = np.asarray(imbalance), np.asarray(gains[short])
        correlations.append({"metric": short, "pearson": float(np.corrcoef(x,y)[0,1]),
                             "spearman": float(spearmanr(x,y).statistic), "n_cases":65})
    write_csv(output / "LOSS_MASS_DIAGNOSTIC.csv", correlations)
    write_csv(output / "EZ_FRACTION_QUARTILES.csv", prevalence_summary({"A0":p0,"A1":p1}, meta, (("A1_MINUS_A0","A0","A1"),)))
    checks = {"auprc_gain_ge_0_010":c["delta_patient_ez_auprc"]>=0.010,
              "auprc_positive_folds_ge_4":c["positive_auprc_folds"]>=4,
              "mrr_nondecreasing":c["delta_patient_ez_mrr"]>=0,
              "top1_nondecreasing":c["delta_top1_is_ez"]>=0,
              "macro_f1_preserved":c["delta_patient_macro_f1"]>=-0.005,
              "ez_f1_preserved":c["delta_patient_ez_f1"]>=-0.005}
    passed=all(checks.values())
    write_json(output / "CLASS_GEOMETRY_GATE.json", {"pass":passed,"checks":checks,
               "imbalance_ranking_direction_supported":correlations[0]["pearson"]>0 and correlations[0]["spearman"]>0,
               "terminal":"CLASS_GEOMETRY_RANKING_BOTTLENECK_SUPPORTED" if passed else "CLASS_GEOMETRY_RANKING_BOTTLENECK_NOT_SUPPORTED",
               "outer_test_accessed":False})
    print(json.dumps({"audit":"A","gate":passed,"delta_auprc":c["delta_patient_ez_auprc"]}),flush=True)


def record_diagnostics(record):
    valid=np.asarray(record["channel_mask"],dtype=bool)
    labels=np.asarray(record["labels_ez"],dtype=float)[valid]
    score=np.asarray(record["score_ez"],dtype=float)[valid]
    order=np.argsort(-score,kind="stable")
    ordered=labels[order]
    n_ez=int(sum(labels>0.5))
    positive=score[labels>0.5]
    negative=score[labels<0.5]
    top_negative=np.sort(negative)[-min(16,len(negative)):]
    def ndcg(k):
        weights=1/np.log2(np.arange(2,min(k,len(labels))+2))
        dcg=float(np.sum(ordered[:len(weights)]*weights))
        ideal=float(np.sum(weights[:min(n_ez,len(weights))]))
        return dcg/ideal if ideal else 0.0
    return {"recall_at_1":float(sum(ordered[:1])/n_ez),
            "recall_at_3":float(sum(ordered[:3])/n_ez),
            "recall_at_5":float(sum(ordered[:5])/n_ez),
            "ndcg_at_5":ndcg(5),"ndcg_at_10":ndcg(10),
            "true_k_ez_recall":float(sum(ordered[:n_ez])/n_ez),
            "hard_negative_margin":float(min(positive)-max(top_negative)),
            "best_ez_minus_best_nez":float(max(positive)-max(negative)),
            "median_ez_minus_top16_nez_mean":float(np.median(positive)-np.mean(top_negative))}


def selected_prediction_diagnostics(variant):
    import torch
    from common import A1_RUNTIME
    from train import build_model
    exp=make_experiment()
    ez_weight=torch.tensor(2.0,dtype=torch.float32,device=exp.device)
    rows=[]
    for split in exp.outer_splits:
        fold, _, train_loader, val_loader, test_loader, _=build_fold(exp,split,"validation")
        assert_no_outer_loader(test_loader)
        selected_rows=read_csv(RUNTIME/"private"/"selected"/f"{variant}_fold_{fold}.csv")
        selected_by_id={row["subject_id"]:row for row in selected_rows}
        initial=torch.load(A1_RUNTIME/"initial"/f"fold_{fold}_initial.pt",map_location="cpu",weights_only=True)
        source_variant="B10" if variant=="CAP0_BALANCED" else variant
        if source_variant in ("B00","CAP0_ORIGINAL"):
            model=build_model(exp,train_loader,initial,fold,"B10")
        else:
            model=build_model(exp,train_loader,initial,fold,source_variant)
        for epoch in sorted({int(row["selected_epoch"]) for row in selected_rows}):
            checkpoint=(source_checkpoint(fold,epoch) if source_variant in ("B00","CAP0_ORIGINAL") else
                        RUNTIME/"private"/"training"/source_variant/f"fold_{fold}"/f"epoch_{epoch:02d}.pt")
            saved=torch.load(checkpoint,map_location=exp.device,weights_only=False)
            model.load_state_dict(saved["model_state_dict"],strict=True)
            _,_,records=exp._evaluate(model,val_loader,ez_weight,split_name="val")
            for record in records:
                subject=str(record["subject_id"])
                if int(selected_by_id[subject]["selected_epoch"])==epoch:
                    rows.append({"variant":variant,"fold":fold,"subject_id":subject,
                                 **record_diagnostics(record)})
    if len(rows)!=65 or len({(r["fold"],r["subject_id"]) for r in rows})!=65:
        raise RuntimeError("Selected prediction diagnostic membership changed")
    write_csv(RUNTIME/"private"/"selected_diagnostics"/f"{variant}.csv",rows)
    return rows


def audit_b():
    output=EXPERIMENT/"audit_b_top_heavy"
    folds,patients={},{}
    for variant in ("B00","B10","B01","B11"):
        folds[variant],patients[variant]=selected(variant,output,f"{variant}_VLOO_BY_FOLD.csv")
    compares=[comparison(folds[base],folds[new],new) for base,new in (("B00","B01"),("B10","B11"))]
    write_csv(output/"TOP_HEAVY_COMPARISON.csv",compares)
    diagnostic={variant:selected_prediction_diagnostics(variant) for variant in folds}
    topk=[]
    for variant,rows in diagnostic.items():
        item={"variant":variant,"n_cases":len(rows)}
        for key in ("recall_at_1","recall_at_3","recall_at_5","ndcg_at_5","ndcg_at_10","true_k_ez_recall"):
            item[key]=mean(rows,key)
        topk.append(item)
    write_csv(output/"TOPK_RANKING_METRICS.csv",topk)
    margin_rows=[]
    checks={}
    for base,new in (("B00","B01"),("B10","B11")):
        pairs=aligned(diagnostic[base],diagnostic[new])
        margin={"matched_effect":f"{new}_minus_{base}","n_cases":65}
        for key in ("hard_negative_margin","best_ez_minus_best_nez","median_ez_minus_top16_nez_mean"):
            margin[f"baseline_{key}"]=mean(diagnostic[base],key)
            margin[f"candidate_{key}"]=mean(diagnostic[new],key)
            margin[f"delta_{key}"]=float(np.mean([float(b[key])-float(a[key]) for _,a,b in pairs]))
        margin_rows.append(margin)
        c=next(row for row in compares if row["candidate"]==new)
        check={"mrr_gain_ge_0_010":c["delta_patient_ez_mrr"]>=0.010,
               "top1_gain_ge_0_030":c["delta_top1_is_ez"]>=0.030,
               "auprc_preserved":c["delta_patient_ez_auprc"]>=-0.005,
               "macro_f1_preserved":c["delta_patient_macro_f1"]>=-0.005,
               "positive_mrr_folds_ge_4":c["positive_mrr_folds"]>=4,
               "hard_negative_margin_improves":margin["delta_hard_negative_margin"]>0}
        checks[new]={"pass":all(check.values()),"checks":check}
    write_csv(output/"HARD_NEGATIVE_MARGIN_DIAGNOSTICS.csv",margin_rows)
    gate_a=json.loads((EXPERIMENT/"audit_a_class_geometry"/"CLASS_GEOMETRY_GATE.json").read_text(encoding="utf-8"))
    triggered=all(not checks[new]["checks"]["mrr_gain_ge_0_010"] for new in ("B01","B11"))
    passed=any(item["pass"] for item in checks.values())
    write_json(output/"HARD_NEGATIVE_GATE.json",{"pass":passed,"effects":checks,
               "first_positive_triggered":triggered,"first_positive_base":"balanced" if gate_a["pass"] else "original",
               "terminal":"HARD_NEGATIVE_RANKING_SUPPORTED" if passed else "HARD_NEGATIVE_RANKING_NOT_SUPPORTED",
               "outer_test_accessed":False})
    meta=validation_prevalence()
    write_csv(EXPERIMENT/"diagnostics"/"EZ_PREVALENCE_EFFECT_SUMMARY.csv",
              prevalence_summary(patients,meta,(("B01_MINUS_B00","B00","B01"),("B11_MINUS_B10","B10","B11"))))
    print(json.dumps({"audit":"B","gate":passed,"first_positive_triggered":triggered,
                      "B01_mrr":compares[0]["delta_patient_ez_mrr"],"B11_mrr":compares[1]["delta_patient_ez_mrr"]}),flush=True)


def capacity_trajectories(base_variant):
    import torch
    import types
    from common import A1_RUNTIME
    from losses import loss_for_variant
    from train import build_model, fit_ap
    exp=make_experiment()
    exp._compute_loss=types.MethodType(loss_for_variant("CAP0","balanced" if base_variant=="CAP0_BALANCED" else "original"),exp)
    ez_weight=torch.tensor(2.0,dtype=torch.float32,device=exp.device)
    rows=[]
    for split in exp.outer_splits:
        fold,train_set,train_loader,_,test_loader,_=build_fold(exp,split,"validation")
        assert_no_outer_loader(test_loader)
        fit_loader=exp._make_loader(train_set,shuffle=False,batch_size=2)
        initial=torch.load(A1_RUNTIME/"initial"/f"fold_{fold}_initial.pt",map_location="cpu",weights_only=True)
        model=build_model(exp,train_loader,initial,fold,"B10")
        frozen_grids=grids(base_variant,fold)
        for epoch in range(1,31):
            checkpoint=(source_checkpoint(fold,epoch) if base_variant=="CAP0_ORIGINAL" else
                        RUNTIME/"private"/"training"/"B10"/f"fold_{fold}"/f"epoch_{epoch:02d}.pt")
            saved=torch.load(checkpoint,map_location=exp.device,weights_only=False)
            model.load_state_dict(saved["model_state_dict"],strict=True)
            fit_auprc,fit_loss=fit_ap(exp,model,fit_loader,ez_weight)
            grid=frozen_grids[epoch-1]
            rows.append({"variant":"CAP0","fold":fold,"epoch":epoch,"fit_auprc":fit_auprc,
                         "validation_auprc":float(np.mean([r["fixed"]["patient_ez_auprc"] for r in grid["patients"]])),
                         "fit_bce":fit_loss,
                         "validation_macro_f1_at_0_5":float(np.mean([r["grid"]["patient_macro_f1"][9] for r in grid["patients"]]))})
            print(f"[CAP0-TRAJECTORY] fold={fold} epoch={epoch}",flush=True)
    return rows


def audit_c():
    gate_a=json.loads((EXPERIMENT/"audit_a_class_geometry"/"CLASS_GEOMETRY_GATE.json").read_text(encoding="utf-8"))
    base="CAP0_BALANCED" if gate_a["pass"] else "CAP0_ORIGINAL"
    output=EXPERIMENT/"audit_c_capacity"
    folds,patients={},{}
    for variant,alias in ((base,"CAP0"),("CAP1","CAP1"),("CAP2","CAP2")):
        folds[alias],patients[alias]=selected(variant,output,f"{alias}_VLOO_BY_FOLD.csv")
    trajectory_file=RUNTIME/"private"/"CAP0_trajectory.json"
    if trajectory_file.is_file():
        source_trajectory=json.loads(trajectory_file.read_text(encoding="utf-8"))
    else:
        source_trajectory=capacity_trajectories(base)
        write_json(trajectory_file,source_trajectory)
    trajectories=list(source_trajectory)
    for variant in ("CAP1","CAP2"):
        for fold in range(1,6):
            frozen_grids=grids(variant,fold)
            for epoch in range(1,31):
                row=json.loads((RUNTIME/"private"/"training"/variant/f"fold_{fold}"/f"epoch_{epoch:02d}_trajectory.json").read_text(encoding="utf-8"))
                grid=frozen_grids[epoch-1]
                trajectories.append({"variant":variant,"fold":fold,"epoch":epoch,
                                     "fit_auprc":row["fit_auprc"],"fit_bce":row["fit_bce"],
                                     "validation_auprc":float(np.mean([r["fixed"]["patient_ez_auprc"] for r in grid["patients"]])),
                                     "validation_macro_f1_at_0_5":row["validation_macro_f1_at_0_5"]})
    write_csv(output/"TRAIN_VALIDATION_RANKING_GAP.csv",trajectories)
    topk=[]
    for source_variant, public_name in ((base,"CAP0"),("CAP1","CAP1"),("CAP2","CAP2")):
        cases=selected_prediction_diagnostics(source_variant)
        item={"variant":public_name,"n_cases":len(cases)}
        for key in ("recall_at_1","recall_at_3","recall_at_5","ndcg_at_5","ndcg_at_10","true_k_ez_recall"):
            item[key]=mean(cases,key)
        topk.append(item)
    write_csv(output/"CAPACITY_TOPK_DIAGNOSTIC.csv",topk)
    by_trajectory={(r["variant"],int(r["fold"]),int(r["epoch"])):r for r in trajectories}
    fit_selected={}
    for variant in ("CAP0","CAP1","CAP2"):
        values=[]
        for row in patients[variant]:
            values.append(float(by_trajectory[(variant,int(row["fold"]),int(row["selected_epoch"]))]["fit_auprc"]))
        fit_selected[variant]=float(np.mean(values))
    comparisons=[]
    checks={}
    for variant in ("CAP1","CAP2"):
        c=comparison(folds["CAP0"],folds[variant],variant)
        c["baseline_fit_auprc"]=fit_selected["CAP0"]
        c["candidate_fit_auprc"]=fit_selected[variant]
        c["baseline_train_validation_gap"]=fit_selected["CAP0"]-aggregate(folds["CAP0"])["patient_ez_auprc"]
        c["candidate_train_validation_gap"]=fit_selected[variant]-aggregate(folds[variant])["patient_ez_auprc"]
        c["gap_increase"]=c["candidate_train_validation_gap"]-c["baseline_train_validation_gap"]
        comparisons.append(c)
        check={"auprc_gain_ge_0_010":c["delta_patient_ez_auprc"]>=0.010,
               "positive_auprc_folds_ge_4":c["positive_auprc_folds"]>=4,
               "mrr_nondecreasing":c["delta_patient_ez_mrr"]>=0,
               "top1_nondecreasing":c["delta_top1_is_ez"]>=0,
               "macro_f1_preserved":c["delta_patient_macro_f1"]>=-0.005,
               "gap_not_substantially_larger":c["gap_increase"]<0.010}
        checks[variant]={"pass":all(check.values()),"checks":check}
    write_csv(output/"CAPACITY_COMPARISON.csv",comparisons)
    passed=any(item["pass"] for item in checks.values())
    write_json(output/"CAPACITY_GATE.json",{"pass":passed,"effects":checks,"capacity_base":base,
               "terminal":"FEATURE_ENCODER_UNDERCAPACITY_SUPPORTED" if passed else "FEATURE_ENCODER_UNDERCAPACITY_NOT_SUPPORTED",
               "outer_test_accessed":False})
    meta=validation_prevalence()
    previous=read_csv(EXPERIMENT/"diagnostics"/"EZ_PREVALENCE_EFFECT_SUMMARY.csv")
    write_csv(EXPERIMENT/"diagnostics"/"EZ_PREVALENCE_EFFECT_SUMMARY.csv",
              previous+prevalence_summary(patients,meta,(("CAP1_MINUS_CAP0","CAP0","CAP1"),("CAP2_MINUS_CAP0","CAP0","CAP2"))))
    print(json.dumps({"audit":"C","gate":passed,"base":base,
                      "CAP1_auprc":comparisons[0]["delta_patient_ez_auprc"],
                      "CAP2_auprc":comparisons[1]["delta_patient_ez_auprc"]}),flush=True)


def first_positive():
    gate=json.loads((EXPERIMENT/"audit_b_top_heavy"/"HARD_NEGATIVE_GATE.json").read_text(encoding="utf-8"))
    if not gate["first_positive_triggered"]:
        raise RuntimeError("First-positive diagnostic was not predeclared-triggered")
    output=EXPERIMENT/"first_positive_diagnostic"
    fp,pf=selected("FP",output,"FIRST_POSITIVE_BY_FOLD.csv")
    baseline="B10" if gate["first_positive_base"]=="balanced" else "B00"
    folds,pb=selected(baseline,output,"FIRST_POSITIVE_BASE_PRIVATE.csv")
    (output/"FIRST_POSITIVE_BASE_PRIVATE.csv").unlink()
    c=comparison(folds,fp,"FP")
    write_json(output/"FIRST_POSITIVE_DIAGNOSTIC.json",{"diagnostic_only":True,"baseline":baseline,
               "delta_auprc":c["delta_patient_ez_auprc"],"delta_mrr":c["delta_patient_ez_mrr"],
               "delta_top1":c["delta_top1_is_ez"],"delta_macro_f1":c["delta_patient_macro_f1"],
               "outer_test_accessed":False})
    meta=validation_prevalence()
    previous=read_csv(EXPERIMENT/"diagnostics"/"EZ_PREVALENCE_EFFECT_SUMMARY.csv")
    write_csv(EXPERIMENT/"diagnostics"/"EZ_PREVALENCE_EFFECT_SUMMARY.csv",
              previous+prevalence_summary({baseline:pb,"FP":pf},meta,(("FP_MINUS_BASE",baseline,"FP"),)))
    print(json.dumps({"audit":"FP","delta_mrr":c["delta_patient_ez_mrr"]}),flush=True)


def paired_bootstrap():
    source_pairs=(("A1_minus_A0","B00","B10"),("B01_minus_B00","B00","B01"),
                  ("B11_minus_B10","B10","B11"))
    gate=json.loads((EXPERIMENT/"audit_a_class_geometry"/"CLASS_GEOMETRY_GATE.json").read_text(encoding="utf-8"))
    base="CAP0_BALANCED" if gate["pass"] else "CAP0_ORIGINAL"
    pairs=(*source_pairs,("CAP1_minus_CAP0",base,"CAP1"),("CAP2_minus_CAP0",base,"CAP2"))
    rng=np.random.default_rng(42)
    rows=[]
    for effect,old,new in pairs:
        old_rows=[r for fold in range(1,6) for r in read_csv(RUNTIME/"private"/"selected"/f"{old}_fold_{fold}.csv")]
        new_rows=[r for fold in range(1,6) for r in read_csv(RUNTIME/"private"/"selected"/f"{new}_fold_{fold}.csv")]
        cases=aligned(old_rows,new_rows)
        for metric in ("patient_ez_auprc","patient_ez_mrr","patient_macro_f1"):
            delta=np.asarray([float(b[metric])-float(a[metric]) for _,a,b in cases])
            boots=np.mean(delta[rng.integers(0,65,size=(10000,65))],axis=1)
            rows.append({"effect":effect,"metric":metric,"n_paired_cases":65,
                         "mean_delta":float(delta.mean()),"median_delta":float(np.median(delta)),
                         "ci_95_lower":float(np.quantile(boots,0.025)),
                         "ci_95_upper":float(np.quantile(boots,0.975)),
                         "fraction_positive":float(np.mean(delta>0))})
    write_csv(EXPERIMENT/"diagnostics"/"PAIRED_BOOTSTRAP_SUMMARY.csv",rows)
    print(json.dumps({"paired_effects":5,"metrics":3,"bootstrap_resamples":10000}),flush=True)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("a","b","c","fp","bootstrap"), required=True)
    options=parser.parse_args()
    ensure_source()
    if options.phase=="a": audit_a()
    if options.phase=="b": audit_b()
    if options.phase=="c": audit_c()
    if options.phase=="fp": first_positive()
    if options.phase=="bootstrap": paired_bootstrap()


if __name__=="__main__":
    main()
