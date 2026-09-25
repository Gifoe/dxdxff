from __future__ import annotations
import math,csv,hashlib
from data_factory import build_outer_splits,split_train_val_subjects
def make_protocol(subject_ids,n_splits=5,val_ratio=.2,seed=42):
    outer=build_outer_splits({s:{} for s in subject_ids},n_splits=n_splits,random_seed=seed)
    for f in outer:f["train_subjects"],f["val_subjects"]=split_train_val_subjects(f.pop("train_subjects"),val_ratio=val_ratio,random_seed=seed,fold_idx=f["fold_idx"])
    return outer
def make_protocol_from_ledger(subject_ids,ledger_path,n_splits=5,val_ratio=.2,seed=42):
    with open(ledger_path,newline="",encoding="utf-8-sig") as f: rows=list(csv.DictReader(f))
    key="outer_fold" if rows and "outer_fold" in rows[0] else "fold_idx"; assigned={}
    for r in rows:
      sid=str(r["subject_id"]); fold=int(r[key]);
      if sid in assigned: raise ValueError(f"duplicate subject assignment in fold ledger: {sid}")
      assigned[sid]=fold
    cohort=set(subject_ids)
    if set(assigned)!=cohort: raise ValueError(f"fold ledger cohort mismatch: missing={sorted(cohort-set(assigned))}, unknown={sorted(set(assigned)-cohort)}")
    if set(assigned.values())!=set(range(1,n_splits+1)): raise ValueError("fold ledger fold count mismatch")
    folds=[]
    for k in range(1,n_splits+1):
      outer=[s for s in subject_ids if assigned[s]!=k]; tr,va=split_train_val_subjects(outer,val_ratio=val_ratio,random_seed=seed,fold_idx=k); folds.append({"fold_idx":k,"train_subjects":tr,"val_subjects":va,"test_subjects":[s for s in subject_ids if assigned[s]==k]})
    return folds
def ledger_sha256(path):
    return hashlib.sha256(open(path,"rb").read()).hexdigest()
def audit_protocol(folds,subject_ids,oof_rows,patient_index,required_n):
    errors=[]; tests=[]
    for f in folds:
      tr=set(f["train_subjects"]); va=set(f.get("validation_subjects",f.get("val_subjects",[]))); te=set(f["test_subjects"]); nf=set(f["normalizer_fit_subjects"]); ts=set(f["threshold_selection_subjects"])
      if tr&va or tr&te or va&te:errors.append("train/validation/test overlap")
      if not nf<=tr or nf&(va|te):errors.append("normalizer subject leakage")
      if ts!=va or ts&te:errors.append("threshold subject leakage")
      if f["threshold_source"]!="validation":errors.append("test-derived threshold")
      tests+=list(te)
    if sorted(tests)!=sorted(subject_ids):errors.append("outer test membership is not exactly once")
    if len(set(r["subject_id"] for r in oof_rows))!=len(subject_ids):errors.append("OOF patient coverage incomplete")
    keys=[(r["subject_id"],r["channel_name"]) for r in oof_rows]
    if len(keys)!=len(set(keys)):errors.append("duplicate patient-channel OOF rows")
    for r in oof_rows:
      if any(not math.isfinite(float(r[k])) for k in ("score_nez_probability","candidate_risk","threshold")):errors.append("NaN/Inf in OOF");break
      if int(r["predicted_nez"])!=int(float(r["score_nez_probability"])>=float(r["threshold"])) or int(r["predicted_ez"])!=1-int(r["predicted_nez"]):errors.append("predictions not derived solely from fold threshold");break
      if int(r["true_nez"])+int(r["true_ez"])!=1:errors.append("labels are not complementary");break
    non_success=[s for s,m in patient_index.items() if m.get("outcome_group")!="success"]
    if non_success:errors.append("non-success patients in Task 1 cohort")
    if len(subject_ids)!=required_n:errors.append("observed patient count differs from required")
    result={"observed_n_patients":len(subject_ids),"required_n_patients":required_n,"all_task1_patients_are_success":not non_success,"train_validation_test_disjoint":not any("overlap" in x for x in errors),"unique_outer_test_membership":sorted(tests)==sorted(subject_ids),"oof_complete":len(set(r["subject_id"] for r in oof_rows))==len(subject_ids),"normalizer_train_only":not any("normalizer" in x for x in errors),"threshold_validation_only":not any("threshold subject" in x for x in errors),"test_derived_threshold":any("test-derived" in x for x in errors),"predictions_threshold_only":not any("predictions" in x for x in errors),"duplicate_patient_channel_rows":len(keys)!=len(set(keys)),"finite_oof":not any("NaN/Inf" in x for x in errors),"labels_complementary":not any("complementary" in x for x in errors),"errors":errors}
    if errors:raise ValueError("Task 1 protocol audit failed: "+"; ".join(sorted(set(errors))))
    return result
__all__=["make_protocol","make_protocol_from_ledger","ledger_sha256","audit_protocol"]
