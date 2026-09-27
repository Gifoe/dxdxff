"""FIT-only lambda selection and fixed-query active calibration. Private cell resume."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickle

import numpy as np
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.preprocessing import StandardScaler

from common import (BUDGETS, LAMBDA_GRID, LOCK_SHA, POLICIES, PRIOR, PRIOR_RUNTIME,
                    ROOT, RUNTIME, fit_residual, load_representation, oracle_direction,
                    preflight, query_metrics, rank_metrics, read_csv, split_indices,
                    stable_seed, write_json)


def source_replay_check():
    audit=json.loads((PRIOR/"SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))
    r4=json.loads((PRIOR/"representations"/"R4_REPLAY_AUDIT.json").read_text(encoding="utf-8"))
    expected=(.6549997115717437,.6410958089865288,.6101185971670673,.6329791804569156,.5907877503876969)
    if audit["checkpoints"]!=150 or audit["max_grid_error"]>1e-6 or not np.allclose(audit["fold_macro_f1"],expected,atol=1e-6,rtol=0):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    if r4["D"]!=64 or r4["max_logit_replay_error"]>1e-6:
        raise RuntimeError("R4 source classifier replay failed")
    for fold in range(1,6):
        status=json.loads((PRIOR_RUNTIME/"private"/f"fold_{fold}"/"extract_status.json").read_text(encoding="utf-8"))
        if status["source_replayed"]!=30 or status["max_grid_error"]>1e-6 or status["max_r4_replay_error"]>1e-6 or status["outer_loader"]:
            raise RuntimeError("Fresh source replay status incomplete")
    write_json(ROOT/"SOURCE_REPRODUCTION.json",{**audit,"R4_D":64,"R4_max_logit_replay_error":r4["max_logit_replay_error"],
              "representation_provenance":"selected-epoch private R4 payloads from exact reference checkpoint, verified after fresh 150-checkpoint A1 replay",
              "no_prior_aggregate_used_as_model_input":True})


def scaler_for(payload):
    fit=payload["fit"]
    scaler=StandardScaler().fit(np.concatenate([fit[sid]["R4"] for sid in sorted(fit)]))
    if scaler.mean_.shape!=(64,) or scaler.scale_.shape!=(64,):
        raise RuntimeError("FIT R4 scaler shape changed")
    return scaler


def margin_for(row, threshold):
    if threshold<=0 or threshold>=1:
        raise RuntimeError("Invalid VLOO NEZ threshold")
    b=math.log(threshold/(1-threshold))
    return b + np.asarray(row["source_ez"],dtype=np.float64)


def verify_b0(row, threshold, m0):
    # A1 original logit is NEZ-positive; source_ez is its negation.
    original=(expit(-np.asarray(row["source_ez"],dtype=np.float64))<threshold)
    if not np.array_equal(m0>0,original):
        raise RuntimeError("ZERO_BUDGET_A1_DECISION_REPLAY_FAILED")
    if not np.array_equal(np.argsort(-m0,kind="stable"),np.argsort(-row["source_ez"],kind="stable")):
        raise RuntimeError("ZERO_BUDGET_A1_RANKING_REPLAY_FAILED")


def fit_episode_data(fit, scaler, fold, epoch, threshold):
    episodes={}
    for sid in sorted(fit):
        row=fit[sid]
        z=scaler.transform(row["R4"])
        m0=margin_for(row,threshold)
        y=row["y"]
        parts=[]
        for rep in range(3):
            candidate,query=split_indices(len(y),42,fold,sid,rep,"fit_lambda")
            if len(candidate)<4:
                continue
            order=np.random.default_rng(stable_seed(42,fold,sid,rep,"fit_lambda_random")).permutation(candidate)
            support=order[:4]
            if len(np.unique(y[query]))!=2:
                continue
            parts.append((m0[support],z[support],y[support],m0[query],z[query],y[query]))
        if parts:
            episodes[sid]=parts
    if not episodes:
        raise RuntimeError("No estimable FIT lambda episodes")
    return episodes


def select_lambda(payload, scaler, fold, epoch, threshold):
    key=f"f{fold}_e{epoch:02d}_t{np.float64(threshold).hex().replace('.','_')}"
    path=RUNTIME/"private"/"fit_lambda"/f"{key}.pkl"
    if path.exists():
        with path.open("rb") as f: cache=pickle.load(f)
        if cache["lock_sha"]!=LOCK_SHA or cache["fold"]!=fold or cache["epoch"]!=epoch or cache["threshold"]!=threshold:
            raise RuntimeError("FIT lambda resume cache mismatch")
        return cache
    episodes=fit_episode_data(payload["fit"],scaler,fold,epoch,threshold)
    scores={}
    counts={}
    for lam in LAMBDA_GRID:
        patient_scores=[]
        for parts in episodes.values():
            vals=[]
            for sm,sz,sy,qm,qz,qy in parts:
                b,w=fit_residual(sm,sz,sy,lam)
                vals.append(float(rank_metrics(qy,qm+b+qz@w)["ap"]))
            patient_scores.append(float(np.mean(vals)))
        scores[str(lam)]=float(np.mean(patient_scores))
        counts[str(lam)]=len(patient_scores)
    # The exact tie rule is absolute score difference <=1e-6, then larger lambda.
    maximum=max(scores.values())
    best=max(lam for lam in LAMBDA_GRID if maximum-scores[str(lam)]<=1e-6)
    cache={"lock_sha":LOCK_SHA,"fold":fold,"epoch":epoch,"threshold":threshold,"selected_lambda":best,
           "scores":scores,"estimable_fit_patients":counts}
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    with temp.open("wb") as f: pickle.dump(cache,f,protocol=5)
    temp.replace(path)
    return cache


def random_support(candidate, fold, sid, rep):
    return list(np.random.default_rng(stable_seed(42,fold,sid,rep,"random")).permutation(candidate))


def oracle_balanced_support(candidate, y, fold, sid, rep, budget):
    if budget<2:
        return None
    ez=np.asarray([c for c in candidate if y[c]==1],dtype=int)
    nez=np.asarray([c for c in candidate if y[c]==0],dtype=int)
    ez=np.random.default_rng(stable_seed(42,fold,sid,rep,"oracle_ez")).permutation(ez)
    nez=np.random.default_rng(stable_seed(42,fold,sid,rep,"oracle_nez")).permutation(nez)
    want_ez=min(budget//2,len(ez))
    want_nez=min(budget-want_ez,len(nez))
    if want_ez+want_nez<budget:
        want_ez=min(budget-want_nez,len(ez))
    return list(ez[:want_ez])+list(nez[:want_nez])


def active_path(policy,candidate,m0,z,y,lam,fold,sid,rep):
    support=[]
    fits={0:(0.0,np.zeros(z.shape[1]))}
    for step in range(1,min(16,len(candidate))+1):
        remaining=np.asarray([c for c in candidate if c not in support],dtype=int)
        b,w=fits[step-1]
        margin=m0[remaining]+b+z[remaining]@w
        tie=np.asarray([stable_seed(42,fold,sid,rep,policy,int(c)) for c in remaining])
        if policy=="UNCERTAINTY":
            order=np.lexsort((tie,np.abs(margin)))
            chosen=int(remaining[order[0]])
        else:
            uncertainty=rankdata(-np.abs(margin),method="average")/len(remaining)
            if not support:
                diversity=np.ones(len(remaining))
            else:
                a=z[remaining]; bmat=z[np.asarray(support)]
                an=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
                bn=np.maximum(np.linalg.norm(bmat,axis=1,keepdims=True),1e-12)
                closest=np.min(1-(a/an)@(bmat/bn).T,axis=1)
                diversity=rankdata(closest,method="average")/len(remaining)
            score=.5*uncertainty+.5*diversity
            chosen=int(remaining[np.lexsort((tie,-score))[0]])
        # Only the selected candidate's label is acquired here.
        support.append(chosen)
        fits[step]=fit_residual(m0[support],z[support],y[support],lam)
    return support,fits


def evaluate_target(payload, row, lambda_result, scaler, fold, sid, epoch, threshold):
    target=payload["val"][sid]
    z=scaler.transform(target["R4"])
    m0=margin_for(target,threshold)
    verify_b0(target,threshold,m0)
    y=target["y"]  # The private vault is indexed only for explicitly acquired support or after predictions freeze.
    lam=lambda_result["selected_lambda"]
    records=[]
    split_audit=[]
    pending=[]
    for rep in range(20):
        candidate,query=split_indices(len(y),42,fold,sid,rep)
        candidates=set(int(c) for c in candidate)
        if len(candidates & set(int(q) for q in query)) or len(candidate)+len(query)!=len(y):
            raise RuntimeError("Calibration/query leakage")
        score_specs=[]
        random_order=random_support(candidate,fold,sid,rep)
        active={p:active_path(p,candidate,m0,z,y,lam,fold,sid,rep)
                for p in ("UNCERTAINTY","UNCERTAINTY_DIVERSITY")}
        for policy in POLICIES:
            for budget in BUDGETS:
                if budget>len(candidate) or (policy=="ORACLE_BALANCED_RANDOM" and budget==1):
                    continue
                if policy=="RANDOM":
                    support=random_order[:budget]
                    b,w=fit_residual(m0[support],z[support],y[support],lam) if budget else (0.,np.zeros(64))
                elif policy in active:
                    path,fits=active[policy]
                    support=path[:budget]
                    b,w=fits[budget]
                else:
                    support=oracle_balanced_support(candidate,y,fold,sid,rep,budget) if budget else []
                    b,w=fit_residual(m0[support],z[support],y[support],lam) if budget else (0.,np.zeros(64))
                bm,bw=fit_residual(m0[support],z[support],y[support],lam,bias_only=True) if budget else (0.,np.zeros(64))
                for calibration_type,delta_b,delta_w in (("FULL_RESIDUAL",b,w),("BIAS_ONLY",bm,bw)):
                    score_specs.append((policy,calibration_type,budget,list(support),delta_b,delta_w,
                                        m0[query]+delta_b+z[query]@delta_w))
        full_b,full_w=fit_residual(m0[candidate],z[candidate],y[candidate],lam)
        score_specs.append(("FULL_POOL_CALIBRATION","FULL_RESIDUAL",len(candidate),list(candidate),full_b,full_w,
                            m0[query]+full_b+z[query]@full_w))
        pending.append((rep,candidate,query,score_specs))
        print(f"[PLAN] fold={fold} done={rep+1}/20",flush=True)
    # All 20 repetitions' support trajectories and every fixed-query score are frozen.
    # Target query labels and nondeployable full-patient oracle are accessed only now.
    oracle_raw=oracle_direction(target["R4"],y)
    oracle_z=oracle_raw*scaler.scale_ if oracle_raw is not None else None
    if oracle_z is not None: oracle_z/=np.linalg.norm(oracle_z)
    for rep,candidate,query,score_specs in pending:
        query_y=y[query]
        baseline=query_metrics(query_y,m0[query])
        full_pool_ap=query_metrics(query_y,score_specs[-1][-1])["ap"]
        split_audit.append({"repetition":rep,"valid_channels":len(y),"candidate_n":len(candidate),"query_n":len(query),
                            "query_estimable":bool(np.isfinite(baseline["ap"])),"available_budgets":[b for b in BUDGETS if b<=len(candidate)]})
        for policy,kind,budget,support,delta_b,delta_w,margin in score_specs:
            metric=query_metrics(query_y,margin)
            support_y=y[support]
            norm=float(np.linalg.norm(delta_w))
            alignment=float((delta_w@oracle_z)/norm) if norm>1e-12 and oracle_z is not None else np.nan
            records.append({"subject_id":sid,"fold":fold,"epoch":epoch,"threshold":threshold,"repetition":rep,
                            "acquisition_policy":policy,"calibration_type":kind,"budget":budget,
                            "candidate_n":len(candidate),"query_n":len(query),"query_estimable":bool(np.isfinite(metric["ap"])),
                            "support_ez":int(support_y.sum()),"support_nez":int(len(support_y)-support_y.sum()),
                            "support_both":bool(len(np.unique(support_y))==2),"delta_w_norm":norm,
                            "oracle_direction_cosine":alignment,"delta_b":delta_b,"lambda_w":lam,
                            **metric,**{f"a1_{k}":v for k,v in baseline.items() if k in metric},
                            "full_pool_ap":full_pool_ap})
        print(f"[EVAL] fold={fold} done={rep+1}/20",flush=True)
    return {"lock_sha":LOCK_SHA,"subject_id":sid,"fold":fold,"epoch":epoch,"threshold":threshold,
            "lambda_result":lambda_result,"split_audit":split_audit,"records":records,
            "zero_budget_exact":True,"target_scores_frozen_before_query_labels":True}


def run_fold(fold, max_new_cells=None):
    folder=RUNTIME/"private"/f"fold_{fold}"
    folder.mkdir(parents=True,exist_ok=True)
    selected=read_csv(PRIOR_RUNTIME/"private"/f"fold_{fold}"/"A1_VLOO_PRIVATE.csv")
    if len(selected)!=13 or {int(r["fold"]) for r in selected}!={fold}:
        raise RuntimeError("VLOO 13-target structure changed")
    new_cells=0
    for ix,row in enumerate(selected,1):
        sid=row["subject_id"]
        epoch=int(row["selected_epoch"])
        threshold=float(row["selected_threshold"])
        target=folder/(hashlib.sha256(sid.encode()).hexdigest()[:16]+".pkl")
        if target.exists():
            with target.open("rb") as f: prior=pickle.load(f)
            if prior["lock_sha"]!=LOCK_SHA or prior["subject_id"]!=sid or prior["epoch"]!=epoch or prior["threshold"]!=threshold or len(prior["split_audit"])!=20:
                raise RuntimeError("Cell resume provenance failed")
            print(f"[RESUME] fold={fold} done={ix}/13",flush=True)
            continue
        payload=load_representation(fold,epoch)
        if sid not in payload["val"] or len(payload["val"])!=13 or len(payload["fit"])!=(51,51,50,52,51)[fold-1]:
            raise RuntimeError("FIT/validation role mismatch")
        scaler=scaler_for(payload)
        lam=select_lambda(payload,scaler,fold,epoch,threshold)
        cell=evaluate_target(payload,row,lam,scaler,fold,sid,epoch,threshold)
        temp=target.with_suffix(".tmp")
        with temp.open("wb") as f: pickle.dump(cell,f,protocol=5)
        temp.replace(target)
        print(f"[CELL] fold={fold} done={ix}/13 epoch={epoch} lambda={lam['selected_lambda']}",flush=True)
        new_cells+=1
        if max_new_cells is not None and new_cells>=max_new_cells:
            return


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",type=int,choices=range(1,6),required=True)
    parser.add_argument("--max-new-cells",type=int,default=None,
                        help="engineering-only process memory bound; private cell resume preserves protocol")
    args=parser.parse_args()
    preflight(); source_replay_check()
    run_fold(args.fold,args.max_new_cells)


if __name__=="__main__":
    main()
