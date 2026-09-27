"""Frozen FIT-selected teachers on 65 VLOO cells; all 20 query scores freeze before labels."""
from __future__ import annotations

import argparse
import hashlib
import pickle

import numpy as np

import teacher_core as tc
from fit_select import context_key
from prototype_select import prototype_score


def load_choices(ctx):
    key=context_key(ctx)
    with (tc.RUNTIME/"private"/"fit_selection"/(key+".pkl")).open("rb") as f:sel=pickle.load(f)
    if sel["lock_sha"]!=tc.LOCK_SHA:raise RuntimeError("FIT selection changed")
    fold=ctx["fold"]
    metas={}
    for d in tc.DIMS:
        with (tc.RUNTIME/"private"/"meta"/f"fold_{fold}_d{d}.pkl").open("rb") as f:metas[d]=pickle.load(f)
        if metas[d]["lock_sha"]!=tc.LOCK_SHA:raise RuntimeError("FIT meta projection changed")
    with (tc.RUNTIME/"private"/"meta"/f"fold_{fold}_prototype.pkl").open("rb") as f:proto=pickle.load(f)
    if proto["lock_sha"]!=tc.LOCK_SHA:raise RuntimeError("FIT prototype selection changed")
    return sel,metas,proto


def score_variant(ep,name,basis,lw,lb,full=False,support=None):
    if support is None:
        return tc.prediction(ep,basis,lw,lb,full)
    r=ep["z"]@basis.T
    b,w=tc.fit_head(ep["m0"][support],r[support],ep["y"][support],lw,lb)
    return ep["m0"][ep["query"]]+b+r[ep["query"]]@w,b,basis.T@w


def cell_run(fold,cell_index):
    tc.preflight()
    rows=tc.selected_rows(fold)
    if cell_index<0 or cell_index>=len(rows):raise RuntimeError("Cell index out of range")
    row=rows[cell_index];sid=row["subject_id"]
    stem=hashlib.sha256(sid.encode()).hexdigest()[:16]
    path=tc.RUNTIME/"private"/"target"/f"fold_{fold}"/(stem+".pkl")
    if path.exists():
        with path.open("rb") as f:done=pickle.load(f)
        if done["lock_sha"]!=tc.LOCK_SHA or done["sid"]!=sid or len(done["records"])<20*10:
            raise RuntimeError("Target evaluation resume mismatch")
        print(f"[RESUME] fold={fold} cell={cell_index} sid_hash={stem}",flush=True)
        return
    ctx=tc.context_for_row(fold,row)
    selected,metas,proto=load_choices(ctx)
    prior=tc.private_active_cell(fold,sid)
    prior_records={(r["repetition"],r["acquisition_policy"],r["calibration_type"],r["budget"]):r for r in prior["records"]}
    pending=[]
    for rep in range(20):
        ep=tc.target_episode(ctx,sid,rep)
        spec=[]
        def add(name,score,b,w,support,deployable):
            score=np.asarray(score,dtype=np.float64)
            if not np.isfinite(score).all() or len(score)!=len(ep["query"]):raise RuntimeError("Invalid frozen query score")
            spec.append((name,score.copy(),float(b),np.asarray(w,dtype=np.float64).copy(),np.asarray(support,dtype=int).copy(),deployable))
        add("FROZEN_A1",ep["m0"][ep["query"]],0.,np.zeros(64),np.asarray([],dtype=int),True)
        old=ctx["original_lambda"]
        s=ep["support"];c=ep["candidate"]
        score,b,w=score_variant(ep,"CURRENT_64D_B8",np.eye(64),old,1.)
        add("CURRENT_64D_B8",score,b,w,s,True)
        choice=selected["variants"]["R64"]
        score,b,w=score_variant(ep,"RETUNED_64D_B8",np.eye(64),choice["b8"]["lambda_w"],choice["b8"]["lambda_b"])
        add("RETUNED_64D_B8",score,b,w,s,True)
        score,b,w=score_variant(ep,"CURRENT_64D_FULLPOOL",np.eye(64),old,1.,True)
        add("CURRENT_64D_FULLPOOL",score,b,w,c,False)
        score,b,w=score_variant(ep,"RETUNED_64D_FULLPOOL",np.eye(64),choice["fullpool"]["lambda_w"],choice["fullpool"]["lambda_b"],True)
        add("RETUNED_64D_FULLPOOL",score,b,w,c,False)
        for d in tc.DIMS:
            item=selected["variants"][f"PCA{d}"]
            p=item["basis"]
            lam=item["b8"]
            score,b,w=score_variant(ep,f"PCA_D{d}_B8",p,lam["lambda_w"],lam["lambda_b"])
            add(f"PCA_D{d}_B8",score,b,w,s,True)
            lam=item["fullpool"]
            score,b,w=score_variant(ep,f"PCA_D{d}_FULLPOOL",p,lam["lambda_w"],lam["lambda_b"],True)
            add(f"PCA_D{d}_FULLPOOL",score,b,w,c,False)
            p=metas[d]["projection"].astype(np.float64)
            lam=item["b8"]
            score,b,w=score_variant(ep,f"META_D{d}_B8",p,lam["lambda_w"],lam["lambda_b"])
            add(f"META_D{d}_B8",score,b,w,s,True)
            lam=item["fullpool"]
            score,b,w=score_variant(ep,f"META_D{d}_FULLPOOL",p,lam["lambda_w"],lam["lambda_b"],True)
            add(f"META_D{d}_FULLPOOL",score,b,w,c,False)
        p=proto["projection"]
        score,fallback=prototype_score(ep,p,proto["gamma"])
        y=ep["y"][s]
        if fallback:direction=np.zeros(64)
        else:
            z=ep["z"]@p.T
            ez=z[s[y==1]].mean(axis=0);nez=z[s[y==0]].mean(axis=0)
            ez/=max(np.linalg.norm(ez),1e-12);nez/=max(np.linalg.norm(nez),1e-12)
            direction=p.T@(ez-nez)
        add("PROTOTYPE_B8",score,0.,direction,s,True)
        oracle_s=tc.afr.oracle_balanced_support(c,ep["y"],fold,sid,rep,8)
        if oracle_s is not None and len(oracle_s)==8:
            score,b,w=score_variant(ep,"ORACLE_BALANCED_B8",np.eye(64),choice["b8"]["lambda_w"],choice["b8"]["lambda_b"],support=oracle_s)
            add("ORACLE_BALANCED_B8",score,b,w,oracle_s,False)
        pending.append((rep,ep,spec,bool(fallback)))
    # No target query labels or target all-label oracle touched above. The full
    # collection of every score vector for all 20 fixed-query repetitions is frozen.
    first=pending[0][1]
    scaler=tc.afr.scaler_for(ctx["payload"])
    oracle_raw=tc.afc.oracle_direction(first["r4"],first["y"])
    oracle_z=oracle_raw*scaler.scale_ if oracle_raw is not None else None
    if oracle_z is not None:oracle_z/=max(np.linalg.norm(oracle_z),1e-12)
    records=[];max_replay=0.
    for rep,ep,spec,fallback in pending:
        qy=ep["y"][ep["query"]]
        for name,score,b,w,support,deployable in spec:
            metrics=tc.afc.query_metrics(qy,score)
            norm=float(np.linalg.norm(w))
            cosine=float(w@oracle_z/norm) if oracle_z is not None and norm>1e-12 else np.nan
            sy=ep["y"][support]
            rec={"sid":sid,"fold":fold,"repetition":rep,"variant":name,
                 "support_n":len(support),"support_ez":int(sy.sum()),"support_nez":int(len(sy)-sy.sum()),
                 "one_class_support":len(support)>0 and len(np.unique(sy))==1,
                 "prototype_fallback":bool(fallback and name=="PROTOTYPE_B8"),
                 "deployable_b8":deployable,"direction_cosine":cosine,"direction_norm":norm,"bias":b,**metrics}
            records.append(rec)
            if name in ("CURRENT_64D_B8","CURRENT_64D_FULLPOOL"):
                k=(rep,"UNCERTAINTY" if name.endswith("B8") else "FULL_POOL_CALIBRATION","FULL_RESIDUAL",8 if name.endswith("B8") else len(ep["candidate"]))
                if k not in prior_records:raise RuntimeError(f"Previous reference row missing: {k}")
                old_ap=prior_records[k]["ap"]
                max_replay=max(max_replay,abs(float(metrics["ap"])-float(old_ap)) if np.isfinite(metrics["ap"]) else 0.)
    if max_replay>1e-8:raise RuntimeError(f"Current B8/FULLPOOL replay disagreement {max_replay}")
    tc.write_private(path,{"lock_sha":tc.LOCK_SHA,"fold":fold,"sid":sid,"records":records,
                           "score_vectors_frozen_before_query_labels":True,"prior_replay_max_ap_error":max_replay,
                           "all_target_query_predictions_completed":True})
    print(f"[TARGET] fold={fold} cell={cell_index} reps=20 records={len(records)} replay_error={max_replay:.3g}",flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fold",type=int,required=True)
    parser.add_argument("--cell",type=int,required=True)
    args=parser.parse_args();cell_run(args.fold,args.cell)


if __name__=="__main__":main()
