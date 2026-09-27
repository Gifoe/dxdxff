"""Fold-specific FIT-only episodic meta projection; no target-patient query data."""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score

import teacher_core as tc
from fit_select import context_key


def fold_pca(contexts,d):
    # Exact FIT-only covariance across all selected checkpoint contexts; no val rows.
    count=0; mean=np.zeros(64); second=np.zeros((64,64))
    for ctx in contexts:
        scaler=tc.afr.scaler_for(ctx["payload"])
        for sid in sorted(ctx["payload"]["fit"]):
            z=np.asarray(scaler.transform(ctx["payload"]["fit"][sid]["R4"]),dtype=np.float64)
            count+=len(z); mean+=z.sum(axis=0); second+=z.T@z
    mean/=count
    cov=second/count-np.outer(mean,mean)
    _,vec=np.linalg.eigh(cov)
    p=vec[:,-d:][:,::-1].T.copy()
    for row in p:
        j=np.argmax(np.abs(row));row*=np.sign(row[j])
    return p


def prepared_path(fold):
    return tc.RUNTIME/"private"/tc.META_FOLDER/f"fold_{fold}_episodes.pkl"


def prepare(fold):
    path=prepared_path(fold)
    if path.exists():
        with path.open("rb") as f: obj=pickle.load(f)
        if obj["lock_sha"]!=tc.LOCK_SHA or obj["fold"]!=fold:raise RuntimeError("Meta resume provenance mismatch")
        return obj
    contexts=tc.fold_contexts(fold)
    train=[]; val=[]
    for ctx in contexts:
        with (tc.RUNTIME/"private"/tc.FIT_SELECTION_FOLDER/(context_key(ctx)+".pkl")).open("rb") as f:
            selected=pickle.load(f)
        if selected["lock_sha"]!=tc.LOCK_SHA:raise RuntimeError("FIT lambda selection not locked")
        for sid in sorted(ctx["payload"]["fit"]):
            dest=val if tc.afc.stable_seed(42,fold,sid,"meta_split")%5==0 else train
            reps=2 if dest is val else 3
            for rep in range(reps):
                ep=tc.fit_episode(ctx,sid,rep,"teacher_fit_meta")
                if len(ep["candidate"])<8:continue
                # Only FIT labels are included in this private cache.
                dest.append({"sid":sid,"context":context_key(ctx),"z":ep["z"],"m0":ep["m0"],
                             "y":ep["y"],"support":ep["support"],"query":ep["query"],
                             "lambda_by_dim":{d:selected["variants"][f"PCA{d}"]["b8"] for d in tc.DIMS}})
    if not train or not val or set(e["sid"] for e in train)&set(e["sid"] for e in val):
        raise RuntimeError("FIT patient-disjoint meta split invalid")
    obj={"lock_sha":tc.LOCK_SHA,"fold":fold,"train":train,"val":val,
         "pca":{d:fold_pca(contexts,d) for d in tc.DIMS},
         "train_patients":len(set(e["sid"] for e in train)),"val_patients":len(set(e["sid"] for e in val))}
    tc.write_private(path,obj)
    return obj


def pack(batch,d,device):
    n=len(batch); qmax=max(len(e["query"]) for e in batch)
    sz=np.zeros((n,8,64),np.float32);sm=np.zeros((n,8),np.float32);sy=np.zeros((n,8),np.float32)
    qz=np.zeros((n,qmax,64),np.float32);qm=np.zeros((n,qmax),np.float32)
    qy=np.zeros((n,qmax),np.float32);qmask=np.zeros((n,qmax),np.float32)
    lw=np.zeros(n,np.float32);lb=np.zeros(n,np.float32)
    for i,e in enumerate(batch):
        s=e["support"];q=e["query"];j=len(q)
        sz[i]=e["z"][s];sm[i]=e["m0"][s];sy[i]=e["y"][s]
        qz[i,:j]=e["z"][q];qm[i,:j]=e["m0"][q];qy[i,:j]=e["y"][q];qmask[i,:j]=1
        lw[i]=e["lambda_by_dim"][d]["lambda_w"]
        lb[i]=e["lambda_by_dim"][d]["lambda_b"]
    cast=lambda a:torch.from_numpy(a).to(device)
    return tuple(map(cast,(sz,sm,sy,qz,qm,qy,qmask,lw,lb)))


def unrolled_scores(P,batch):
    sz,sm,sy,qz,qm,qy,qmask,lw,lb=batch
    r=sz@P.T
    n,_,d=r.shape
    b=torch.zeros(n,device=P.device,dtype=P.dtype)
    w=torch.zeros((n,d),device=P.device,dtype=P.dtype)
    # A conservative FIT-support Hessian bound, differentiated through P.
    eta_w=.5/(2*lw+.25*torch.mean(torch.sum(r*r,dim=2),dim=1)+1e-6)
    eta_b=.5/(2*lb+.25)
    for _ in range(64):
        m=sm+b[:,None]+torch.sum(r*w[:,None,:],dim=2)
        e=torch.sigmoid(m)-sy
        b=b-eta_b*(e.mean(dim=1)+2*lb*b)
        w=w-eta_w[:,None]*(torch.mean(e[:,:,None]*r,dim=1)+2*lw[:,None]*w)
    score=qm+b[:,None]+torch.sum((qz@P.T)*w[:,None,:],dim=2)
    return score,qy,qmask


def outer_loss(score,qy,qmask):
    bce=(F.binary_cross_entropy_with_logits(score,qy,reduction="none")*qmask).sum(dim=1)/qmask.sum(dim=1)
    ranking=[]
    for i in range(len(score)):
        pos=torch.where((qy[i]>0.5)&(qmask[i]>0))[0]
        neg=torch.where((qy[i]<0.5)&(qmask[i]>0))[0]
        if len(pos) and len(neg):
            pairs=torch.cartesian_prod(pos,neg)[:128]
            ranking.append(F.softplus(score[i,pairs[:,1]]-score[i,pairs[:,0]]).mean())
        else:ranking.append(score[i].sum()*0)
    return (bce+.1*torch.stack(ranking)).mean()


def validation_ap(P,eps,d):
    # Evaluate exactly the locked final support-only convex head, not unrolled proxy.
    p=P.detach().cpu().numpy().astype(np.float64)
    by_patient={}
    for e in eps:
        s=e["support"];q=e["query"]
        z=e["z"]@p.T;lam=e["lambda_by_dim"][d]
        b,w=tc.fit_head(e["m0"][s],z[s],e["y"][s],lam["lambda_w"],lam["lambda_b"])
        y=e["y"][q]
        if len(np.unique(y))<2:continue
        score=e["m0"][q]+b+z[q]@w
        by_patient.setdefault(e["sid"],[]).append(float(average_precision_score(y,score)))
    if not by_patient:raise RuntimeError("No estimable FIT-meta-val episodes")
    return float(np.mean([np.mean(v) for v in by_patient.values()])),len(by_patient)


def train(fold,d):
    tc.preflight()
    obj=prepare(fold)
    path=tc.RUNTIME/"private"/tc.META_FOLDER/f"fold_{fold}_d{d}.pkl"
    progress_path=tc.RUNTIME/"private"/tc.META_FOLDER/f"fold_{fold}_d{d}_progress.pkl"
    if path.exists():
        with path.open("rb") as f: result=pickle.load(f)
        if result["lock_sha"]!=tc.LOCK_SHA:raise RuntimeError("Meta projection resume mismatch")
        print(f"[RESUME] fold={fold} d={d} FIT-val AP={result['fit_val_ap']:.5f}",flush=True)
        return result
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(tc.afc.stable_seed(42,fold,d,"projection"))
    if device.type=="cuda":torch.cuda.manual_seed_all(tc.afc.stable_seed(42,fold,d,"projection"))
    init=torch.tensor(obj["pca"][d].copy(),device=device,dtype=torch.float32)
    P=torch.nn.Parameter(init.clone())
    opt=torch.optim.AdamW([P],lr=.001,weight_decay=.0001)
    train_eps=obj["train"];val_eps=obj["val"]
    best_ap=-np.inf;best_p=init.cpu().numpy().copy();best_epoch=0;wait=0;history=[];start_epoch=1
    if progress_path.exists():
        with progress_path.open("rb") as f:state=pickle.load(f)
        if state["lock_sha"]!=tc.LOCK_SHA or state["fold"]!=fold or state["dimension"]!=d:
            raise RuntimeError("Meta epoch resume provenance mismatch")
        with torch.no_grad():P.copy_(torch.as_tensor(state["current_projection"],device=device))
        opt.load_state_dict(state["optimizer"])
        best_ap=state["best_ap"];best_p=state["best_projection"]
        best_epoch=state["best_epoch"];wait=state["wait"];history=state["history"]
        start_epoch=state["last_epoch"]+1
        if wait>=5:start_epoch=31
    for epoch in range(start_epoch,31):
        order=np.random.default_rng(tc.afc.stable_seed(42,fold,d,epoch,"meta_order")).permutation(len(train_eps))
        total=0.0
        for start in range(0,len(order),32):
            batch=pack([train_eps[int(i)] for i in order[start:start+32]],d,device)
            opt.zero_grad(set_to_none=True)
            score,qy,mask=unrolled_scores(P,batch)
            loss=outer_loss(score,qy,mask)+.01*torch.mean((P-init)**2)
            loss.backward();torch.nn.utils.clip_grad_norm_([P],5)
            opt.step()
            with torch.no_grad():P[:]=F.normalize(P,dim=1)
            total+=float(loss.detach().cpu())
        ap,nval=validation_ap(P,val_eps,d)
        history.append({"epoch":epoch,"fit_val_patient_equal_ap":ap,"training_batch_loss_sum":total})
        print(f"[META] f{fold} d{d} epoch={epoch} AP={ap:.5f} val_patients={nval}",flush=True)
        if ap>best_ap+1e-6:
            best_ap=ap;best_p=P.detach().cpu().numpy().copy();best_epoch=epoch;wait=0
        else:wait+=1
        tc.write_private(progress_path,{"lock_sha":tc.LOCK_SHA,"fold":fold,"dimension":d,
            "last_epoch":epoch,"current_projection":P.detach().cpu().numpy().copy(),
            "optimizer":opt.state_dict(),"best_ap":best_ap,"best_projection":best_p,
            "best_epoch":best_epoch,"wait":wait,"history":history})
        if wait>=5:break
    result={"lock_sha":tc.LOCK_SHA,"fold":fold,"dimension":d,"projection":best_p,
            "fit_val_ap":best_ap,"selected_epoch":best_epoch,"history":history,
            "train_patients":obj["train_patients"],"val_patients":obj["val_patients"],
            "train_episodes":len(train_eps),"val_episodes":len(val_eps),"device":device.type}
    tc.write_private(path,result)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--fold",type=int,required=True)
    parser.add_argument("--dimension",type=int,required=True,choices=tc.DIMS)
    args=parser.parse_args();train(args.fold,args.dimension)


if __name__=="__main__":main()
