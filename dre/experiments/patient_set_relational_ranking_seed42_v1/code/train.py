"""FIT-only hyperparameter grid and full-FIT frozen-R4 relational readouts."""
from __future__ import annotations

import argparse
import copy
import json

import numpy as np
import torch
from sklearn.metrics import average_precision_score

import protocol as p
from relational import RelationalReadout, patient_equal_loss


def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_model(fold, context_id, arch):
    value = p.afc.stable_seed(42, fold, context_id, arch, "readout_init")
    torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)


def rows_on_device(fit, ids, scaler, threshold, where):
    output = {}
    for sid in ids:
        z, m0, y = p.patient_arrays(fit[sid], scaler, threshold)
        output[sid] = (torch.from_numpy(z).to(where), torch.from_numpy(m0).to(where),
                       torch.from_numpy(y).to(where))
    return output


@torch.no_grad()
def validation(model, patients):
    model.eval()
    aps=[];mrr=[];top=[]
    for sid in sorted(patients):
        z,m0,y=patients[sid]
        score=model(z,m0,self_index=torch.arange(len(z),device=z.device)).detach().cpu().numpy()
        labels=y.detach().cpu().numpy().astype(np.int8)
        if set(np.unique(labels))!={0,1}:continue
        aps.append(float(average_precision_score(labels,score)))
        order=np.argsort(-score,kind="stable")
        mrr.append(1/(int(np.flatnonzero(labels[order]==1)[0])+1))
        top.append(float(labels[order[0]]))
    if not aps:raise RuntimeError("No estimable FIT meta-validation patients")
    return dict(ap=float(np.mean(aps)),mrr=float(np.mean(mrr)),top1=float(np.mean(top)),n=len(aps))


def _cpu_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}


def _path(fold, context_id, arch, lambda_index, stage, arm=None):
    if stage=="fit":
        return p.RUNTIME/"fit"/f"fold_{fold}"/context_id/arch/f"lambda_{lambda_index}"
    return p.RUNTIME/"full"/f"fold_{fold}"/context_id/arch/str(arm)


def train_one(fold, context_id, arch, lambda_index, stage, arm=None, epochs=None):
    p.preflight()
    if arch not in p.ARCHS or lambda_index not in range(len(p.LAMBDA_GRID)) or stage not in {"fit","full"}:
        raise RuntimeError("Invalid readout training request")
    ctx=p.context_by_id(fold,context_id)
    if stage=="full" and (arm not in p.ARMS or epochs is None or epochs not in range(21)):
        raise RuntimeError("Full-FIT requires a locked arm and FIT-selected epoch count")
    folder=_path(fold,context_id,arch,lambda_index,stage,arm)
    folder.mkdir(parents=True,exist_ok=True)
    summary=folder/"summary.json"
    if summary.is_file():
        row=json.loads(summary.read_text(encoding="utf-8"))
        if (row["lock_sha"]!=p.LOCK_SHA or row["fold"]!=fold or row["context_id"]!=context_id or
                row["arch"]!=arch or row["lambda_index"]!=lambda_index or row["stage"]!=stage):
            raise RuntimeError("Completed readout cache identity mismatch")
        print(f"[SKIP] {stage} fold={fold} {context_id} {arch} lambda={lambda_index} arm={arm}",flush=True)
        return
    source=p.payload(fold,ctx["epoch"])
    fit=source["fit"]
    full_ids=sorted(fit)
    if stage=="fit":train_ids,val_ids=p.fit_partition(full_ids,fold)
    else:train_ids,val_ids=full_ids,[]
    scaler=p.scaler_for(fit,train_ids)
    where=device()
    train=rows_on_device(fit,train_ids,scaler,ctx["threshold"],where)
    val=rows_on_device(fit,val_ids,scaler,ctx["threshold"],where) if val_ids else {}
    seed_model(fold,context_id,arch)
    model=RelationalReadout(arch).to(where)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    lambda_rank=p.LAMBDA_GRID[lambda_index]
    best=validation(model,val) if stage=="fit" else None
    best_epoch=0;best_state=_cpu_state(model);wait=0;start=1;last=0
    progress=folder/"progress.pt"
    if progress.is_file():
        state=torch.load(progress,map_location="cpu",weights_only=False)
        if (state["lock_sha"]!=p.LOCK_SHA or state["lambda_index"]!=lambda_index or
                state["arch"]!=arch or state["context_id"]!=context_id or state["stage"]!=stage):
            raise RuntimeError("Readout epoch-resume identity mismatch")
        model.load_state_dict(state["model"]);optimizer.load_state_dict(state["optimizer"])
        best=state["best_metrics"];best_epoch=state["best_epoch"]
        best_state=state["best_state"];wait=state["wait"];last=state["last_epoch"]
        start=last+1
        if stage=="fit" and wait>=5:start=21
    limit=20 if stage=="fit" else int(epochs)
    for epoch in range(start,limit+1):
        model.train()
        order=np.random.default_rng(p.afc.stable_seed(42,fold,context_id,arch,epoch,"patient_order")).permutation(train_ids)
        losses=[]
        for sid in order:
            z,m0,y=train[str(sid)]
            optimizer.zero_grad(set_to_none=True)
            margin=model(z,m0,self_index=torch.arange(len(z),device=where))
            loss,_,_=patient_equal_loss(margin,y,lambda_rank)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        last=epoch
        if stage=="fit":
            score=validation(model,val)
            if (score["ap"]>best["ap"]+1e-9 or
                    (abs(score["ap"]-best["ap"])<=1e-9 and score["mrr"]>best["mrr"]+1e-9) or
                    (abs(score["ap"]-best["ap"])<=1e-9 and abs(score["mrr"]-best["mrr"])<=1e-9 and score["top1"]>best["top1"]+1e-9)):
                best=score;best_epoch=epoch;best_state=_cpu_state(model);wait=0
            else:wait+=1
            print(f"[FIT] fold={fold} {context_id} {arch} lambda={lambda_rank} epoch={epoch} AP={score['ap']:.6f} best={best['ap']:.6f}",flush=True)
        else:
            print(f"[FULL] fold={fold} {context_id} {arch} arm={arm} lambda={lambda_rank} epoch={epoch}/{limit} loss={np.mean(losses):.5f}",flush=True)
        tmp=progress.with_suffix(".tmp")
        torch.save(dict(lock_sha=p.LOCK_SHA,fold=fold,context_id=context_id,arch=arch,
                        stage=stage,lambda_index=lambda_index,last_epoch=epoch,
                        model=_cpu_state(model),optimizer=optimizer.state_dict(),
                        best_metrics=best,best_epoch=best_epoch,best_state=best_state,wait=wait),tmp)
        tmp.replace(progress)
        if stage=="fit" and wait>=5:break
    if stage=="fit":
        selected_state=best_state
        selected_epoch=best_epoch
    else:
        selected_state=_cpu_state(model)
        selected_epoch=int(epochs)
    ckpt=folder/"selected.pt"
    tmp=ckpt.with_suffix(".tmp")
    torch.save(dict(lock_sha=p.LOCK_SHA,fold=fold,context_id=context_id,arch=arch,
                    stage=stage,arm=arm,lambda_index=lambda_index,lambda_rank=lambda_rank,
                    selected_epoch=selected_epoch,model=selected_state,
                    scaler_mean=scaler.mean_,scaler_scale=scaler.scale_,
                    train_ids=tuple(sorted(train_ids)),meta_val_ids=tuple(sorted(val_ids))),tmp)
    tmp.replace(ckpt)
    row=dict(lock_sha=p.LOCK_SHA,fold=fold,context_id=context_id,epoch=ctx["epoch"],
             threshold=ctx["threshold"],arch=arch,stage=stage,arm=arm,
             lambda_index=lambda_index,lambda_rank=lambda_rank,
             selected_epoch=selected_epoch,last_epoch=last,
             best_ap=best["ap"] if best else None,best_mrr=best["mrr"] if best else None,
             best_top1=best["top1"] if best else None,
             n_train=len(train_ids),n_meta_val=len(val_ids),training_all_pairs=True,
             checkpoint_sha=p.sha(ckpt))
    p.atomic_json(summary,row)
    print(f"[DONE] {stage} fold={fold} {context_id} {arch} lambda={lambda_rank} arm={arm}",flush=True)


def main():
    q=argparse.ArgumentParser()
    q.add_argument("--fold",type=int,required=True)
    q.add_argument("--context-id",required=True)
    q.add_argument("--arch",choices=p.ARCHS,required=True)
    q.add_argument("--lambda-index",type=int,choices=range(4),required=True)
    q.add_argument("--stage",choices=("fit","full"),required=True)
    q.add_argument("--arm",choices=p.ARMS)
    q.add_argument("--epochs",type=int)
    a=q.parse_args()
    train_one(a.fold,a.context_id,a.arch,a.lambda_index,a.stage,a.arm,a.epochs)


if __name__=="__main__":main()
