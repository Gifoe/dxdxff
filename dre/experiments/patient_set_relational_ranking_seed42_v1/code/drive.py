"""Process-isolated/resumable FIT grid, selection lock, then full-FIT readouts."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

import protocol as p

HERE=Path(__file__).resolve().parent
NATIVE={0xC0000005,0xC000001D,0xC0000096,0xC0000409,0x80000003,
        -1073741819,-1073741795,-1073741674,-1073740791,-2147483645}


def subprocess_train(fold,context_id,arch,lambda_index,stage,arm=None,epochs=None):
    folder=(p.RUNTIME/stage/f"fold_{fold}"/context_id/arch/
            (f"lambda_{lambda_index}" if stage=="fit" else str(arm)))
    summary=folder/"summary.json"
    if summary.is_file():
        row=json.loads(summary.read_text(encoding="utf-8"))
        if row["lock_sha"]!=p.LOCK_SHA or row["arch"]!=arch or row["context_id"]!=context_id:
            raise RuntimeError("Completed readout summary identity mismatch")
        return
    command=[sys.executable,"-u",str(HERE/"train.py"),"--stage",stage,"--fold",str(fold),
             "--context-id",context_id,"--arch",arch,"--lambda-index",str(lambda_index)]
    if stage=="full":command.extend(("--arm",str(arm),"--epochs",str(epochs)))
    for attempt in range(3):
        status=subprocess.run(command,check=False).returncode
        if status==0:return
        if status not in NATIVE or attempt==2:
            raise RuntimeError(f"Readout training failed status={status}: {' '.join(command)}")
        print(f"[NATIVE_RETRY] fold={fold} {context_id} {arch} lambda={lambda_index} status={status}",flush=True)


def fit_row(fold,ctx,arch,index):
    path=p.RUNTIME/"fit"/f"fold_{fold}"/ctx["context_id"]/arch/f"lambda_{index}"/"summary.json"
    row=json.loads(path.read_text(encoding="utf-8"))
    if row["lock_sha"]!=p.LOCK_SHA or row["context_id"]!=ctx["context_id"]:
        raise RuntimeError("FIT grid summary incomplete or changed")
    return row


def select_arch(fold,arch):
    contexts=p.source_contexts(fold)
    grid={i:[fit_row(fold,c,arch,i) for c in contexts] for i in range(4)}
    aggregates=[]
    for i,rows in grid.items():
        aggregates.append(dict(lambda_index=i,lambda_rank=p.LAMBDA_GRID[i],
                               ap=float(np.mean([r["best_ap"] for r in rows])),
                               mrr=float(np.mean([r["best_mrr"] for r in rows])),
                               top1=float(np.mean([r["best_top1"] for r in rows]))))
    def order(r):return (round(r["ap"],12),round(r["mrr"],12),round(r["top1"],12),-r["lambda_rank"])
    best=max(aggregates,key=order)
    positive=max(aggregates[1:],key=order)
    return dict(fold=fold,arch=arch,n_contexts=len(contexts),
                best_overall_index=best["lambda_index"],best_overall_fit_ap=best["ap"],
                positive_index=positive["lambda_index"],
                positive_fit_ap=positive["ap"],
                bce_fit_ap=aggregates[0]["ap"],grid=aggregates)


def write_csv(path,rows):
    if not rows:raise RuntimeError("Refuse empty FIT selection CSV")
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("w",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def run_fit():
    p.preflight()
    p.all_contexts()
    selections=[]
    for fold in range(1,6):
        for arch in (p.R1,p.R2):
            for ctx in p.source_contexts(fold):
                for index in range(4):
                    subprocess_train(fold,ctx["context_id"],arch,index,"fit")
            selections.append(select_arch(fold,arch))
            print(f"[FIT_SELECT] fold={fold} {arch} positive_lambda={p.LAMBDA_GRID[selections[-1]['positive_index']]} FIT_AP={selections[-1]['positive_fit_ap']:.6f}",flush=True)
    r1={r["fold"]:r for r in selections if r["arch"]==p.R1}
    r2={r["fold"]:r for r in selections if r["arch"]==p.R2}
    diff=[r2[f]["best_overall_fit_ap"]-r1[f]["best_overall_fit_ap"] for f in range(1,6)]
    run_r3=bool(float(np.mean(diff))>=.01 and sum(v>0 for v in diff)>=4)
    print(f"[R3_FIT_GATE] run={run_r3} mean_R2_minus_R1={np.mean(diff):+.6f} positive_folds={sum(v>0 for v in diff)}/5",flush=True)
    if run_r3:
        for fold in range(1,6):
            for ctx in p.source_contexts(fold):
                for index in range(4):
                    subprocess_train(fold,ctx["context_id"],p.R3,index,"fit")
            selections.append(select_arch(fold,p.R3))
    candidates=[]
    for arch in ((p.R1,p.R2,p.R3) if run_r3 else (p.R1,p.R2)):
        for arm in p.ARMS:
            metric=float(np.mean([r["bce_fit_ap"] if arm=="BCE_ONLY" else r["positive_fit_ap"]
                                  for r in selections if r["arch"]==arch]))
            candidates.append(dict(arch=arch,arm=arm,fit_ap=metric))
    # Global primary is fixed using FIT data before any new target labels.
    primary=max(candidates,key=lambda r:(round(r["fit_ap"],12),-((p.R1,p.R2,p.R3).index(r["arch"])),
                                         1 if r["arm"]=="BCE_ONLY" else 0))
    lock=dict(lock_sha=p.LOCK_SHA,fit_grid_complete=True,folds=5,n_contexts=17,
              r3_gate=dict(run=run_r3,fold_deltas=diff,mean_delta=float(np.mean(diff)),
                           positive_folds=sum(v>0 for v in diff)),
              selections=selections,candidates=candidates,primary_candidate=primary,
              target_outcomes_not_accessed_by_this_stage=True)
    path=p.ROOT/"FIT_SELECTION_LOCK.json"
    if path.is_file() and json.loads(path.read_text(encoding="utf-8"))!=lock:
        raise RuntimeError("Frozen FIT selection changed")
    p.atomic_json(path,lock)
    rows=[]
    for selected in selections:
        fold=selected["fold"];arch=selected["arch"]
        for ctx in p.source_contexts(fold):
            for index in range(4):
                row=fit_row(fold,ctx,arch,index)
                rows.append(dict(fold=fold,context_id=ctx["context_id"],source_epoch=ctx["epoch"],
                                 source_threshold=ctx["threshold"],arch=arch,
                                 lambda_rank=p.LAMBDA_GRID[index],meta_val_ap=row["best_ap"],
                                 meta_val_mrr=row["best_mrr"],meta_val_top1=row["best_top1"],
                                 selected_epoch=row["selected_epoch"],
                                 selected_positive=index==selected["positive_index"],
                                 selected_bce=index==0))
    write_csv(p.ROOT/"FIT_HYPERPARAM_SELECTION.csv",rows)
    print(f"[FIT_SELECTION_FROZEN] contexts=17 fit_grid_rows={len(rows)} primary={primary['arch']} {primary['arm']}",flush=True)


def run_full():
    p.preflight()
    path=p.ROOT/"FIT_SELECTION_LOCK.json"
    if not path.is_file():raise RuntimeError("FIT selection must be frozen first")
    lock=json.loads(path.read_text(encoding="utf-8"))
    if lock["lock_sha"]!=p.LOCK_SHA or not lock["fit_grid_complete"]:
        raise RuntimeError("FIT selection lock invalid")
    frozen=p.RUNTIME/"FIT_SELECTION_LOCK_SHA.json"
    check=dict(lock_sha=p.LOCK_SHA,selection_sha=p.sha(path))
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8"))!=check:
        raise RuntimeError("FIT selections changed after full-FIT start")
    p.atomic_json(frozen,check)
    for choice in lock["selections"]:
        fold=choice["fold"];arch=choice["arch"]
        for ctx in p.source_contexts(fold):
            for arm,index in (("BCE_ONLY",0),("RANK_SELECTED",choice["positive_index"])):
                epochs=fit_row(fold,ctx,arch,index)["selected_epoch"]
                subprocess_train(fold,ctx["context_id"],arch,index,"full",arm,epochs)
    print("[ALL_FULL_FIT_RELATIONAL_READOUTS_COMPLETE]",flush=True)


def main():
    q=argparse.ArgumentParser();q.add_argument("--stage",choices=("fit","full"),required=True)
    a=q.parse_args()
    if a.stage=="fit":run_fit()
    else:run_full()


if __name__=="__main__":main()
