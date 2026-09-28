"""Evaluate only after every score and control is hash-frozen."""
from __future__ import annotations

import hashlib
import json
import pickle

import numpy as np

import protocol as p


def frozen():
    p.preflight()
    path=p.RUNTIME/"ALL_SCORES_FROZEN_BEFORE_NEW_TARGET_LABEL_USE.json"
    row=json.loads(path.read_text(encoding="utf-8"))
    if row["lock_sha"]!=p.LOCK_SHA or row["selection_sha"]!=p.sha(p.ROOT/"FIT_SELECTION_LOCK.json"):
        raise RuntimeError("Relational score/selection freeze invalid")
    if row["target_cells"]!=65 or row["context_repetitions"]!=20:
        raise RuntimeError("Relational score freeze coverage invalid")
    for rel,digest in row["files"].items():
        if p.sha(p.RUNTIME/rel)!=digest:raise RuntimeError("Frozen target score changed: "+rel)
    for rel,digest in row["checkpoints"].items():
        if p.sha(p.RUNTIME/rel)!=digest:raise RuntimeError("Frozen readout changed: "+rel)
    return row


def evaluate_cell(ctx,sid,arch,arm,manifest):
    fold=ctx["fold"]
    stem=hashlib.sha256(sid.encode()).hexdigest()[:16]
    rel=(f"private/scores/fold_{fold}/{ctx['context_id']}/"
         f"{stem}_{arch}_{arm}.pkl")
    if rel not in manifest["files"]:raise RuntimeError("Cell missing from score freeze")
    with (p.RUNTIME/rel).open("rb") as f:score=pickle.load(f)
    if score["sid"]!=sid or score["fold"]!=fold or len(score["reps"])!=20:
        raise RuntimeError("Cell provenance invalid")
    # First indexing of this target's labels in this study. All scores were
    # frozen and hashed before this function is permitted to execute.
    y=np.asarray(p.payload(fold,ctx["epoch"])["val"][sid]["y"],dtype=np.int8)
    if len(y)!=score["n_channels"] or not np.isin(y,[0,1]).all():
        raise RuntimeError("Target label count/encoding mismatch")
    metric_rows=[]
    for item in score["reps"]:
        query=np.asarray(item["query"],dtype=np.int32)
        labels=y[query]
        for name in ("a1","full","query_only","wrong","shuffled"):
            s=np.asarray(item[name],dtype=np.float64)
            metrics=p.afc.query_metrics(labels,s)
            metric_rows.append(dict(fold=fold,sid=sid,rep=item["rep"],arch=arch,arm=arm,
                                    context=name,**metrics))
    return metric_rows,dict(fold=fold,sid=sid,arch=arch,arm=arm,
                            y=y,full_all=score["full_all"],a1_all=score["a1_all"])


def main():
    manifest=frozen()
    rows=[];rank=[]
    for ctx in p.all_contexts():
        for sid in ctx["target_ids"]:
            for arch in manifest["active_architectures"]:
                for arm in p.ARMS:
                    metrics,rank_row=evaluate_cell(ctx,sid,arch,arm,manifest)
                    rows.extend(metrics);rank.append(rank_row)
        print(f"[LABEL_EVALUATED] fold={ctx['fold']} {ctx['context_id']}",flush=True)
    expected=65*len(manifest["active_architectures"])*2*20*5
    if len(rows)!=expected or len({r["sid"] for r in rows})!=47:
        raise RuntimeError("Evaluated cell/ID grid incomplete")
    p.atomic_pickle(p.RUNTIME/"private"/"EVALUATED_METRICS.pkl",rows)
    p.atomic_pickle(p.RUNTIME/"private"/"RANK_CHANGE_INPUTS.pkl",rank)
    p.atomic_json(p.RUNTIME/"EVALUATION_STATUS.json",
                  dict(lock_sha=p.LOCK_SHA,score_manifest_sha=p.sha(p.RUNTIME/"ALL_SCORES_FROZEN_BEFORE_NEW_TARGET_LABEL_USE.json"),
                       metric_rows=len(rows),rank_rows=len(rank),unique_patient_ids=47,
                       target_labels_used_for_metrics_after_score_freeze=True))
    print(f"[EVALUATION_COMPLETE] rows={len(rows)}",flush=True)


if __name__=="__main__":main()
