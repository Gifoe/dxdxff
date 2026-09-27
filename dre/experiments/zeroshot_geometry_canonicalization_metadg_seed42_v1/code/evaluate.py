"""Freeze all new-model scores before any new target-label evaluation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import pickle
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

from train import LOCK_SHA, PROJECT, ROOT, RUNTIME, A1_RUNTIME, VARIANTS, preflight, selected_config, write_json
sys.path.insert(0, str(PROJECT / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import epoch_grid, finalize_fold  # noqa: E402
sys.path.insert(0, str(PROJECT / "active_fewshot_patient_calibration_seed42_v1" / "code"))
import common as afc  # noqa: E402
import run as afr  # noqa: E402


@lru_cache(maxsize=16)
def private_snapshot(fold, variant, epoch):
    index = selected_config(fold, variant)["config_index"]
    path = RUNTIME / "full" / f"fold_{fold}" / variant / f"config_{index:02d}" / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"
    with path.open("rb") as f: return pickle.load(f)


def source_epoch30_snapshot(fold):
    """One exact A1 checkpoint per fold for valid cross-patient R4 geometry."""
    path=RUNTIME/"private"/"a1_common_checkpoint"/f"fold_{fold}_epoch30.pkl"
    if path.is_file():return path
    import torch
    import train as tr
    from run_matched import build_fold, install_interleaved_hlv_view, make_args
    import exp_ez_hybrid as core
    if not getattr(source_epoch30_snapshot,"_installed",False):
        install_interleaved_hlv_view();source_epoch30_snapshot._installed=True
    args=make_args("R0",RUNTIME/"diagnostic_source")
    exp=core.Exp_EZHybridLocalization(args)
    split=next(s for s in exp.outer_splits if int(s["fold_idx"])==fold)
    _,_,train_loader,val_loader,_,_=build_fold(exp,split,"validation")
    model=exp.runtime["model_cls"](args).to(exp.device)
    exp._dry_initialize_lazy_layers(model,train_loader)
    ckpt=torch.load(A1_RUNTIME/"A1"/f"fold_{fold}"/"epoch_30.pt",
                    map_location=exp.device,weights_only=False)
    if ckpt["epoch"]!=30 or ckpt["variant"]!="A1" or ckpt["fold"]!=fold:
        raise RuntimeError("Source A1 epoch-30 identity mismatch")
    model.load_state_dict(ckpt["model_state_dict"],strict=True)
    capture=tr.CaptureR4(model)
    snapshot=tr.score_only_snapshot(exp,model,capture,val_loader)
    capture.close();path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(".tmp")
    with temp.open("wb") as f:pickle.dump(snapshot,f,protocol=5)
    temp.replace(path)
    return path


def freeze():
    """Read metadata and bytes of score files, not target labels/outcomes."""
    preflight()
    files = {}; selections = {}; source_common = {}
    for fold in range(1, 6):
        for variant in VARIANTS:
            selection_path=RUNTIME/"selection"/f"fold_{fold}"/f"{variant}.json"
            selections[str(selection_path.relative_to(RUNTIME))]=afc.sha(selection_path)
            index = selected_config(fold, variant)["config_index"]
            folder = RUNTIME / "full" / f"fold_{fold}" / variant / f"config_{index:02d}"
            summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
            if summary["lock_sha"] != LOCK_SHA or summary["checkpoints"] != 30 or not summary["validation_scores_frozen"]:
                raise RuntimeError("Full-FIT score grid incomplete")
            for epoch in range(1, 31):
                path = folder / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"
                digest = afc.sha(path)
                if digest != summary["score_snapshot_hashes"][epoch-1]:
                    raise RuntimeError("Score freeze digest mismatch")
                files[str(path.relative_to(RUNTIME))] = digest
        source_path=source_epoch30_snapshot(fold)
        source_common[str(source_path.relative_to(RUNTIME))]=afc.sha(source_path)
    amendment=ROOT/"DIAGNOSTIC_AMENDMENT.json"
    amendment_sha=afc.sha(amendment)
    if amendment_sha!="765107a265c48d139430c97d95da2aeb513dbadff6a639984ff2b04afa69e0bf":
        raise RuntimeError("Common-checkpoint diagnostic amendment changed")
    manifest = dict(lock_sha=LOCK_SHA, score_files=len(files), fit_selection_files=len(selections),
                    source_common_files=len(source_common),diagnostic_amendment_sha=amendment_sha,
                    model_variants=len(VARIANTS), folds=5, epochs=30,
                    target_labels_not_read_in_this_command=True, files=files,
                    selections=selections, source_common=source_common)
    path = RUNTIME / "SCORE_FREEZE_BEFORE_TARGET_LABELS.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior != manifest: raise RuntimeError("Frozen score manifest changed")
    else:
        write_json(path, manifest)
    print(f"SCORE_FREEZE_PASS files={len(files)}", flush=True)


@lru_cache(maxsize=16)
def _source_payload(fold, epoch):
    return afc.load_representation(fold, epoch)


@lru_cache(maxsize=5)
def _source_selected_epochs(fold):
    rows=afc.read_csv(afc.PRIOR_RUNTIME/"private"/f"fold_{fold}"/"A1_VLOO_PRIVATE.csv")
    if len(rows)!=13:raise RuntimeError("Source selected grid incomplete")
    return {r["subject_id"]:int(r["selected_epoch"]) for r in rows}


def reference_y(fold, epoch, sid):
    # Prior exact-R4 extraction materialized only the distinct A1-selected
    # epochs, not all 30. Channel order and labels are epoch-invariant, so
    # retrieve labels from that patient's source-selected epoch.
    selected=_source_selected_epochs(fold)[sid]
    payload=_source_payload(fold,selected)
    return np.asarray(payload["val"][sid]["y"],dtype=np.int8)


def make_grid(fold, variant):
    """All 30 score grids were frozen before this label-using VLOO selection."""
    payloads = []
    for epoch in range(1, 31):
        snap = private_snapshot(fold, variant, epoch)
        records = []
        for sid in sorted(snap):
            row = snap[sid]
            y = reference_y(fold, epoch, sid)
            if len(y) != row["n_channels"] or set(np.unique(y)) != {0, 1}:
                raise RuntimeError("Source target label order/size mismatch")
            score = np.asarray(row["score_ez"], dtype=np.float64)
            score_nez = np.asarray(row["score_nez"], dtype=np.float64)
            records.append(dict(subject_id=sid, channel_mask=np.ones(len(y), bool),
                                labels_ez=y, labels=1-y, labels_nez=1-y,
                                score_ez=score, score_nez=score_nez))
        payloads.append(epoch_grid(records, epoch))
    private_csv = RUNTIME / "private" / f"fold_{fold}" / f"{variant}_VLOO_PRIVATE.csv"
    public, full = finalize_fold(payloads, variant, fold, private_csv)
    write_json(RUNTIME / "private" / f"fold_{fold}" / f"{variant}_VLOO_SUMMARY.json",
               dict(public=public, full=full))
    return afc.read_csv(private_csv)


def target_cell(fold, variant, selection):
    sid = selection["subject_id"]
    epoch = int(selection["selected_epoch"]); tau = float(selection["selected_threshold"])
    row = private_snapshot(fold, variant, epoch)[sid]
    logit = np.asarray(row["logit_nez"], dtype=np.float64)
    r4 = np.asarray(row["R4"], dtype=np.float64)
    margin = math.log(tau / (1-tau)) - logit
    if not np.isfinite(margin).all() or r4.shape != (len(margin), 64):
        raise RuntimeError("Nonfinite or malformed frozen R4/scores")
    # Construct fixed query indices without candidate features or labels.
    frozen = []
    for rep in range(20):
        candidate, query = afc.split_indices(len(margin), 42, fold, sid, rep)
        if set(candidate) & set(query) or len(candidate) + len(query) != len(margin):
            raise RuntimeError("Fixed query partition invalid")
        frozen.append(dict(rep=rep, query=query, score=margin[query].copy()))
    pending = RUNTIME / "private" / "pending" / variant / f"fold_{fold}" / (hashlib.sha256(sid.encode()).hexdigest()[:16] + ".pkl")
    pending.parent.mkdir(parents=True, exist_ok=True)
    tmp = pending.with_suffix(".tmp")
    with tmp.open("wb") as f: pickle.dump(dict(lock_sha=LOCK_SHA, fold=fold, sid=sid, epoch=epoch,
                                                 tau=tau, r4=r4, frozen=frozen), f, protocol=5)
    tmp.replace(pending)
    # All candidate score grids were frozen before VLOO label use. Exact A1
    # VLOO necessarily accesses this patient's label while selecting other
    # patients; the patient's own label is never in its own selection metric.
    # This is disclosed, rather than mislabeled strict no-target-label access.
    y = reference_y(fold, epoch, sid)
    if len(y) != len(margin): raise RuntimeError("Target channel order mismatch")
    metrics = []
    for item in frozen:
        q = item["query"]
        score = item["score"]
        m = afc.query_metrics(y[q], score)
        metrics.append(dict(rep=item["rep"], **m))
    result = dict(lock_sha=LOCK_SHA, fold=fold, sid=sid, variant=variant,
                  epoch=epoch, threshold=tau, pending_sha=afc.sha(pending), metrics=metrics)
    path = RUNTIME / "private" / "cells" / variant / f"fold_{fold}" / (hashlib.sha256(sid.encode()).hexdigest()[:16] + ".pkl")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as f: pickle.dump(result, f, protocol=5)
    tmp.replace(path)
    return result


def evaluate(fold, variant):
    freeze_path = RUNTIME / "SCORE_FREEZE_BEFORE_TARGET_LABELS.json"
    if not freeze_path.is_file(): raise RuntimeError("All-model score freeze required")
    frozen=json.loads(freeze_path.read_text(encoding="utf-8"))
    if frozen["score_files"] != 900 or frozen["fit_selection_files"] != 30 or frozen["source_common_files"] != 5:
        raise RuntimeError("Incomplete pre-label score freeze")
    index=selected_config(fold,variant)["config_index"]
    folder=RUNTIME/"full"/f"fold_{fold}"/variant/f"config_{index:02d}"
    selection=RUNTIME/"selection"/f"fold_{fold}"/f"{variant}.json"
    if afc.sha(selection)!=frozen["selections"][str(selection.relative_to(RUNTIME))]:
        raise RuntimeError("FIT selection changed after target score freeze")
    for epoch in range(1,31):
        score=folder/f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"
        if afc.sha(score)!=frozen["files"][str(score.relative_to(RUNTIME))]:
            raise RuntimeError("Target score/R4 changed after freeze")
    path = RUNTIME / "private" / f"fold_{fold}" / f"{variant}_VLOO_PRIVATE.csv"
    rows = afc.read_csv(path) if path.exists() else make_grid(fold, variant)
    if len(rows) != 13: raise RuntimeError("Expected 13 VLOO target cells")
    for selection in rows:
        sid = selection["subject_id"]
        dst = RUNTIME / "private" / "cells" / variant / f"fold_{fold}" / (hashlib.sha256(sid.encode()).hexdigest()[:16] + ".pkl")
        if dst.is_file(): continue
        target_cell(fold, variant, selection)
    print(f"[TARGET] {variant} fold={fold} cells=13", flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--stage", choices=("freeze", "evaluate"), required=True)
    p.add_argument("--fold", type=int, choices=range(1,6))
    p.add_argument("--variant", choices=VARIANTS)
    args=p.parse_args(); preflight()
    if args.stage == "freeze": freeze(); return
    if args.fold is None or args.variant is None: p.error("--fold and --variant required")
    evaluate(args.fold,args.variant)


if __name__ == "__main__": main()
