"""Replay 150 A1 checkpoints; capture exact seizure embedding and R4 privately."""
from __future__ import annotations

import argparse
import json
import math
import pickle

import numpy as np
import torch

from common import (EXPECTED_FOLDS, ROOT, RUNTIME, build_fold, checkpoint, core, epoch_grid,
                    experiment, finalize_fold, preflight, read_csv, source_grid, write_json)


def grid_error(actual, reference):
    if actual["epoch"] != reference["epoch"] or [r["subject_id"] for r in actual["patients"]] != [r["subject_id"] for r in reference["patients"]]:
        raise RuntimeError("A1 epoch/patient grid changed")
    error = 0.0
    for a, b in zip(actual["patients"], reference["patients"], strict=True):
        if a["n_channels"] != b["n_channels"]:
            raise RuntimeError("A1 channel count changed")
        for group in ("grid", "fixed"):
            if set(a[group]) != set(b[group]):
                raise RuntimeError("A1 grid metric schema changed")
            for metric in a[group]:
                error = max(error, float(np.max(np.abs(np.asarray(a[group][metric]) - np.asarray(b[group][metric])))))
    return error


def capture(model, loader, device, patient_index):
    seen = {}

    def agg_hook(_module, inputs):
        seen["agg_in"] = inputs[0].detach()
        seen["seizure_mask"] = inputs[1].detach()
        seen["seizure_channel_mask"] = inputs[2].detach()

    def r4_hook(_module, inputs):
        seen["r4"] = inputs[0].detach()

    agg_handle = model.seizure_aggregator.register_forward_pre_hook(agg_hook)
    r4_handle = model.channel_classifier.classifier.register_forward_pre_hook(r4_hook)
    patients = {}
    max_agg_error = max_r4_error = 0.0
    checked_seizures = 0
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                seen.clear()
                moved = core._move_tensors_to_device(batch, device)
                output = model(moved)
                if set(seen) != {"agg_in", "seizure_mask", "seizure_channel_mask", "r4"}:
                    raise RuntimeError("Exact aggregator/R4 hook missing")
                e = output["seizure_channel_embedding"]
                if e.ndim != 4 or not torch.equal(e, seen["agg_in"]):
                    raise RuntimeError("Captured tensor differs from source seizure_channel_embedding")
                if not torch.equal(seen["seizure_mask"], moved["seizure_mask"]) or not torch.equal(seen["seizure_channel_mask"], moved["seizure_channel_mask"]):
                    raise RuntimeError("Aggregator mask differs from source batch")
                replay_patient, _ = model.seizure_aggregator(e, moved["seizure_mask"], moved["seizure_channel_mask"])
                valid = moved["channel_mask"]
                error = float((replay_patient[valid] - output["patient_channel_embedding"][valid]).abs().max())
                max_agg_error = max(max_agg_error, error)
                if error > 1e-6:
                    raise RuntimeError("Frozen aggregator replay failed")
                r4 = seen["r4"]
                replay_logit = model.channel_classifier.classifier(r4).squeeze(-1)
                err = float((replay_logit[valid] - output["logits"][valid]).abs().max())
                max_r4_error = max(max_r4_error, err)
                if err > 1e-6:
                    raise RuntimeError("Frozen R4 classifier replay failed")
                for i, sid in enumerate(batch["subject_id"]):
                    sid = str(sid)
                    names = list(batch["canonical_channels"][i])
                    if names != list(patient_index[sid]["canonical_channels"]) or len(names) != len(set(names)):
                        raise RuntimeError("Canonical channel name alignment failed")
                    cmask = batch["channel_mask"][i].numpy().astype(bool)
                    smask = batch["seizure_mask"][i].numpy().astype(bool)
                    present = batch["seizure_channel_mask"][i].numpy().astype(bool)[smask][:, cmask]
                    if present.shape[1] != int(cmask.sum()) or not present.any(axis=1).all():
                        raise RuntimeError("Seizure-channel canonical alignment/mask invalid")
                    count = int(smask.sum())
                    checked_seizures += count
                    patients[sid] = {"E": e[i][moved["seizure_mask"][i]][:, moved["channel_mask"][i]].cpu().numpy().astype(np.float32),
                                     "present": present, "R4": r4[i][valid[i]].cpu().numpy().astype(np.float32),
                                     "y": batch["labels_ez"][i].numpy()[cmask].astype(np.int8),
                                     "source_ez": (-output["logits"][i][valid[i]]).cpu().numpy().astype(np.float32),
                                     "n_seizures": count}
                    if set(np.unique(patients[sid]["y"])) != {0, 1}:
                        raise RuntimeError("FIT/validation patient missing class")
    finally:
        agg_handle.remove()
        r4_handle.remove()
    return patients, max_agg_error, max_r4_error, checked_seizures


def run_fold(exp, split):
    fold, train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
    if test_loader is not None:
        raise RuntimeError("Outer loader constructed")
    fit_ids, val_ids = set(split["fit_subjects"]), set(split["validation_subjects"])
    if fit_ids & val_ids or len(fit_ids) != (51,51,50,52,51)[fold-1] or len(val_ids) != 13:
        raise RuntimeError("FIT/validation membership mismatch")
    folder = RUNTIME / "private" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    fit_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
    model = exp.runtime["model_cls"](exp.args).to(exp.device)
    exp._dry_initialize_lazy_layers(model, train_loader)
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    errors = []
    for epoch in range(1,31):
        state = torch.load(checkpoint(fold,epoch), map_location=exp.device, weights_only=False)
        if (state["variant"],state["fold"],state["epoch"]) != ("A1",fold,epoch):
            raise RuntimeError("A1 source checkpoint identity mismatch")
        if not np.array_equal(state["normalizer_mean"], normalizer.mean) or not np.array_equal(state["normalizer_std"], normalizer.std):
            raise RuntimeError("A1 FIT normalizer mismatch")
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        _,_,records = exp._evaluate(model,val_loader,ez_weight,split_name="val")
        error = grid_error(epoch_grid(records,epoch),source_grid(fold,epoch))
        if error > 1e-6:
            raise RuntimeError(f"SOURCE_A1_REPRODUCTION_FAILED fold={fold} epoch={epoch} error={error}")
        errors.append(error)
    public,_ = finalize_fold([source_grid(fold,e) for e in range(1,31)],"A1",fold,folder/"A1_VLOO_PRIVATE.csv")
    if not math.isclose(public["patient_macro_f1"],EXPECTED_FOLDS[fold-1],abs_tol=1e-6):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED VLOO")
    chosen = read_csv(folder/"A1_VLOO_PRIVATE.csv")
    if set(r["subject_id"] for r in chosen) != val_ids:
        raise RuntimeError("VLOO target membership changed")
    agg_errors, r4_errors = [], []
    seizure_count = 0
    dimension = None
    for epoch in sorted({int(r["selected_epoch"]) for r in chosen}):
        target = folder/f"epoch_{epoch:02d}_representations.pkl"
        if target.exists():
            with target.open("rb") as f:
                payload=pickle.load(f)
            if payload["fold"] != fold or payload["epoch"] != epoch or set(payload["fit"]) != fit_ids or set(payload["val"]) != val_ids:
                raise RuntimeError("Private extraction cache provenance mismatch")
        else:
            state=torch.load(checkpoint(fold,epoch),map_location=exp.device,weights_only=False)
            model.load_state_dict(state["model_state_dict"],strict=True)
            model.eval()
            fit, ae1, re1, sc1 = capture(model,fit_loader,exp.device,exp.patient_index)
            val, ae2, re2, sc2 = capture(model,val_loader,exp.device,exp.patient_index)
            if set(fit)!=fit_ids or set(val)!=val_ids:
                raise RuntimeError("Extracted role membership mismatch")
            payload={"fold":fold,"epoch":epoch,"fit":fit,"val":val,"seizure_dim":int(next(iter(fit.values()))["E"].shape[-1]),
                     "r4_dim":int(next(iter(fit.values()))["R4"].shape[-1]),"agg_error":max(ae1,ae2),
                     "r4_error":max(re1,re2),"seizures_checked":sc1+sc2}
            tmp=target.with_suffix(".tmp")
            with tmp.open("wb") as f:
                pickle.dump(payload,f,protocol=5)
            tmp.replace(target)
            print(f"[EXTRACT] fold={fold} epoch={epoch} seizures={sc1+sc2}",flush=True)
        agg_errors.append(payload["agg_error"])
        r4_errors.append(payload["r4_error"])
        seizure_count+=payload["seizures_checked"]
        dimension=(payload["seizure_dim"],payload["r4_dim"])
    if dimension[1]!=64 or max(agg_errors)>1e-6 or max(r4_errors)>1e-6:
        raise RuntimeError("Frozen seizure/R4 interface failed")
    write_json(folder/"extract_status.json",{"fold":fold,"source_replayed":30,"max_grid_error":max(errors),
               "selected_epochs":len({int(r["selected_epoch"]) for r in chosen}),"seizures_checked_across_epoch_extractions":seizure_count,
               "seizure_embedding_dim":dimension[0],"r4_dim":dimension[1],"max_aggregator_replay_error":max(agg_errors),
               "max_r4_replay_error":max(r4_errors),"canonical_name_alignment_pass":True,"outer_loader":False})
    return public


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--fold",type=int,choices=range(1,6))
    opt=parser.parse_args()
    preflight()
    exp=experiment()
    results=[]
    for split in exp.outer_splits:
        if opt.fold is None or int(split["fold_idx"])==opt.fold:
            results.append(run_fold(exp,split))
    if opt.fold is None:
        if len(results)!=5 or not math.isclose(np.mean([r["patient_macro_f1"] for r in results]),0.6259962097139906,abs_tol=1e-6):
            raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED mean")
        status=[json.loads((RUNTIME/"private"/f"fold_{f}"/"extract_status.json").read_text(encoding="utf-8")) for f in range(1,6)]
        write_json(ROOT/"SOURCE_REPRODUCTION.json",{"pass":True,"terminal":"SOURCE_A1_REPRODUCED","checkpoints":150,
                   "max_grid_error":max(r["max_grid_error"] for r in status),"mean_macro_f1":float(np.mean([r["patient_macro_f1"] for r in results])),
                   "fold_macro_f1":[r["patient_macro_f1"] for r in results],"outer_predictions_or_metrics":False,
                   "legacy_monolithic_loader_materializes_all_80_labels":True})
        write_json(ROOT/"CHANNEL_ALIGNMENT_AUDIT.json",{"pass":True,"patients_checked_per_selected_epoch":[(51,51,50,52,51)[f-1]+13 for f in range(1,6)],
                   "seizure_observations_checked_across_selected_epochs":sum(r["seizures_checked_across_epoch_extractions"] for r in status),
                   "mapping_source":"source dataset.py patient_meta canonical_channels; local sample channel_names_norm mapped into canonical index; source dataset SHA256 ccb4fadd9416a63bdceb46fe7edb515a67008f896435e622059b5713cbb42702",
                   "verification":"batch canonical names match source patient_index; names unique; source seizure_channel_mask and seizure_mask exactly consumed by aggregator",
                   "missing_channel_handling":"exclude absent seizure-channel entries for distributions; across-seizure persistence uses only observed entries; pair correlations use intersection of present canonical channels"})
        write_json(ROOT/"representations"/"SEIZURE_REPRESENTATION_AUDIT.json",{"pass":True,"source_output_key":"seizure_channel_embedding",
                   "capture":"cross-seizure aggregator forward-pre-hook input 0, bitwise-equal to source forward output key",
                   "conceptual_shape":"[batch,seizure,canonical_channel,D_s]", "D_s":status[0]["seizure_embedding_dim"],
                   "max_aggregator_replay_error":max(r["max_aggregator_replay_error"] for r in status)})
        write_json(ROOT/"representations"/"R4_REPLAY_AUDIT.json",{"pass":True,"source":"original channel classifier forward-pre-hook",
                   "D":64,"max_logit_replay_error":max(r["max_r4_replay_error"] for r in status)})


if __name__=="__main__":
    main()
