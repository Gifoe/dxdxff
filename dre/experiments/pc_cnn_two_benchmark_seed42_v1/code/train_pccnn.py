"""Validation-only PC-CNN Stage B/C; loads matched B0, never test data.

Stage B trains only zero-output residual conditioning/context modules with the
approved small nonzero gate initialization. Stage C is entered only when the
predeclared Stage-B validation signal exceeds B0. The architecture and split
remain fixed; each patient has one optimizer step and equal loss weight.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from descriptor_norm import apply as normalize_descriptors
from descriptor_norm import save as save_norm
from official_spectrum import OfficialSpectrum, load_official_module
from patient_bank import IctalBank, OmniTrainBank
from pc_cnn import PCCNN
from raw_metrics import ictal_validation, omni_validation
from train_rawcnn import digest, save_json, save_torch, selection_rank


def build_bank(args, lock):
    if args.benchmark == "ictal":
        if args.fold not in range(1, 6) or digest(args.ictal_manifest) != lock["ictal_fold_manifest_sha256"]:
            raise RuntimeError("Ictal fold manifest differs from lock")
        bank = IctalBank(args.ictal_cache, args.ictal_manifest)
        return bank, bank.folds[args.fold]["fit"], bank.folds[args.fold]["validation"]
    if args.fold != 1 or digest(args.omni_official_split) != lock["omni_official_split_sha256"] or \
            digest(args.omni_inner_split) != lock["omni_inner_train_val_split_sha256"]:
        raise RuntimeError("Omni TRAIN/inner split differs from lock")
    bank = OmniTrainBank(args.omni_cache, args.omni_inner_split, args.omni_official_split)
    return bank, bank.patients("inner_train"), bank.patients("inner_val")


def normalization(bank, fit, args, work, protocol_sha):
    path = work / "descriptor_normalization.json"
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved["protocol_sha256"] != protocol_sha or saved["fit_patient_count"] != len(fit) or \
                saved["benchmark"] != args.benchmark or saved["fold"] != args.fold:
            raise RuntimeError("Descriptor fit resume provenance mismatch")
        return saved
    moments = bank.fit_moments(args.fold) if isinstance(bank, IctalBank) else bank.fit_moments()
    frozen = moments.freeze(benchmark=args.benchmark, fold=args.fold,
                            fit_patient_count=len(fit), protocol_sha256=protocol_sha)
    save_norm(path, frozen)
    return frozen


def record_tensors(record, frozen, device):
    values = normalize_descriptors(record["descriptors"], record["descriptor_mask"], frozen)
    return (torch.from_numpy(record["waveforms"][None]).to(device),
            torch.from_numpy(values[None]).to(device),
            torch.from_numpy(record["channel_mask"][None]).to(device),
            torch.from_numpy(record["descriptor_mask"][None].astype(np.float32)).to(device),
            torch.from_numpy(record["labels"].astype(np.float32)).to(device))


def records_for(bank, patient, epoch, validation):
    return (bank.records(patient) if isinstance(bank, IctalBank) else
            bank.records(patient, epoch, all_clips=validation))


def configure_stage(model, stage):
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    if stage == "B":
        for parameter in model.raw.parameters():
            parameter.requires_grad_(False)
        groups = [{"params": [p for name, p in model.named_parameters()
                              if not name.startswith("raw.")], "lr": 2e-4}]
    elif stage == "C":
        for parameter in model.raw.parameters():
            parameter.requires_grad_(False)
        for module in (model.raw.cnn.layer4, model.raw.fc, model.raw.bn,
                       model.raw.fc1, model.raw.bn1, model.raw.fc_out):
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        groups = [
            {"params": [p for name, p in model.named_parameters() if not name.startswith("raw.")],
             "lr": 2e-4},
            {"params": list(model.raw.cnn.layer4.parameters()), "lr": 2e-5},
            {"params": [p for module in (model.raw.fc, model.raw.bn, model.raw.fc1,
                                          model.raw.bn1, model.raw.fc_out)
                        for p in module.parameters()], "lr": 5e-5},
        ]
    else:
        raise ValueError("Only frozen Stage B and controlled Stage C are allowed")
    return torch.optim.AdamW(groups, weight_decay=5e-4)


def patient_step(model, preprocessor, bank, patient, epoch, frozen, optimizer, device):
    denominator = bank.labeled_observations(patient)
    if denominator <= 0:
        raise RuntimeError("No labeled observations in fit patient")
    model.train()
    model.set_batch_norm_running_state(True)
    optimizer.zero_grad(set_to_none=True)
    observed = 0
    for record in records_for(bank, patient, epoch, validation=False):
        wave, descriptor, channel, available, labels = record_tensors(record, frozen, device)
        logits = model.forward_record(wave, record["sampling_rate_hz"], descriptor,
                                      channel, available, preprocessor,
                                      channel_chunk=4, checkpoint_backbone=True)[0]
        valid = channel[0] & (labels >= 0)
        if not bool(valid.any()):
            continue
        weight = torch.where(labels[valid] > 0, 2.0, 1.0)
        loss = F.binary_cross_entropy_with_logits(
            logits[valid], labels[valid], weight=weight, reduction="sum") / denominator
        loss.backward()
        observed += int(valid.sum())
    if observed != denominator:
        raise RuntimeError("Patient-equal PC-CNN denominator mismatch")
    torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), 5.0)
    optimizer.step()
    return observed


def evaluate(model, preprocessor, bank, patients, benchmark, epoch, frozen, device,
             *, physiology=True, context=True):
    model.eval()
    predictions = {}
    with torch.inference_mode():
        for patient in patients:
            current = []
            for record in records_for(bank, patient, epoch, validation=True):
                wave, descriptor, channel, available, labels = record_tensors(record, frozen, device)
                score = model.forward_record(
                    wave, record["sampling_rate_hz"], descriptor, channel,
                    available, preprocessor, physiology=physiology,
                    context=context, channel_chunk=4)[0].float().sigmoid().cpu().numpy()
                chosen = np.flatnonzero(record["channel_mask"])
                current.append({"channel": [record["channel_names"][int(i)] for i in chosen],
                                "label": labels[chosen].int().cpu().numpy().tolist(),
                                "score": score[chosen].tolist(), "edf": record.get("edf", "")})
            predictions[patient] = current
    return ictal_validation(predictions) if benchmark == "ictal" else omni_validation(predictions)


def restore_rng(state, device):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda_rng"].detach().cpu())


def training_stage(model, stage, bank, fit, val, preprocessor, frozen,
                   work, benchmark, fold, protocol_sha, epochs, device):
    optimizer = configure_stage(model, stage)
    last, best = work / f"stage_{stage}_last.pt", work / f"stage_{stage}_best.pt"
    selected, first = None, 1
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["protocol_sha256"] != protocol_sha or state["benchmark"] != benchmark or \
                state["fold"] != fold or state["stage"] != stage:
            raise RuntimeError("PC-CNN stage resume provenance mismatch")
        model.load_state_dict(state["model_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        restore_rng(state, device)
        selected, first = state["selected"], state["epoch"] + 1
    for epoch in range(first, epochs + 1):
        started = time.perf_counter()
        shuffled = list(fit)
        random.shuffle(shuffled)
        for patient in shuffled:
            patient_step(model, preprocessor, bank, patient, epoch, frozen, optimizer, device)
        metrics, private = evaluate(model, preprocessor, bank, val, benchmark,
                                    epoch, frozen, device)
        if metrics["patients"] != len(val):
            raise RuntimeError("Validation cohort count changed")
        rank = selection_rank(metrics, benchmark, epoch)
        if selected is None or tuple(selected["rank"]) < rank:
            selected = {"epoch": epoch, "rank": rank, "metrics": metrics}
            save_torch(best, {"protocol_sha256": protocol_sha, "benchmark": benchmark,
                              "fold": fold, "stage": stage, "selected": selected,
                              "model_state": model.state_dict(), "test_accessed": False})
        save_json(work / f"stage_{stage}_validation_epoch_{epoch:02d}_private.json",
                  {"epoch": epoch, "protocol_sha256": protocol_sha,
                   "private_patient_scores": private})
        save_torch(last, {"protocol_sha256": protocol_sha, "benchmark": benchmark,
                          "fold": fold, "stage": stage, "epoch": epoch,
                          "selected": selected, "model_state": model.state_dict(),
                          "optimizer_state": optimizer.state_dict(),
                          "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state().detach().cpu()
                          if device.type == "cuda" else None})
        print(json.dumps({"stage": stage, "benchmark": benchmark, "fold": fold,
                          "epoch": epoch, "max_epochs": epochs, "validation": metrics,
                          "selected_epoch": selected["epoch"],
                          "seconds_epoch": time.perf_counter() - started,
                          "test_accessed": False}), flush=True)
    if not best.is_file():
        raise RuntimeError("No selected PC-CNN checkpoint")
    return selected, best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=("ictal", "omni"), required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--official-cnn", type=Path, required=True)
    p.add_argument("--ictal-cache", type=Path)
    p.add_argument("--ictal-manifest", type=Path)
    p.add_argument("--omni-cache", type=Path)
    p.add_argument("--omni-official-split", type=Path)
    p.add_argument("--omni-inner-split", type=Path)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    lock = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_sha = digest(args.protocol)
    bank, fit, val = build_bank(args, lock)
    if set(fit) & set(val):
        raise RuntimeError("Fit/validation patient overlap")
    work = args.runtime / args.benchmark / f"fold{args.fold}"
    b0_dir = work / "B0_RawCNN"
    b0_selected = json.loads((b0_dir / "RAW_B0_SELECTION.json").read_text(encoding="utf-8"))
    b0_ckpt = torch.load(b0_dir / "selected_best.pt", map_location="cpu", weights_only=False)
    if b0_selected["protocol_sha256"] != protocol_sha or b0_ckpt["protocol_sha256"] != protocol_sha or \
            b0_selected["selected"] != b0_ckpt["selected"]:
        raise RuntimeError("Matched B0 not fully selected under same protocol")
    frozen = normalization(bank, fit, args, work, protocol_sha)
    random.seed(4200 + args.fold)
    np.random.seed(4200 + args.fold)
    torch.manual_seed(4200 + args.fold)
    torch.cuda.manual_seed_all(4200 + args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_official_module(args.official_cnn)
    raw = module.NeuralCNN(in_channels=1, outputs=1)
    raw.load_state_dict(b0_ckpt["raw_state"])
    model = PCCNN(raw).to(device)
    preprocessor = OfficialSpectrum(module, channel_chunk=4)
    stage_b, best_b = training_stage(model, "B", bank, fit, val,
                                     preprocessor, frozen, work, args.benchmark,
                                     args.fold, protocol_sha,
                                     int(lock["training"]["warmup_max_epochs"]), device)
    base = b0_selected["selected"]["metrics"]
    improvement = (stage_b["metrics"]["auroc"] > base["auroc"] or
                   args.benchmark == "ictal" and stage_b["metrics"]["ap"] > base["ap"])
    final = {"stage": "B", "selected": stage_b,
             "checkpoint": str(best_b), "stage_c_entered": bool(improvement)}
    if improvement:
        state = torch.load(best_b, map_location=device, weights_only=False)
        model.load_state_dict(state["model_state"])
        stage_c, best_c = training_stage(model, "C", bank, fit, val,
                                         preprocessor, frozen, work, args.benchmark,
                                         args.fold, protocol_sha,
                                         int(lock["training"]["stage_c_max_epochs"]), device)
        if tuple(stage_c["rank"]) > tuple(stage_b["rank"]):
            final.update({"stage": "C", "selected": stage_c,
                          "checkpoint": str(best_c)})
    save_json(work / "PC_VALIDATION_SELECTION.json",
              {"status": "TRAIN_VALIDATION_COMPLETE", "protocol_sha256": protocol_sha,
               "benchmark": args.benchmark, "fold": args.fold,
               "fit_patients": len(fit), "validation_patients": len(val),
               "matched_b0": b0_selected["selected"], "stage_b": stage_b,
               "final": final, "test_accessed": False})
    print(json.dumps({"status": "PC_VALIDATION_SELECTED", "benchmark": args.benchmark,
                      "fold": args.fold, "final_stage": final["stage"],
                      "stage_c_entered": bool(improvement), "test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
