"""Matched B0 RawCNN TRAIN/inner-validation, resumable and test-inaccessible.

One optimizer step is taken per patient. Each patient's labeled channel-record
loss is averaged before that step; record count and channel count cannot give
that patient extra optimizer weight. Within a patient, channel chunks are
backpropagated immediately to bound ResNet18 activation memory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from official_spectrum import OfficialSpectrum, load_official_module
from patient_bank import IctalBank, OmniTrainBank
from pc_cnn import _channel_groups
from raw_metrics import ictal_validation, omni_validation


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save_torch(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def save_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def to_tensors(record, device):
    return (torch.from_numpy(record["waveforms"]).to(device),
            torch.from_numpy(record["labels"].astype(np.float32)).to(device),
            torch.from_numpy(record["channel_mask"]).to(device))


def patient_train_step(raw, preprocessor, bank, patient, epoch, optimizer, device,
                       channel_chunk=4):
    denominator = bank.labeled_observations(patient)
    if denominator <= 0:
        raise RuntimeError("TRAIN patient has no supervised channel-record observations")
    raw.train()
    optimizer.zero_grad(set_to_none=True)
    seen = 0
    records = (bank.records(patient) if isinstance(bank, IctalBank) else
               bank.records(patient, epoch, all_clips=False))
    for record in records:
        waves, labels, present = to_tensors(record, device)
        selected = torch.nonzero(present, as_tuple=False).flatten()
        if len(selected) < 2:
            raise RuntimeError("RawCNN BatchNorm requires >=2 channels per record")
        for ids in _channel_groups(selected, channel_chunk):
            image = preprocessor(waves[ids], record["sampling_rate_hz"])
            logits = raw(image).squeeze(-1)
            y = labels[ids]
            valid = y >= 0
            if not bool(valid.any()):
                continue
            weights = torch.where(y[valid] > 0, 2.0, 1.0)
            piece = F.binary_cross_entropy_with_logits(
                logits[valid], y[valid], weight=weights, reduction="sum") / denominator
            piece.backward()
            seen += int(valid.sum())
    if seen != denominator:
        raise RuntimeError(f"Patient-equal denominator mismatch {seen} != {denominator}")
    optimizer.step()
    return seen


def evaluate(raw, preprocessor, bank, patients, benchmark, epoch, device,
             channel_chunk=4):
    raw.eval()
    predicted = {}
    with torch.inference_mode():
        for patient in patients:
            records = (bank.records(patient) if isinstance(bank, IctalBank) else
                       bank.records(patient, epoch, all_clips=True))
            current = []
            for record in records:
                waves, labels, present = to_tensors(record, device)
                selected = torch.nonzero(present, as_tuple=False).flatten()
                logits = []
                for ids in _channel_groups(selected, channel_chunk):
                    image = preprocessor(waves[ids], record["sampling_rate_hz"])
                    logits.append(raw(image).squeeze(-1).float().sigmoid().cpu().numpy())
                current.append({"channel": [record["channel_names"][int(index)] for index in selected],
                                "label": labels[selected].int().cpu().numpy().tolist(),
                                "score": np.concatenate(logits).tolist(),
                                "edf": record.get("edf", "")})
            predicted[patient] = current
    return ictal_validation(predicted) if benchmark == "ictal" else omni_validation(predicted)


def selection_rank(metrics: dict, benchmark: str, epoch: int):
    names = ("auroc", "ap", "mrr") if benchmark == "ictal" else \
        ("auroc", "ap", "macro_f1_0_5")
    return tuple(round(float(metrics[name]) if metrics[name] is not None else -1, 12)
                 for name in names) + (-epoch,)


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
    p.add_argument("--smoke-only", action="store_true")
    args = p.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    protocol_sha = digest(args.protocol)
    if args.benchmark == "ictal":
        if args.fold not in range(1, 6) or not args.ictal_manifest or \
                digest(args.ictal_manifest) != protocol["ictal_fold_manifest_sha256"]:
            raise RuntimeError("Frozen ictal fold mismatch")
        bank = IctalBank(args.ictal_cache, args.ictal_manifest)
        fit, val = bank.folds[args.fold]["fit"], bank.folds[args.fold]["validation"]
    else:
        if args.fold != 1 or not args.omni_official_split or not args.omni_inner_split or \
                digest(args.omni_official_split) != protocol["omni_official_split_sha256"] or \
                digest(args.omni_inner_split) != protocol["omni_inner_train_val_split_sha256"]:
            raise RuntimeError("Frozen Omni TRAIN/inner split mismatch")
        bank = OmniTrainBank(args.omni_cache, args.omni_inner_split, args.omni_official_split)
        fit, val = bank.patients("inner_train"), bank.patients("inner_val")
    if set(fit) & set(val):
        raise RuntimeError("TRAIN/validation patient leakage")
    random.seed(42 + args.fold)
    np.random.seed(42 + args.fold)
    torch.manual_seed(42 + args.fold)
    torch.cuda.manual_seed_all(42 + args.fold)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    module = load_official_module(args.official_cnn)
    raw = module.NeuralCNN(in_channels=1, outputs=1).to(device)
    preprocessor = OfficialSpectrum(module, channel_chunk=4)
    optimizer = torch.optim.Adam(raw.parameters(), lr=protocol["training"]["rawcnn_lr"])
    if args.smoke_only:
        start = time.perf_counter()
        observed = patient_train_step(raw, preprocessor, bank, fit[0], 1,
                                      optimizer, device)
        print(json.dumps({"status": "RAW_B0_SMOKE_PASS", "benchmark": args.benchmark,
                          "fit_patients": 1, "labeled_observations": observed,
                          "seconds": time.perf_counter() - start,
                          "validation_or_test_accessed": False}), flush=True)
        return
    work = args.runtime / args.benchmark / f"fold{args.fold}" / "B0_RawCNN"
    work.mkdir(parents=True, exist_ok=True)
    last, best = work / "last.pt", work / "selected_best.pt"
    selected, start_epoch = None, 1
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if state["protocol_sha256"] != protocol_sha or state["benchmark"] != args.benchmark or \
                state["fold"] != args.fold:
            raise RuntimeError("RawCNN resume provenance mismatch")
        raw.load_state_dict(state["raw_state"])
        optimizer.load_state_dict(state["optimizer_state"])
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(state["cuda_rng"].detach().cpu())
        selected, start_epoch = state["selected"], state["epoch"] + 1
    limit = int(protocol["training"]["rawcnn_max_epochs"])
    for epoch in range(start_epoch, limit + 1):
        started = time.perf_counter()
        order = list(fit)
        random.shuffle(order)
        patient_steps = 0
        for patient in order:
            patient_train_step(raw, preprocessor, bank, patient, epoch,
                               optimizer, device)
            patient_steps += 1
        metrics, private = evaluate(raw, preprocessor, bank, val,
                                    args.benchmark, epoch, device)
        if metrics["patients"] != len(val):
            raise RuntimeError("Inner-validation cohort count differs")
        rank = selection_rank(metrics, args.benchmark, epoch)
        if selected is None or tuple(selected["rank"]) < rank:
            selected = {"epoch": epoch, "rank": rank, "metrics": metrics}
            save_torch(best, {"protocol_sha256": protocol_sha,
                              "benchmark": args.benchmark, "fold": args.fold,
                              "selected": selected, "raw_state": raw.state_dict(),
                              "fit_patients": len(fit), "validation_patients": len(val),
                              "test_accessed": False})
        save_json(work / f"validation_epoch_{epoch:02d}_private.json",
                  {"protocol_sha256": protocol_sha, "epoch": epoch,
                   "private_patient_scores": private})
        save_torch(last, {"protocol_sha256": protocol_sha,
                          "benchmark": args.benchmark, "fold": args.fold,
                          "epoch": epoch, "selected": selected,
                          "raw_state": raw.state_dict(),
                          "optimizer_state": optimizer.state_dict(),
                          "python_rng": random.getstate(),
                          "numpy_rng": np.random.get_state(),
                          "torch_rng": torch.get_rng_state(),
                          "cuda_rng": torch.cuda.get_rng_state().detach().cpu()
                          if device.type == "cuda" else None})
        print(json.dumps({"stage": "B0_RawCNN", "benchmark": args.benchmark,
                          "fold": args.fold, "epoch": epoch, "max_epochs": limit,
                          "fit_patient_steps": patient_steps,
                          "validation": metrics, "selected_epoch": selected["epoch"],
                          "seconds_epoch": time.perf_counter() - started,
                          "test_accessed": False}), flush=True)
    save_json(work / "RAW_B0_SELECTION.json",
              {"status": "TRAIN_VALIDATION_COMPLETE", "protocol_sha256": protocol_sha,
               "benchmark": args.benchmark, "fold": args.fold,
               "fit_patients": len(fit), "validation_patients": len(val),
               "selected": selected, "test_accessed": False})


if __name__ == "__main__":
    main()
