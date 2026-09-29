"""Resume-safe train/validation-only CRST-0 and CRST-FULL training.

This entry point does not open outer/test patient spectral files. Private
checkpoints and patient records remain under --runtime, never in GitHub.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from crst_metrics import aggregate_patients, frozen_threshold
from crst_model import CRSTiEEG
from crst_objectives import ssl_objective, supervised_parts
from patient_bank import IctalPatientBank, OmniPatientBank, to_device


def sha(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(part)
    return h.hexdigest()


def save_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)


def json_atomic(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_patient(bank, name, epoch, validation=False):
    if isinstance(bank, IctalPatientBank):
        return bank.load(name)
    return bank.load(name, epoch, all_clips=validation)


def evaluation(model, bank, names, device, epoch, intervention=None):
    model.eval()
    rows, threshold_rows, private_scores = [], [], {}
    with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                enabled=device.type == "cuda"):
        for name in names:
            sample = to_device(load_patient(bank, name, epoch, validation=True), device)
            logits = model(sample["patches"], sample["frequency_mask"],
                           sample["window_mask"], sample["edges"],
                           intervention=intervention)[0]
            truth = sample["labels"][0]
            valid = truth >= 0
            rows.append((truth[valid].int().cpu().numpy(),
                         logits[valid].sigmoid().float().cpu().numpy()))
            private_scores[name] = {"labels": rows[-1][0].tolist(),
                                    "scores": rows[-1][1].tolist()}
            if "record_labels" in sample:
                rtruth = sample["record_labels"][0]
                # Official classification is one (EDF, channel), not one
                # (clip, channel).  Multiple clips from an EDF must not give
                # that EDF extra weight in threshold selection.
                edf_ids = np.asarray(sample["record_edf_ids"])
                first = [int(np.flatnonzero(edf_ids == edf)[0])
                         for edf in np.unique(edf_ids)]
                edf_truth = rtruth[first]
                rvalid = edf_truth >= 0
                repeated_score = logits[None].expand_as(edf_truth)
                threshold_rows.append((edf_truth[rvalid].int().cpu().numpy(),
                                       repeated_score[rvalid].sigmoid().float().cpu().numpy()))
            else:
                threshold_rows.append(rows[-1])
    return aggregate_patients(rows), rows, threshold_rows, private_scores


def fixed_ssl_roles(names, fold):
    ordered = sorted(names, key=lambda name: hashlib.sha256(
        f"42|{fold}|{name}".encode()).hexdigest())
    n = max(5, math.ceil(0.1 * len(names)))
    return ordered[n:], ordered[:n]


def run_ssl(bank, names, fold, work: Path, protocol_sha: str, train_sha: str,
            device, *, max_epochs=50, intervention=None):
    work.mkdir(parents=True, exist_ok=True)
    train_names, heldout_names = fixed_ssl_roles(names, fold)
    seed_everything(420000 + fold)
    model = CRSTiEEG(activation_checkpointing=False).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
    last = work / "ssl_last.pt"
    best = work / "ssl_best.pt"
    start, best_loss, no_gain = 1, math.inf, 0
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if (state["protocol_sha"] != protocol_sha or state["train_sha"] != train_sha or
                state["fold"] != fold or state["phase"] != "SSL" or
                state.get("intervention") != intervention):
            raise RuntimeError("SSL resume identity mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start = state["epoch"] + 1
        best_loss, no_gain = state["best_loss"], state["no_gain"]
    if (work / "ssl_complete.json").exists():
        return best
    total_steps = max_epochs * len(train_names)
    for epoch in range(start, max_epochs + 1):
        model.train()
        epoch_loss = []
        order = sorted(train_names, key=lambda name: hashlib.sha256(
            f"42|ssl|{fold}|{epoch}|{name}".encode()).hexdigest())
        started = time.perf_counter()
        for ordinal, name in enumerate(order):
            seed = 420000000 + fold * 1000000 + epoch * 10000 + ordinal
            seed_everything(seed)
            sample = to_device(load_patient(bank, name, epoch), device)
            step = (epoch - 1) * len(order) + ordinal
            warmup = max(1, math.ceil(0.05 * total_steps))
            if step < warmup:
                factor = (step + 1) / warmup
            else:
                factor = 0.5 * (1 + math.cos(math.pi * (step - warmup) /
                                             max(1, total_steps - warmup)))
            for group in optimizer.param_groups:
                group["lr"] = 3e-4 * factor
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                parts = ssl_objective(model, sample["patches"], sample["frequency_mask"],
                                      sample["window_mask"], sample["edges"], seed=seed,
                                      intervention=intervention)
            loss = parts["total"]
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite SSL training loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss.append(float(loss.detach()))
        model.eval()
        val_losses = []
        with torch.inference_mode(), torch.autocast(device.type, dtype=torch.bfloat16,
                                                    enabled=device.type == "cuda"):
            for ordinal, name in enumerate(heldout_names):
                sample = to_device(load_patient(bank, name, 0, validation=False), device)
                result = ssl_objective(model, sample["patches"],
                                       sample["frequency_mask"], sample["window_mask"],
                                       sample["edges"], seed=42000000 + fold * 100 + ordinal,
                                       intervention=intervention)
                val_losses.append(float(result["total"]))
        val = float(np.mean(val_losses))
        if val < best_loss - 1e-7:
            best_loss, no_gain = val, 0
            save_atomic(best, {"model": model.state_dict(), "epoch": epoch,
                               "validation_ssl_loss": val, "protocol_sha": protocol_sha,
                               "train_sha": train_sha, "fold": fold, "phase": "SSL",
                               "intervention": intervention})
        else:
            no_gain += 1
        save_atomic(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "epoch": epoch, "best_loss": best_loss, "no_gain": no_gain,
                           "protocol_sha": protocol_sha, "train_sha": train_sha,
                           "fold": fold, "phase": "SSL", "intervention": intervention})
        print(json.dumps({"benchmark": "ictal" if isinstance(bank, IctalPatientBank) else "omni",
                          "fold": fold, "phase": "SSL", "epoch": epoch,
                          "intervention": intervention,
                          "train_ssl": float(np.mean(epoch_loss)), "heldout_ssl": val,
                          "best_ssl": best_loss, "seconds": round(time.perf_counter()-started, 2)}),
              flush=True)
        if epoch >= 10 and no_gain >= 8:
            break
    json_atomic(work / "ssl_complete.json", {"best_checkpoint_sha256": sha(best),
                                             "best_validation_ssl_loss": best_loss,
                                             "last_epoch": epoch, "heldout_patients": len(heldout_names),
                                             "intervention": intervention,
                                             "test_waveforms_used": False})
    return best


def center_weights(bank, names):
    centers = {}
    for name in names:
        if isinstance(bank, IctalPatientBank):
            centers[name] = bank.center[name]
        else:
            centers[name] = load_patient(bank, name, 0)["center"]
    counts = Counter(centers.values())
    q = {center: count / len(names) for center, count in counts.items()}
    return centers, counts, q


def supervised_optimizer(model, stage):
    if stage == 1:
        model.tokenizer.requires_grad_(False)
        return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                 lr=2e-4, weight_decay=5e-4)
    model.tokenizer.requires_grad_(True)
    head_names = ("patient_pma", "classifier", "window_pool", "record_pool", "pma_seed")
    tokenizer, backbone, head = [], [], []
    for name, parameter in model.named_parameters():
        if name.startswith("tokenizer"):
            tokenizer.append(parameter)
        elif name.startswith(head_names):
            head.append(parameter)
        else:
            backbone.append(parameter)
    return torch.optim.AdamW([{"params": tokenizer, "lr": 2e-5},
                              {"params": backbone, "lr": 1e-4},
                              {"params": head, "lr": 2e-4}], weight_decay=5e-4)


def run_supervised(bank, train_names, val_names, fold, variant, work: Path,
                   protocol_sha: str, train_sha: str, device, ssl_best: Path | None,
                   intervention=None):
    work.mkdir(parents=True, exist_ok=True)
    seed_everything(42000 + fold)
    model = CRSTiEEG(activation_checkpointing=False).to(device)
    if ssl_best is not None:
        source = torch.load(ssl_best, map_location=device, weights_only=False)
        if source["protocol_sha"] != protocol_sha or source["train_sha"] != train_sha:
            raise RuntimeError("SSL source provenance mismatch")
        model.load_state_dict(source["model"])
    centers, counts, q = center_weights(bank, train_names)
    last = work / "supervised_last.pt"
    best = work / "supervised_best.pt"
    start, best_rank, no_gain = 1, None, 0
    state = None
    if last.exists():
        state = torch.load(last, map_location=device, weights_only=False)
        if (state["protocol_sha"] != protocol_sha or state["train_sha"] != train_sha or
                state["fold"] != fold or state["variant"] != variant or
                state.get("intervention") != intervention):
            raise RuntimeError("Supervised resume identity mismatch")
        model.load_state_dict(state["model"])
        start = state["epoch"] + 1
        best_rank = tuple(state["best_rank"]) if state["best_rank"] is not None else None
        q, no_gain = state["group_q"], state["no_gain"]
        for previous in range(1, start):
            if not (work / f"epoch_{previous:02d}_validation_private.json").is_file():
                raise RuntimeError("Cannot resume without complete private epoch score grid")
    if (work / "supervised_complete.json").exists():
        return best
    optimizer = supervised_optimizer(model, 1 if start <= 5 else 2)
    if state is not None and ((state["epoch"] <= 5 and start <= 5) or
                              (state["epoch"] >= 6 and start >= 6)):
        optimizer.load_state_dict(state["optimizer"])
    for epoch in range(start, 36):
        stage = 1 if epoch <= 5 else 2
        if epoch == 6:
            optimizer = supervised_optimizer(model, 2)
        model.train()
        loss_parts = defaultdict(list)
        by_center = defaultdict(list)
        order = sorted(train_names, key=lambda name: hashlib.sha256(
            f"42|supervised|{fold}|{variant}|{epoch}|{name}".encode()).hexdigest())
        started = time.perf_counter()
        for ordinal, name in enumerate(order):
            seed = 420000000 + fold * 1000000 + epoch * 10000 + ordinal
            seed_everything(seed)
            sample = to_device(load_patient(bank, name, epoch), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, dtype=torch.bfloat16,
                                enabled=device.type == "cuda"):
                output = model(sample["patches"], sample["frequency_mask"],
                               sample["window_mask"], sample["edges"],
                               intervention=intervention, return_aux=True)
                parts = supervised_parts(output["logits"], sample["labels"],
                                         output["record_logits"], sample["window_mask"], seed,
                                         sample.get("record_labels"))
                correction = q[centers[name]] / (counts[centers[name]] / len(order))
                loss = parts["total"] * correction
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            for key, value in parts.items():
                loss_parts[key].append(float(value.detach()))
            by_center[centers[name]].append(float(parts["bce"].detach()))
        # Smooth Group-DRO update from TRAIN group losses only, never validation.
        updated = {center: q[center] * math.exp(0.01 * np.mean(values))
                   for center, values in by_center.items()}
        denom = sum(updated.values())
        q = {center: value / denom for center, value in updated.items()}
        validation, val_rows, threshold_rows, private_scores = evaluation(
            model, bank, val_names, device, epoch, intervention=intervention)
        json_atomic(work / f"epoch_{epoch:02d}_validation_private.json",
                    {"fold": fold, "variant": variant, "epoch": epoch,
                     "intervention": intervention,
                     "protocol_sha": protocol_sha, "train_sha": train_sha,
                     "clinical_positive": "EZ" if isinstance(bank, IctalPatientBank)
                     else "official_pathological",
                     "patient_scores_private": private_scores})
        rank = (validation["auroc"], validation["ap"], validation["mrr"], -epoch)
        if not all(math.isfinite(float(x)) for x in rank):
            raise RuntimeError("Nonfinite primary validation metric")
        if best_rank is None or rank > best_rank:
            best_rank, no_gain = rank, 0
            threshold = frozen_threshold(threshold_rows)
            save_atomic(best, {"model": model.state_dict(), "epoch": epoch,
                               "validation": validation, "threshold": threshold,
                               "group_q": q, "fold": fold, "variant": variant,
                               "protocol_sha": protocol_sha, "train_sha": train_sha,
                               "intervention": intervention})
        else:
            no_gain += 1
        save_atomic(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "epoch": epoch, "best_rank": best_rank,
                           "no_gain": no_gain, "group_q": q, "fold": fold,
                           "variant": variant, "protocol_sha": protocol_sha,
                           "train_sha": train_sha, "intervention": intervention})
        print(json.dumps({"benchmark": "ictal" if isinstance(bank, IctalPatientBank) else "omni",
                          "fold": fold, "variant": variant, "stage": stage,
                          "intervention": intervention,
                          "epoch": epoch, "train_loss": float(np.mean(loss_parts["total"])),
                          "val_patient_equal_auroc": validation["auroc"],
                          "val_patient_equal_ap": validation["ap"],
                          "val_mrr": validation["mrr"], "selected_epoch": -best_rank[3],
                          "group_q": q, "seconds": round(time.perf_counter()-started, 2)}),
              flush=True)
        if epoch >= 15 and no_gain >= 8:
            break
    json_atomic(work / "supervised_complete.json",
                {"best_checkpoint_sha256": sha(best), "selected_epoch": -best_rank[3],
                 "last_epoch": epoch, "variant": variant, "fold": fold,
                 "intervention": intervention,
                 "validation_only_selection": True, "test_accessed": False})
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--benchmark", choices=("ictal", "omni"), required=True)
    p.add_argument("--fold", type=int, default=1)
    p.add_argument("--variant", choices=("CRST-0", "CRST-FULL"), required=True)
    p.add_argument("--ablation", choices=("A_ONLY", "NO_CHANNEL_ATTENTION"))
    p.add_argument("--spectral-cache", type=Path, required=True)
    p.add_argument("--manifest", type=Path)
    p.add_argument("--feature-cache", type=Path)
    p.add_argument("--train-val-split", type=Path)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training-lock", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    if args.ablation and args.variant != "CRST-FULL":
        raise RuntimeError("B1/B2 ablations must use the full SSL/objective recipe")
    if not torch.cuda.is_available():
        raise RuntimeError("CRST training requires server GPU")
    device = torch.device("cuda")
    torch.set_num_threads(4)
    protocol_sha, train_sha = sha(args.protocol), sha(args.training_lock)
    if args.benchmark == "ictal":
        if args.fold not in range(1, 6) or not args.manifest or not args.feature_cache:
            raise RuntimeError("Ictal requires exact fold/manifest/feature metadata")
        bank = IctalPatientBank(args.spectral_cache, args.manifest, args.feature_cache)
        train_names = bank.folds[args.fold]["fit"]
        val_names = bank.folds[args.fold]["validation"]
    else:
        if args.fold != 1 or not args.train_val_split:
            raise RuntimeError("Omni has one official train/validation split")
        bank = OmniPatientBank(args.spectral_cache, args.train_val_split)
        train_names = [name for name, role in bank.roles.items() if role == "inner_train"]
        val_names = [name for name, role in bank.roles.items() if role == "inner_val"]
    root = args.runtime / args.benchmark / f"fold{args.fold}"
    if args.ablation:
        root = root / "ablation" / args.ablation
    work = root / args.variant
    ssl_best = None
    if args.variant == "CRST-FULL":
        ssl_work = root / "SSL"
        ssl_best = run_ssl(bank, train_names, args.fold, ssl_work,
                           protocol_sha, train_sha, device,
                           intervention=args.ablation)
    best = run_supervised(bank, train_names, val_names, args.fold, args.variant,
                          work, protocol_sha, train_sha, device, ssl_best,
                          intervention=args.ablation)
    state = torch.load(best, map_location="cpu", weights_only=False)
    print(json.dumps({"status": "TRAIN_VALIDATION_COMPLETE", "benchmark": args.benchmark,
                      "fold": args.fold, "variant": args.variant,
                      "intervention": args.ablation,
                      "selected_epoch": state["epoch"],
                      "validation": state["validation"],
                      "frozen_validation_threshold": state["threshold"],
                      "test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
