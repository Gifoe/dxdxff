"""Frozen-epoch refit on all official Omni TRAIN patients, before test access.

The selected epoch and threshold come exclusively from the prior inner
train/validation run.  This script never reads official test membership or
waveforms and cannot select a new checkpoint from an outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from crst_model import CRSTiEEG
from crst_objectives import supervised_parts
from patient_bank import OmniPatientBank, to_device
from train_crst import (center_weights, json_atomic, load_patient, save_atomic,
                        seed_everything, sha, supervised_optimizer)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=("CRST-0", "CRST-FULL"), required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--split", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--training-lock", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    args = p.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("Omni refit requires the server GPU")
    torch.set_num_threads(4)
    audit = json.loads((args.runtime / "OMNI_TRAIN_CACHE_AUDIT.json").read_text())
    if not audit["pass"] or audit["test_split_accessed"]:
        raise RuntimeError("Official TRAIN cache audit is not clean")
    lock = json.loads(args.training_lock.read_text(encoding="utf-8"))
    if lock["omni_final_refit"] != (
            "after inner checkpoint/threshold freeze, refit all 141 official train patients "
            "for selected epoch count with threshold unchanged"):
        raise RuntimeError("Frozen refit rule differs")
    protocol_sha, train_sha = sha(args.protocol), sha(args.training_lock)
    selection = args.runtime / "omni" / "fold1" / args.variant / "supervised_best.pt"
    complete = args.runtime / "omni" / "fold1" / args.variant / "supervised_complete.json"
    if not selection.is_file() or not complete.is_file():
        raise RuntimeError("Inner validation selection is incomplete")
    selection_sha = sha(selection)
    selected = torch.load(selection, map_location="cpu", weights_only=False)
    if selected["protocol_sha"] != protocol_sha or selected["train_sha"] != train_sha:
        raise RuntimeError("Selected checkpoint provenance mismatch")
    chosen_epoch = int(selected["epoch"])
    threshold = selected["threshold"]
    if not 1 <= chosen_epoch <= 35 or not 0 <= threshold["threshold"] <= 1:
        raise RuntimeError("Invalid frozen epoch or threshold")
    bank = OmniPatientBank(args.cache, args.split)
    names = sorted(bank.roles)
    if len(names) != 141:
        raise RuntimeError("Expected exact 141 official TRAIN patients")
    work = args.runtime / "omni" / "final_refit" / args.variant
    work.mkdir(parents=True, exist_ok=True)
    final = work / "frozen_model.pt"
    completed = work / "refit_complete.json"
    if completed.is_file():
        marker = json.loads(completed.read_text(encoding="utf-8"))
        if (marker["checkpoint_sha256"] != sha(final) or
                marker["validation_checkpoint_sha256"] != selection_sha or
                marker["selected_epoch"] != chosen_epoch):
            raise RuntimeError("Omni final refit resume identity mismatch")
        print(json.dumps({"variant": args.variant, "status": "REFIT_REUSED",
                          "test_accessed": False}), flush=True)
        return
    device = torch.device("cuda")
    seed_everything(42001)
    model = CRSTiEEG(activation_checkpointing=False).to(device)
    ssl_path = args.runtime / "omni" / "fold1" / "SSL" / "ssl_best.pt"
    ssl_sha = None
    if args.variant == "CRST-FULL":
        if not ssl_path.is_file():
            raise RuntimeError("Train-only SSL checkpoint absent")
        source = torch.load(ssl_path, map_location=device, weights_only=False)
        if source["protocol_sha"] != protocol_sha or source["train_sha"] != train_sha:
            raise RuntimeError("SSL source provenance mismatch")
        model.load_state_dict(source["model"])
        ssl_sha = sha(ssl_path)
    centers, counts, q = center_weights(bank, names)
    last = work / "refit_last.pt"
    start, state = 1, None
    if last.is_file():
        state = torch.load(last, map_location=device, weights_only=False)
        identity = (state["protocol_sha"], state["train_sha"], state["variant"],
                    state["selection_sha"], state["selected_epoch"], state["ssl_sha"])
        expected = (protocol_sha, train_sha, args.variant, selection_sha,
                    chosen_epoch, ssl_sha)
        if identity != expected:
            raise RuntimeError("Frozen refit checkpoint resume identity mismatch")
        model.load_state_dict(state["model"])
        q = state["group_q"]
        start = int(state["epoch"]) + 1
    optimizer = supervised_optimizer(model, 1 if start <= 5 else 2)
    if state is not None and ((state["epoch"] <= 5 and start <= 5) or
                              (state["epoch"] >= 6 and start >= 6)):
        optimizer.load_state_dict(state["optimizer"])
    for epoch in range(start, chosen_epoch + 1):
        if epoch == 6:
            optimizer = supervised_optimizer(model, 2)
        model.train()
        losses = []
        by_center = defaultdict(list)
        order = sorted(names, key=lambda name: hashlib.sha256(
            f"42|supervised|1|{args.variant}|{epoch}|{name}".encode()).hexdigest())
        began = time.perf_counter()
        for ordinal, name in enumerate(order):
            seed = 420000000 + 1000000 + epoch * 10000 + ordinal
            seed_everything(seed)
            sample = to_device(load_patient(bank, name, epoch), device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(sample["patches"], sample["frequency_mask"],
                            sample["window_mask"], sample["edges"], return_aux=True)
                parts = supervised_parts(out["logits"], sample["labels"],
                                         out["record_logits"], sample["window_mask"],
                                         seed, sample.get("record_labels"))
                correction = q[centers[name]] / (counts[centers[name]] / len(order))
                loss = parts["total"] * correction
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite frozen-refit loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(float(parts["total"].detach()))
            by_center[centers[name]].append(float(parts["bce"].detach()))
        updated = {center: q[center] * math.exp(0.01 * np.mean(values))
                   for center, values in by_center.items()}
        norm = sum(updated.values())
        q = {center: value / norm for center, value in updated.items()}
        save_atomic(last, {"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                           "epoch": epoch, "group_q": q, "variant": args.variant,
                           "protocol_sha": protocol_sha, "train_sha": train_sha,
                           "selection_sha": selection_sha, "selected_epoch": chosen_epoch,
                           "ssl_sha": ssl_sha})
        print(json.dumps({"variant": args.variant, "epoch": epoch,
                          "selected_epoch": chosen_epoch, "train_loss": float(np.mean(losses)),
                          "group_q": q, "seconds": round(time.perf_counter()-began, 2),
                          "test_accessed": False}), flush=True)
    if not last.is_file():
        raise RuntimeError("Refit last checkpoint absent")
    current = torch.load(last, map_location="cpu", weights_only=False)
    if current["epoch"] != chosen_epoch:
        raise RuntimeError("Refit did not reach selected frozen epoch")
    save_atomic(final, {"model": current["model"], "variant": args.variant,
                        "epoch": chosen_epoch, "threshold": threshold,
                        "group_q": q, "protocol_sha": protocol_sha,
                        "train_sha": train_sha, "selection_sha": selection_sha,
                        "ssl_sha": ssl_sha, "official_train_patients": 141,
                        "test_accessed_during_refit": False})
    json_atomic(completed, {"variant": args.variant, "selected_epoch": chosen_epoch,
                            "frozen_threshold": threshold,
                            "validation_checkpoint_sha256": selection_sha,
                            "checkpoint_sha256": sha(final),
                            "protocol_sha256": protocol_sha,
                            "training_lock_sha256": train_sha,
                            "ssl_checkpoint_sha256": ssl_sha,
                            "official_train_patients": 141,
                            "test_accessed": False})
    print(json.dumps({"variant": args.variant, "status": "FROZEN_REFIT_COMPLETE",
                      "selected_epoch": chosen_epoch, "test_accessed": False}), flush=True)


if __name__ == "__main__":
    main()
