"""Eligible I1 folds: 15 FIT-only upper-A1 fine-tuning epochs.

No target validation label is indexed here. All 13 score grids are saved for
later leave-one-validation-patient-out checkpoint/threshold selection.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from a1_tf import A1TFModel, freeze_for_ictal_stage
from train_tf_ictal_stage1 import (AMENDMENT_SHA, BASE_LOCK_SHA, RAW_CACHE_SHA,
                                   augment, tf_collate)
from gate_tf_ictal_stage2 import sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    parser.add_argument("--raw-cache", type=Path, required=True)
    parser.add_argument("--runtime", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--amendment", type=Path, required=True)
    parser.add_argument("--gate-audit", type=Path, required=True)
    a = parser.parse_args()
    if sha256(a.lock) != BASE_LOCK_SHA or sha256(a.amendment) != AMENDMENT_SHA:
        raise RuntimeError("I1 protocol/amendment changed")
    if sha256(a.raw_cache) != RAW_CACHE_SHA:
        raise RuntimeError("I1 raw cache changed")
    gate = json.loads(a.gate_audit.read_text(encoding="utf-8"))
    if gate["protocol_sha256"] != BASE_LOCK_SHA or gate["amendment_sha256"] != AMENDMENT_SHA:
        raise RuntimeError("Wrong Stage-2 gate provenance")
    if a.fold not in gate["eligible_folds"]:
        print(f"I1 fold={a.fold} Stage2 correctly skipped by frozen gate", flush=True)
        return
    if sha256(a.runtime / "private/ICTAL_STAGE2_GATE_PRIVATE.csv") != gate["private_gate_sha256"]:
        raise RuntimeError("Stage-2 private gate changed")
    for key in ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST",
                "R1_HLV_RUNTIME", "A1_A2_RUNTIME"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing private A1 environment variable {key}")
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "neuroez_c"))
    sys.path.insert(0, str(source / "r1_hlv_ictal_dynamics_seed42_v1/code"))
    sys.path.insert(0, str(source / "a1_a2_patient_equal_objective_seed42_v1/code"))
    sys.path.insert(0, str(source / "rawtiny_patient_relative_seed42_v1/code"))
    import exp_ez_hybrid as core
    from neuroez_c.dual_view_data import RawAlignmentStore
    from objectives import patient_equal_weighted_bce_loss
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args
    from train_stage_b import score_only_snapshot

    assert_sources()
    install_interleaved_hlv_view()
    model_args = make_args("R0", a.runtime / "scratch")
    exp = core.Exp_EZHybridLocalization(model_args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5 or len(exp.run_records) != 256:
        raise RuntimeError("Historical I1 cohort changed")
    split = next(s for s in exp.outer_splits if int(s["fold_idx"]) == a.fold)
    fit, val = list(split["fit_subjects"]), list(split["validation_subjects"])
    if len(val) != 13 or set(fit) & set(val) or (set(fit) | set(val)) & set(split["test_subjects"]):
        raise RuntimeError("Historical fold role overlap")
    store = RawAlignmentStore(exp.run_records,
                              feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
                              raw_cache_path=a.raw_cache,
                              raw_target_samples=500, raw_target_sampling_rate=250.0)
    store.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1,
                                 expected_patients=80)
    exp.raw_alignment_store = store
    exp.use_n6_dual_view_ema = True
    train_set, val_set, _, _ = exp._build_datasets(fit, val, [])
    exp.use_n6_dual_view_ema = False
    augment(train_set)
    augment(val_set)
    first = a.runtime / f"fold_{a.fold}" / "stage1"
    if not json.loads((first / "summary.json").read_text(encoding="utf-8"))["stage1_complete"]:
        raise RuntimeError("Cannot start Stage 2 without full Stage 1")
    with np.load(first / "tf_normalizer.npz") as normalizer:
        mean, std = normalizer["mean"], normalizer["std"]
    for dataset in (train_set, val_set):
        for example in dataset.patient_examples:
            example["tf_log_power"] = [(tf - mean) / std for tf in example["tf_log_power"]]
    collate = tf_collate(exp.runtime["collate_fn"])
    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=2, shuffle=shuffle, num_workers=0,
                          collate_fn=collate, pin_memory=exp.device.type == "cuda")
    train_loader, val_loader = loader(train_set, True), loader(val_set, False)
    core._set_random_seed(42 + a.fold)
    base = exp.runtime["model_cls"](model_args).to(exp.device)
    exp._dry_initialize_lazy_layers(base, train_loader)
    historical = Path(os.environ["A1_A2_RUNTIME"]) / "A1" / f"fold_{a.fold}" / "epoch_30.pt"
    initial = torch.load(historical, map_location=exp.device, weights_only=False)
    if initial["variant"] != "A1" or initial["fold"] != a.fold or initial["epoch"] != 30:
        raise RuntimeError("Wrong source A1 checkpoint")
    base.load_state_dict(initial["model_state_dict"], strict=True)
    base.eval()
    teacher = copy.deepcopy(base).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = A1TFModel(base).to(exp.device)
    prior = torch.load(first / "last.pt", map_location=exp.device, weights_only=False)
    if (prior["stage"], prior["fold"], prior["epoch"], prior["base_lock_sha256"],
            prior["amendment_sha256"]) != (1, a.fold, 15, BASE_LOCK_SHA, AMENDMENT_SHA):
        raise RuntimeError("Stage 1 checkpoint identity mismatch")
    model.load_state_dict(prior["model"], strict=True)
    freeze_for_ictal_stage(model, 2)
    tf_parameters = [p for name, p in model.named_parameters()
                     if p.requires_grad and not name.startswith("a1.")]
    upper_parameters = [p for name, p in model.named_parameters()
                        if p.requires_grad and name.startswith("a1.")]
    if not tf_parameters or not upper_parameters:
        raise RuntimeError("I1 Stage 2 trainable groups incomplete")
    optimizer = torch.optim.AdamW([
        {"params": tf_parameters, "lr": 1e-4},
        {"params": upper_parameters, "lr": 1e-5}], weight_decay=1e-3)
    folder = a.runtime / f"fold_{a.fold}" / "stage2"
    folder.mkdir(parents=True, exist_ok=True)
    last = folder / "last.pt"
    start = 1
    if last.is_file():
        previous = torch.load(last, map_location=exp.device, weights_only=False)
        if (previous["stage"], previous["fold"], previous["base_lock_sha256"],
                previous["amendment_sha256"]) != (2, a.fold, BASE_LOCK_SHA, AMENDMENT_SHA):
            raise RuntimeError("Wrong Stage 2 resume checkpoint")
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        start = previous["epoch"] + 1
    for epoch in range(start, 16):
        core._set_random_seed(42 * 100000 + a.fold * 1000 + 15 + epoch)
        model.train()
        model.a1.eval()
        for module in (model.a1.temporal_encoder, model.a1.seizure_aggregator,
                       model.a1.channel_classifier):
            module.train()
        losses = []
        for raw_batch in train_loader:
            batch = core._move_tensors_to_device(raw_batch, exp.device)
            optimizer.zero_grad(set_to_none=True)
            out = model(batch)
            with torch.no_grad():
                old = teacher(batch)["logits"]
            bce = patient_equal_weighted_bce_loss(out["logits"], batch["labels"],
                                                   batch["labels_ez"], batch["channel_mask"])
            valid = batch["window_mask"][:, :, :, None] & batch["seizure_channel_mask"][:, :, None, :]
            desc = F.smooth_l1_loss(out["tf_descriptor_prediction"][valid],
                                    batch["b0_features"][..., :9][valid])
            active = batch["channel_mask"]
            anchor = F.mse_loss((-out["logits"])[active], (-old)[active])
            loss = bce + 0.1 * desc + 0.5 * anchor
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite I1 Stage 2 loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        snapshot = score_only_snapshot(exp, model, val_loader)
        with (folder / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl").open("wb") as handle:
            pickle.dump(snapshot, handle, protocol=pickle.HIGHEST_PROTOCOL)
        payload = {"fold": a.fold, "stage": 2, "epoch": epoch,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "base_lock_sha256": BASE_LOCK_SHA, "amendment_sha256": AMENDMENT_SHA}
        temp = folder / "last.pt.tmp"
        torch.save(payload, temp)
        os.replace(temp, last)
        print(f"I1 fold={a.fold} stage=2 epoch={epoch}/15 loss={np.mean(losses):.5f} "
              f"alpha={float(model.alpha.detach()):.6f} validation_score_patients={len(snapshot)}",
              flush=True)
    (folder / "summary.json").write_text(json.dumps({"fold": a.fold, "stage2_complete": True,
        "epochs": 15, "train_patients": len(fit), "validation_patients": len(val),
        "target_labels_indexed_during_training": False}, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
