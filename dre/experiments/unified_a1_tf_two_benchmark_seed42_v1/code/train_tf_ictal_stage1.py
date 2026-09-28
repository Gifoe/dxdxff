"""I1 fold-local 15-epoch preservation stage; FIT subjects only.

This stores score-only snapshots for the historical 13 validation patients.
Their labels are not indexed for training or gate selection by this script.
The historical loader necessarily materializes validation labels; this known
limitation is reported, not disguised as literal outcome sealing.
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
from tf_preprocess import log_frequency_stft_windows, fit_train_frequency_normalizer

HERE = Path(__file__).resolve().parent
BASE_LOCK_SHA = "ace2017e01d0d21e3dda8ddd4d42551366704430b8a10f3666f955150150630d"
AMENDMENT_SHA = "130925a856d229168244fef17d86144e11ca30312b39dacfc0b9d6e2ce2e31bb"
RAW_CACHE_SHA = "011d7ffaa55469c9f34259be04b7136a987d6d95f0ce9cc443a06751557db733"


def augment(dataset, mean=None, std=None):
    for example in dataset.patient_examples:
        if len(example["raw_windows"]) != len(example["b0_features"]):
            raise RuntimeError("A1/raw record count mismatch")
        items = []
        for raw, raw_mask, feature, channel_mask in zip(example["raw_windows"],
                                                        example["raw_window_mask"],
                                                        example["b0_features"],
                                                        example["seizure_channel_mask"]):
            if raw.shape[:2] != feature.shape[:2] or raw.shape[-1] != 500:
                raise RuntimeError("Ictal raw/A1 window or channel mismatch")
            active = np.asarray(channel_mask, dtype=bool)
            if not np.asarray(raw_mask, dtype=bool)[:, active].all():
                raise RuntimeError("Ictal raw cache does not cover every active A1 window")
            tf = log_frequency_stft_windows(raw, 250)
            items.append(tf)
        example["tf_log_power"] = items
    if mean is None or std is None:
        return
    for example in dataset.patient_examples:
        example["tf_log_power"] = [(value - mean) / std
                                    for value in example["tf_log_power"]]


def tf_collate(base):
    def collate(examples):
        batch = base(examples)
        b, records, windows, channels, _ = batch["b0_features"].shape
        tf = torch.zeros((b, records, windows, channels, 32), dtype=torch.float32)
        for index, example in enumerate(examples):
            for record_index, value in enumerate(example["tf_log_power"]):
                w, c, n = value.shape
                if n != 32 or w > windows or c > channels:
                    raise RuntimeError("Ictal TF collate shape mismatch")
                tf[index, record_index, :w, :c] = torch.from_numpy(value)
        batch["tf_log_power"] = tf
        # The raw windows were consumed by the fixed STFT. Do not transfer them
        # to GPU or allow an auxiliary A1 branch to use them.
        for key in ("raw_windows", "raw_window_mask", "raw_seizure_channel_mask"):
            batch.pop(key, None)
        return batch
    return collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--amendment", type=Path, required=True)
    args = p.parse_args()
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
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args, sha256
    from train_stage_b import score_only_snapshot

    if sha256(args.lock) != BASE_LOCK_SHA or sha256(args.amendment) != AMENDMENT_SHA:
        raise RuntimeError("I1 protocol or pre-outcome selection amendment changed")
    if sha256(args.raw_cache) != RAW_CACHE_SHA:
        raise RuntimeError("Frozen ictal raw cache changed")
    assert_sources()
    install_interleaved_hlv_view()
    model_args = make_args("R0", args.runtime / "scratch")
    exp = core.Exp_EZHybridLocalization(model_args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5 or len(exp.run_records) != 256:
        raise RuntimeError("Historical A1 cohort changed")
    split = next(s for s in exp.outer_splits if int(s["fold_idx"]) == args.fold)
    fit, val = list(split["fit_subjects"]), list(split["validation_subjects"])
    if len(val) != 13 or set(fit) & set(val) or (set(fit) | set(val)) & set(split["test_subjects"]):
        raise RuntimeError("Historical fold role overlap")
    store = RawAlignmentStore(exp.run_records,
                              feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
                              raw_cache_path=args.raw_cache,
                              raw_target_samples=500, raw_target_sampling_rate=250.0)
    store.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1,
                                 expected_patients=80)
    exp.raw_alignment_store = store
    exp.use_n6_dual_view_ema = True  # dataset construction only
    train_set, val_set, _, _ = exp._build_datasets(fit, val, [])
    exp.use_n6_dual_view_ema = False
    augment(train_set)
    mean, std = fit_train_frequency_normalizer(
        [tf[:, np.asarray(mask, dtype=bool), :]
         for example in train_set.patient_examples
         for tf, mask in zip(example["tf_log_power"], example["seizure_channel_mask"])])
    augment(val_set)
    for dataset in (train_set, val_set):
        for example in dataset.patient_examples:
            example["tf_log_power"] = [(tf - mean) / std for tf in example["tf_log_power"]]
    folder = args.runtime / f"fold_{args.fold}" / "stage1"
    folder.mkdir(parents=True, exist_ok=True)
    normalizer = folder / "tf_normalizer.npz"
    if normalizer.is_file():
        with np.load(normalizer) as prior:
            if not np.array_equal(prior["mean"], mean) or not np.array_equal(prior["std"], std):
                raise RuntimeError("I1 train-only TF normalizer changed on resume")
    else:
        np.savez(normalizer, mean=mean, std=std)
    collate = tf_collate(exp.runtime["collate_fn"])
    def loader(dataset, shuffle):
        return DataLoader(dataset, batch_size=2, shuffle=shuffle, num_workers=0,
                          collate_fn=collate, pin_memory=exp.device.type == "cuda")
    train_loader, val_loader = loader(train_set, True), loader(val_set, False)
    core._set_random_seed(42 + args.fold)
    base = exp.runtime["model_cls"](model_args).to(exp.device)
    exp._dry_initialize_lazy_layers(base, train_loader)
    checkpoint = Path(os.environ["A1_A2_RUNTIME"]) / "A1" / f"fold_{args.fold}" / "epoch_30.pt"
    state = torch.load(checkpoint, map_location=exp.device, weights_only=False)
    if state["variant"] != "A1" or state["epoch"] != 30 or state["fold"] != args.fold:
        raise RuntimeError("Wrong historical A1 FIT checkpoint")
    base.load_state_dict(state["model_state_dict"], strict=True)
    base.eval()
    teacher = copy.deepcopy(base).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    model = A1TFModel(base).to(exp.device)
    freeze_for_ictal_stage(model, 1)
    mini = core._move_tensors_to_device(next(iter(train_loader)), exp.device)
    model.eval()
    with torch.no_grad():
        error = float((model(mini)["logits"] - teacher(mini)["logits"]).abs().max())
    if error >= 1e-6:
        raise RuntimeError(f"Real FIT batch alpha=0 A1 replay failed: {error}")
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=1e-4, weight_decay=1e-3)
    last = folder / "last.pt"
    start = 1
    if last.is_file():
        previous = torch.load(last, map_location=exp.device, weights_only=False)
        if previous["fold"] != args.fold or previous["stage"] != 1 or previous["epoch"] > 15:
            raise RuntimeError("Wrong I1 Stage1 resume checkpoint")
        model.load_state_dict(previous["model"], strict=True)
        optimizer.load_state_dict(previous["optimizer"])
        start = previous["epoch"] + 1
    for epoch in range(start, 16):
        core._set_random_seed(42 * 100000 + args.fold * 1000 + epoch)
        model.train()
        model.a1.eval()  # frozen A1 dropout must not perturb the teacher path
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
                raise RuntimeError("Nonfinite I1 Stage1 loss")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        score = score_only_snapshot(exp, model, val_loader)
        snapshot = folder / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"
        with snapshot.open("wb") as stream:
            pickle.dump(score, stream, protocol=pickle.HIGHEST_PROTOCOL)
        payload = {"fold": args.fold, "stage": 1, "epoch": epoch,
                   "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                   "base_lock_sha256": BASE_LOCK_SHA, "amendment_sha256": AMENDMENT_SHA}
        temp = folder / "last.pt.tmp"
        torch.save(payload, temp)
        os.replace(temp, last)
        print(f"I1 fold={args.fold} stage=1 epoch={epoch}/15 loss={np.mean(losses):.5f} "
              f"alpha={float(model.alpha.detach()):.6f} validation_score_patients={len(score)}",
              flush=True)
    complete = {"fold": args.fold, "stage1_complete": True, "epochs": 15,
                "real_batch_alpha_zero_max_logit_difference": error,
                "train_patients": len(fit), "validation_patients": len(val),
                "validation_target_labels_indexed_for_selection": False,
                "historical_loader_materialized_validation_labels": True}
    (folder / "summary.json").write_text(json.dumps(complete, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
