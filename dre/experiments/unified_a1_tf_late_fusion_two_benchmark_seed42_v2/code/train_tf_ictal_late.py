"""Independent TF-only FIT training for a fixed ictal fold.

Scores for the 13 validation patients are snapshotted without consulting their
labels here. Target-excluded epoch/beta selection happens in a separate pass.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "unified_a1_tf_two_benchmark_seed42_v1" / "code"))
from train_tf_ictal_stage1 import augment, tf_collate  # noqa: E402
from tf_preprocess import fit_train_frequency_normalizer  # noqa: E402
from late_tf import TFScorer  # noqa: E402
from train_tf_omni_late import sha, save_json  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    a = p.parse_args()
    for key in ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST",
                "R1_HLV_RUNTIME", "A1_A2_RUNTIME"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing private source variable {key}")
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    for folder in (source, source / "neuroez_c", source / "r1_hlv_ictal_dynamics_seed42_v1/code",
                   source / "a1_a2_patient_equal_objective_seed42_v1/code",
                   source / "rawtiny_patient_relative_seed42_v1/code"):
        sys.path.insert(0, str(folder))
    import exp_ez_hybrid as core
    from neuroez_c.dual_view_data import RawAlignmentStore
    from objectives import patient_equal_weighted_bce_loss
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args

    assert_sources()
    install_interleaved_hlv_view()
    model_args = make_args("R0", a.runtime / "scratch")
    exp = core.Exp_EZHybridLocalization(model_args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5 or len(exp.run_records) != 256:
        raise RuntimeError("Historical ictal cohort changed")
    split = next(s for s in exp.outer_splits if int(s["fold_idx"]) == a.fold)
    fit, val = list(split["fit_subjects"]), list(split["validation_subjects"])
    if len(val) != 13 or set(fit) & set(val) or (set(fit) | set(val)) & set(split["test_subjects"]):
        raise RuntimeError("Fixed-fold role overlap")
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
    mean, std = fit_train_frequency_normalizer([
        tf[:, np.asarray(mask, dtype=bool), :]
        for example in train_set.patient_examples
        for tf, mask in zip(example["tf_log_power"], example["seizure_channel_mask"])])
    augment(val_set)
    for dataset in (train_set, val_set):
        for example in dataset.patient_examples:
            example["tf_log_power"] = [(tf - mean) / std for tf in example["tf_log_power"]]
    folder = a.runtime / f"fold_{a.fold}"
    folder.mkdir(parents=True, exist_ok=True)
    normalizer = folder / "normalizer.npz"
    if normalizer.is_file():
        with np.load(normalizer) as old:
            if not np.array_equal(old["mean"], mean) or not np.array_equal(old["std"], std):
                raise RuntimeError("Train-only frequency normalizer changed")
    else:
        np.savez(normalizer, mean=mean, std=std)
    collate = tf_collate(exp.runtime["collate_fn"])
    def loader(data, shuffle):
        return DataLoader(data, batch_size=2, shuffle=shuffle, num_workers=0,
                          collate_fn=collate, pin_memory=exp.device.type == "cuda")
    train_loader, val_loader = loader(train_set, True), loader(val_set, False)
    core._set_random_seed(42 + a.fold)
    model = TFScorer().to(exp.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    last = folder / "last.pt"
    start = 1
    if last.is_file():
        old = torch.load(last, map_location=exp.device, weights_only=False)
        if old["fold"] != a.fold or old["protocol_sha256"] != sha(a.lock) or old["epoch"] > 30:
            raise RuntimeError("Ictal TF resume provenance mismatch")
        model.load_state_dict(old["model"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
        start = int(old["epoch"]) + 1
    for epoch in range(start, 31):
        core._set_random_seed(42 * 100000 + a.fold * 1000 + epoch)
        model.train()
        losses = []
        for raw in train_loader:
            batch = core._move_tensors_to_device(raw, exp.device)
            optimizer.zero_grad(set_to_none=True)
            result = model(batch)
            loss = patient_equal_weighted_bce_loss(-result["margin_tf"], batch["labels"],
                                                   batch["labels_ez"], batch["channel_mask"])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite ictal TF-only BCE")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        model.eval()
        snap = {}
        with torch.inference_mode():
            for raw in val_loader:
                batch = core._move_tensors_to_device(raw, exp.device)
                score = model(batch)["margin_tf"].detach().cpu().numpy()
                mask = batch["channel_mask"].detach().cpu().numpy().astype(bool)
                for i, sid in enumerate(batch["subject_id"]):
                    if sid in snap:
                        raise RuntimeError("Duplicate validation patient")
                    snap[sid] = score[i, mask[i]].astype(np.float32)
        if len(snap) != 13:
            raise RuntimeError("Ictal validation patient denominator changed")
        target = folder / f"epoch_{epoch:02d}_TF_SCORES_PRIVATE.pkl"
        tmp = target.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(snap, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, target)
        state = {"fold": a.fold, "epoch": epoch, "model": model.state_dict(),
                 "optimizer": optimizer.state_dict(), "protocol_sha256": sha(a.lock)}
        tmp = last.with_suffix(".pt.tmp")
        torch.save(state, tmp)
        os.replace(tmp, last)
        print(f"ictal fold={a.fold} epoch={epoch}/30 loss={np.mean(losses):.6f} val_scores=13", flush=True)
    save_json(folder / "training_audit.json", {"fold": a.fold, "fit_patients": len(fit),
              "validation_patients": len(val), "epochs": 30, "protocol_sha256": sha(a.lock),
              "normalizer_sha256": sha(normalizer), "target_labels_used_for_training": False,
              "historical_loader_materialized_validation_labels": True,
              "target_labels_used_for_own_selection": False})


if __name__ == "__main__":
    main()
