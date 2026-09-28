"""Train/validation descriptor reconstruction diagnostics; never use test labels."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from a1_tf import A1TFModel
from gate_tf_ictal_stage2 import sha256
from train_tf_ictal_stage1 import (AMENDMENT_SHA, BASE_LOCK_SHA, RAW_CACHE_SHA,
                                   augment, tf_collate)
from train_tf_omni import V2TFBank, make_model, wrap_collate, check_source, to_device


class MomentAudit:
    def __init__(self):
        self.n = 0
        self.sum = np.zeros(9)
        self.sum_sq = np.zeros(9)
        self.pred_sum = np.zeros(9)
        self.pred_sq = np.zeros(9)
        self.cross = np.zeros(9)
        self.abs_err = np.zeros(9)
        self.sq_err = np.zeros(9)

    def update(self, target, prediction):
        y = np.asarray(target, dtype=np.float64).reshape(-1, 9)
        p = np.asarray(prediction, dtype=np.float64).reshape(-1, 9)
        if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all():
            raise RuntimeError("Invalid A1 descriptor reconstruction tensors")
        self.n += len(y)
        self.sum += y.sum(axis=0)
        self.sum_sq += np.square(y).sum(axis=0)
        self.pred_sum += p.sum(axis=0)
        self.pred_sq += np.square(p).sum(axis=0)
        self.cross += (y*p).sum(axis=0)
        self.abs_err += np.abs(y-p).sum(axis=0)
        self.sq_err += np.square(y-p).sum(axis=0)

    def rows(self, benchmark, split, checkpoint):
        if not self.n:
            raise RuntimeError("No valid descriptor windows")
        n = self.n
        variance_y = self.sum_sq-self.sum*self.sum/n
        variance_p = self.pred_sq-self.pred_sum*self.pred_sum/n
        covariance = self.cross-self.sum*self.pred_sum/n
        return [{"benchmark": benchmark, "split": split, "checkpoint": checkpoint,
                 "descriptor_index": idx, "window_channel_samples": n,
                 "mae": self.abs_err[idx]/n,
                 "correlation": covariance[idx]/np.sqrt(variance_y[idx]*variance_p[idx])
                 if variance_y[idx] > 0 and variance_p[idx] > 0 else np.nan,
                 "r2": 1-self.sq_err[idx]/variance_y[idx] if variance_y[idx] > 0 else np.nan}
                for idx in range(9)]


def accumulate(model, batch, audit):
    output = model(batch)["tf_descriptor_prediction"]
    valid = batch["window_mask"][:, :, :, None] & batch["seizure_channel_mask"][:, :, None, :]
    audit.update(batch["b0_features"][..., :9][valid].detach().cpu().numpy(),
                 output[valid].detach().cpu().numpy())


def omni(a):
    if sha256(a.lock) != BASE_LOCK_SHA:
        raise RuntimeError("Unified A1-TF protocol changed")
    historical_collate, Model = check_source()
    bank = V2TFBank(a.cohort, a.features, a.tf_features, a.train_val_split,
                    a.v2_protocol, a.lock)
    fit = sorted(p for p in bank.patients if bank.roles[p] == "inner_train")
    val = sorted(p for p in bank.patients if bank.roles[p] == "inner_val")
    if (len(fit), len(val)) != (112, 29):
        raise RuntimeError("Official train inner partition changed")
    with np.load(a.omni_runtime / "inner/normalizer.npz") as data:
        mean, std, tf_mean, tf_std = [data[name] for name in ("mean", "std", "tf_mean", "tf_std")]
    bank.set_tf_normalizer(tf_mean, tf_std)
    collate = wrap_collate(historical_collate)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(Model, collate, bank, fit[0], mean, std, device)
    state = torch.load(a.omni_runtime / "inner/best.pt", map_location=device, weights_only=False)
    if state["protocol_sha256"] != BASE_LOCK_SHA:
        raise RuntimeError("Wrong N1 inner checkpoint")
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    rows = []
    with torch.inference_mode():
        for split, patients in (("inner_train", fit), ("inner_validation", val)):
            audit = MomentAudit()
            for number, patient in enumerate(patients, 1):
                batch = to_device(collate([bank.example(patient, mean, std, epoch=None)]), device)
                accumulate(model, batch, audit)
                if number % 25 == 0:
                    print(f"N1 reconstruction {split} {number}/{len(patients)}", flush=True)
            rows += audit.rows("Omni_interictal", split, f"inner_selected_epoch_{state['epoch']}")
    return rows


def ictal(a):
    if sha256(a.lock) != BASE_LOCK_SHA or sha256(a.amendment) != AMENDMENT_SHA or \
            sha256(a.raw_cache) != RAW_CACHE_SHA:
        raise RuntimeError("Ictal source/protocol changed")
    for key in ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST",
                "R1_HLV_RUNTIME", "A1_A2_RUNTIME"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing A1 source environment {key}")
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "neuroez_c"))
    sys.path.insert(0, str(source / "r1_hlv_ictal_dynamics_seed42_v1/code"))
    import exp_ez_hybrid as core
    from neuroez_c.dual_view_data import RawAlignmentStore
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args
    assert_sources()
    install_interleaved_hlv_view()
    rows = []
    gate = json.loads(a.gate_audit.read_text(encoding="utf-8"))
    for fold in range(1, 6):
        args = make_args("R0", a.ictal_runtime / "scratch")
        exp = core.Exp_EZHybridLocalization(args)
        split = next(s for s in exp.outer_splits if int(s["fold_idx"]) == fold)
        store = RawAlignmentStore(exp.run_records,
                                  feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
                                  raw_cache_path=a.raw_cache,
                                  raw_target_samples=500, raw_target_sampling_rate=250.0)
        store.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1,
                                     expected_patients=80)
        exp.raw_alignment_store = store
        exp.use_n6_dual_view_ema = True
        fit, val, _, _ = exp._build_datasets(list(split["fit_subjects"]),
                                             list(split["validation_subjects"]), [])
        exp.use_n6_dual_view_ema = False
        for dataset in (fit, val):
            augment(dataset)
        with np.load(a.ictal_runtime / f"fold_{fold}/stage1/tf_normalizer.npz") as data:
            mean, std = data["mean"], data["std"]
        for dataset in (fit, val):
            for example in dataset.patient_examples:
                example["tf_log_power"] = [(tf - mean)/std for tf in example["tf_log_power"]]
        collate = tf_collate(exp.runtime["collate_fn"])
        loader = DataLoader(fit, batch_size=2, shuffle=False, num_workers=0, collate_fn=collate)
        core._set_random_seed(42 + fold)
        base = exp.runtime["model_cls"](args).to(exp.device)
        exp._dry_initialize_lazy_layers(base, loader)
        stage = 2 if fold in gate["eligible_folds"] else 1
        state = torch.load(a.ictal_runtime / f"fold_{fold}/stage{stage}/last.pt",
                           map_location=exp.device, weights_only=False)
        if (state["fold"], state["stage"], state["epoch"]) != (fold, stage, 15):
            raise RuntimeError("Wrong I1 diagnostic checkpoint")
        model = A1TFModel(base).to(exp.device)
        model.load_state_dict(state["model"], strict=True)
        model.eval()
        with torch.inference_mode():
            for split_name, dataset in (("fold_fit", fit), ("fold_validation", val)):
                audit = MomentAudit()
                for batch in DataLoader(dataset, batch_size=2, shuffle=False,
                                        num_workers=0, collate_fn=collate):
                    accumulate(model, core._move_tensors_to_device(batch, exp.device), audit)
                rows += audit.rows("Ictal", f"fold_{fold}_{split_name}",
                                   f"stage{stage}_endpoint_epoch15_not_VLOO_selected")
        print(f"I1 reconstruction fold={fold}/5 stage_endpoint={stage}", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=("ictal", "omni"), required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--amendment", type=Path)
    parser.add_argument("--raw-cache", type=Path)
    parser.add_argument("--gate-audit", type=Path)
    parser.add_argument("--ictal-runtime", type=Path)
    parser.add_argument("--cohort", type=Path)
    parser.add_argument("--features", type=Path)
    parser.add_argument("--tf-features", type=Path)
    parser.add_argument("--train-val-split", type=Path)
    parser.add_argument("--v2-protocol", type=Path)
    parser.add_argument("--omni-runtime", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    rows = ictal(a) if a.benchmark == "ictal" else omni(a)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(a.output, index=False)
    print(f"TF_RECONSTRUCTION_AUDIT {a.benchmark} rows={len(rows)}", flush=True)


if __name__ == "__main__":
    main()
