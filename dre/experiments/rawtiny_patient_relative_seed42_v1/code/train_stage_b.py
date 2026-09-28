"""Frozen-source 30-epoch training with private resumable score-only snapshots.

The validation loader materializes labels because of legacy dataset design,
but this runner never indexes them. Label-using VLOO is a separate final pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from models import HybridA1RawModel, RawTinyPatientModel, parameter_count

VARIANTS = ("M1_RAWTINY_NOPR", "M2_RAWTINY_PR", "M3_HYBRID_PR")
LOCK_SHA = "028a951709f095459c39986f3823110fa9826078f24290d6afe9017c2ed621d2"


def sha(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for chunk in iter(lambda: h.read(4 * 1024 * 1024), b""):
            d.update(chunk)
    return d.hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    temp.replace(path)


def raw_init_hash(model: torch.nn.Module) -> str:
    d = hashlib.sha256()
    for name, value in sorted(model.raw_encoder.state_dict().items()):
        d.update(name.encode())
        d.update(value.detach().cpu().contiguous().numpy().tobytes())
    return d.hexdigest()


def score_only_snapshot(exp, model, loader) -> dict:
    from exp_ez_hybrid import _move_tensors_to_device
    model.eval()
    patients = {}
    with torch.no_grad():
        for raw in loader:
            batch = _move_tensors_to_device(raw, exp.device)
            outputs = model(batch)
            mask = batch["channel_mask"].detach().cpu().numpy().astype(bool)
            score_ez = outputs["score_ez"].detach().cpu().numpy()
            score_nez = outputs["score_nez"].detach().cpu().numpy()
            logits = outputs["logits"].detach().cpu().numpy()
            representation = outputs["patient_channel_embedding"].detach().cpu().numpy()
            for index, sid in enumerate(batch["subject_id"]):
                if sid in patients:
                    raise RuntimeError("Duplicate validation patient")
                valid = mask[index]
                patients[sid] = {
                    "score_ez": score_ez[index][valid].astype(np.float32),
                    "score_nez": score_nez[index][valid].astype(np.float32),
                    "logit_nez": logits[index][valid].astype(np.float32),
                    "R4": representation[index][valid].astype(np.float32),
                    "n_channels": int(valid.sum()),
                }
    if len(patients) != 13:
        raise RuntimeError(f"Expected 13 score-only validation patients, got {len(patients)}")
    return patients


def run() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fold", type=int, choices=range(1, 6), required=True)
    p.add_argument("--variant", choices=VARIANTS, required=True)
    p.add_argument("--raw-cache", type=Path, required=True)
    p.add_argument("--runtime", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    args = p.parse_args()
    if sha(args.lock) != LOCK_SHA:
        raise RuntimeError("Protocol lock changed")
    if not json.loads(args.lock.read_text(encoding="utf-8"))["evaluation"]["no_outer_test"]:
        raise RuntimeError("Outer-test policy changed")
    for key in ("R1_HLV_SOURCE_ROOT", "R1_HLV_WINDOW_CACHE", "R1_HLV_FIXED_MANIFEST", "R1_HLV_RUNTIME", "A1_A2_RUNTIME"):
        if not os.environ.get(key):
            raise RuntimeError(f"Missing private environment variable {key}")
    source = Path(os.environ["R1_HLV_SOURCE_ROOT"])
    sys.path.insert(0, str(source))
    sys.path.insert(0, str(source / "neuroez_c"))
    sys.path.insert(0, str(source / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
    sys.path.insert(0, str(source / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
    import exp_ez_hybrid as core
    from neuroez_c.dual_view_data import RawAlignmentStore
    from objectives import patient_equal_weighted_bce_loss
    from run_matched import assert_sources, install_interleaved_hlv_view, make_args

    assert_sources()
    install_interleaved_hlv_view()
    model_args = make_args("R0", args.runtime / "private_scratch")
    exp = core.Exp_EZHybridLocalization(model_args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("Frozen A1 cohort changed")
    split = next(s for s in exp.outer_splits if int(s["fold_idx"]) == args.fold)
    fit = list(split["fit_subjects"])
    val = list(split["validation_subjects"])
    # The frozen A1 folds are not all 51/13/16: their FIT counts are
    # 51, 51, 50, 52, 51 (see A1 CLASS_SUPPORT_AUDIT.json).  The previous
    # blanket 51 check incorrectly rejected folds 3 and 4 before training.
    expected_fit = {1: 51, 2: 51, 3: 50, 4: 52, 5: 51}
    test = set(split["test_subjects"])
    if (len(fit) != expected_fit[args.fold] or len(val) != 13 or
            len(fit) + len(val) + len(test) != 80 or
            set(fit) & set(val) or (set(fit) | set(val)) & test):
        raise RuntimeError("Frozen fold role mismatch")
    store = RawAlignmentStore(
        exp.run_records,
        feature_cache_path=os.environ["R1_HLV_WINDOW_CACHE"],
        raw_cache_path=args.raw_cache,
        raw_target_samples=500,
        raw_target_sampling_rate=250.0,
    )
    store.assert_formal_coverage(min_channel_match_rate=1, min_window_match_rate=1, expected_patients=80)
    exp.raw_alignment_store = store
    exp.use_n6_dual_view_ema = True  # dataset construction only
    train_set, val_set, _, normalizer = exp._build_datasets(fit, val, [])
    exp.use_n6_dual_view_ema = False
    train_loader = exp._make_loader(train_set, shuffle=True, batch_size=2)
    val_loader = exp._make_loader(val_set, shuffle=False, batch_size=2)
    core._set_random_seed(42 + args.fold)
    if args.variant in VARIANTS[:2]:
        model = RawTinyPatientModel(patient_relative=args.variant == VARIANTS[1]).to(exp.device)
    else:
        base = exp.runtime["model_cls"](model_args).to(exp.device)
        exp._dry_initialize_lazy_layers(base, train_loader)
        ckpt = Path(os.environ["A1_A2_RUNTIME"]) / "A1" / f"fold_{args.fold}" / "epoch_30.pt"
        state = torch.load(ckpt, map_location=exp.device, weights_only=False)
        if state["variant"] != "A1" or state["epoch"] != 30 or state["fold"] != args.fold:
            raise RuntimeError("M3 source A1 checkpoint identity mismatch")
        base.load_state_dict(state["model_state_dict"], strict=True)
        model = HybridA1RawModel(base).to(exp.device)
    folder = args.runtime / args.variant / f"fold_{args.fold}"
    folder.mkdir(parents=True, exist_ok=True)
    initial_hash = raw_init_hash(model)
    init_file = args.runtime / f"fold_{args.fold}_raw_initializations.json"
    if init_file.is_file():
        prior = json.loads(init_file.read_text(encoding="utf-8"))
    else:
        prior = {}
    if args.variant in VARIANTS[:2]:
        other = VARIANTS[1] if args.variant == VARIANTS[0] else VARIANTS[0]
        if other in prior and prior[other] != initial_hash:
            raise RuntimeError("M1/M2 raw encoder initialization mismatch")
    prior[args.variant] = initial_hash
    atomic_json(init_file, prior)

    def compute_loss(self, outputs, batch, ez_weight, *, split_name="train"):
        if float(ez_weight) != 2.0:
            raise RuntimeError("A1 class weighting changed")
        loss = patient_equal_weighted_bce_loss(outputs["logits"], batch["labels"], batch["labels_ez"], batch["channel_mask"])
        return loss, {"bce": float(loss.detach()), "patient_equal_bce": float(loss.detach())}

    import types
    exp._compute_loss = types.MethodType(compute_loss, exp)
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-4, weight_decay=1e-3)
    ez_weight = torch.tensor(2.0, device=exp.device)
    summary = {"variant": args.variant, "fold": args.fold, "trainable_parameters": parameter_count(model),
               "raw_init_sha256": initial_hash, "epochs": 30, "epoch_rows": []}
    for epoch in range(1, 31):
        ckpt_path = folder / f"epoch_{epoch:02d}.pt"
        score_path = folder / f"epoch_{epoch:02d}_SCORES_PRIVATE.pkl"
        row_path = folder / f"epoch_{epoch:02d}.json"
        if ckpt_path.is_file() and score_path.is_file() and row_path.is_file():
            state = torch.load(ckpt_path, map_location=exp.device, weights_only=False)
            row = json.loads(row_path.read_text(encoding="utf-8"))
            if state["lock_sha"] != LOCK_SHA or state["variant"] != args.variant or state["fold"] != args.fold or state["epoch"] != epoch:
                raise RuntimeError("Resume identity mismatch")
            if sha(score_path) != row["score_sha256"]:
                raise RuntimeError("Resume score snapshot hash mismatch")
            model.load_state_dict(state["model"], strict=True)
            opt.load_state_dict(state["optimizer"])
        elif any(p.exists() for p in (ckpt_path, score_path, row_path)):
            raise RuntimeError("Partial epoch artifacts require engineering inspection; refusing overwrite")
        else:
            core._set_random_seed(42 * 100000 + args.fold * 1000 + epoch)
            exp.current_epoch = epoch
            start = time.perf_counter()
            metrics = exp._train_one_epoch(model, train_loader, opt, ez_weight)
            if not np.isfinite(float(metrics["loss"])):
                raise RuntimeError("Nonfinite training loss")
            snapshot = score_only_snapshot(exp, model, val_loader)
            score_tmp = score_path.with_suffix(".pkl.tmp")
            with score_tmp.open("wb") as h:
                pickle.dump(snapshot, h, protocol=5)
            score_tmp.replace(score_path)
            row = {"variant": args.variant, "fold": args.fold, "epoch": epoch,
                   "train_loss": float(metrics["loss"]), "seconds": time.perf_counter() - start,
                   "score_sha256": sha(score_path), "validation_scores_only": True}
            ckpt_tmp = ckpt_path.with_suffix(".pt.tmp")
            torch.save({"lock_sha": LOCK_SHA, "variant": args.variant, "fold": args.fold, "epoch": epoch,
                        "model": model.state_dict(), "optimizer": opt.state_dict(),
                        "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std}, ckpt_tmp)
            ckpt_tmp.replace(ckpt_path)
            atomic_json(row_path, row)
        summary["epoch_rows"].append(row)
        print(f"[{args.variant}] fold={args.fold} epoch={epoch}/30 loss={row['train_loss']:.6f} sec={row['seconds']:.1f}", flush=True)
    atomic_json(folder / "summary.json", summary)


if __name__ == "__main__":
    run()
