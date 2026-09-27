"""Resume-safe thirty-epoch matched objective/capacity variants; no outer test."""

from __future__ import annotations

import argparse
import json
import math
import types

import numpy as np
import torch
from torch import nn

from common import (A1_RUNTIME, EXPERIMENT, LOCK_SHA256, RUNTIME, assert_no_outer_loader,
                    build_fold, core, ensure_source, epoch_grid, make_experiment, sha256, write_json)
from losses import loss_for_variant

VARIANTS = ("B10", "B01", "B11", "CAP1", "CAP2", "FP")


def capacity_encoder(variant: str, dropout: float) -> nn.Sequential:
    if variant == "CAP1":
        return nn.Sequential(nn.Linear(36, 64), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(64, 32), nn.LayerNorm(32))
    if variant == "CAP2":
        return nn.Sequential(nn.Linear(36, 128), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(128, 64), nn.GELU(), nn.Dropout(dropout),
                             nn.Linear(64, 32), nn.LayerNorm(32))
    raise ValueError(variant)


def build_model(exp, loader, initial: dict, fold: int, variant: str):
    core._set_random_seed(42 + fold)
    model = exp.runtime["model_cls"](exp.args).to(exp.device)
    exp._dry_initialize_lazy_layers(model, loader)
    if variant.startswith("CAP"):
        core._set_random_seed(42 + fold + 900)
        model.b0_encoder.feature_mlp = capacity_encoder(variant, float(exp.args.dropout)).to(exp.device)
        target = model.state_dict()
        common = {key: value for key, value in initial.items() if key in target and target[key].shape == value.shape
                  and not key.startswith("b0_encoder.feature_mlp")}
        target.update(common)
        model.load_state_dict(target, strict=True)
        if not all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in common.items()):
            raise RuntimeError("Common capacity parameters not initialized from source")
    else:
        model.load_state_dict(initial, strict=True)
        if not all(torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in initial.items()):
            raise RuntimeError("Source initialization mismatch")
    return model


def fit_ap(exp, model, loader, ez_weight):
    from development_metrics import patient_grid
    loss, _, records = exp._evaluate(model, loader, ez_weight, split_name="val")
    per_patient = [patient_grid(record)["fixed"]["patient_ez_auprc"] for record in records]
    return float(np.mean(per_patient)), float(loss)


def train_one(exp, fold, train_set, train_loader, val_loader, normalizer, variant, capacity_base):
    folder = RUNTIME / "private" / "training" / variant / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    if (folder / "complete.json").is_file():
        print(f"[KEEP] {variant} fold={fold}", flush=True)
        return
    initial_path = A1_RUNTIME / "initial" / f"fold_{fold}_initial.pt"
    initial = torch.load(initial_path, map_location="cpu", weights_only=True)
    initial_hash = sha256(initial_path)
    model = build_model(exp, train_loader, initial, fold, variant)
    exp._compute_loss = types.MethodType(loss_for_variant(variant, capacity_base), exp)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    diagnostic_loader = exp._make_loader(train_set, shuffle=False, batch_size=2) if variant.startswith("CAP") else None
    for epoch in range(1, 31):
        ckpt = folder / f"epoch_{epoch:02d}.pt"
        grid = folder / f"epoch_{epoch:02d}_validation_grid.json"
        if ckpt.is_file():
            saved = torch.load(ckpt, map_location=exp.device, weights_only=False)
            identity = (saved["variant"], saved["fold"], saved["epoch"], saved["lock_sha256"], saved["initial_sha256"], saved["capacity_base"])
            if identity != (variant, fold, epoch, LOCK_SHA256, initial_hash, capacity_base):
                raise RuntimeError("Resume provenance mismatch")
            if not np.array_equal(saved["normalizer_mean"], normalizer.mean) or not np.array_equal(saved["normalizer_std"], normalizer.std):
                raise RuntimeError("Resume normalizer mismatch")
            model.load_state_dict(saved["model_state_dict"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state_dict"])
            train_metrics = saved["train_metrics"]
        else:
            exp.current_epoch = epoch
            core._set_random_seed(42 * 100000 + fold * 1000 + epoch)
            train_metrics = exp._train_one_epoch(model, train_loader, optimizer, ez_weight)
            if not math.isfinite(float(train_metrics["loss"])):
                raise RuntimeError("Nonfinite training loss")
            temporary = ckpt.with_suffix(".pt.tmp")
            torch.save({"variant": variant, "fold": fold, "epoch": epoch, "lock_sha256": LOCK_SHA256,
                        "initial_sha256": initial_hash, "capacity_base": capacity_base,
                        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics, "normalizer_mean": normalizer.mean,
                        "normalizer_std": normalizer.std}, temporary)
            temporary.replace(ckpt)
        if not grid.is_file():
            val_loss, val_metrics, records = exp._evaluate(model, val_loader, ez_weight, split_name="val")
            write_json(grid, epoch_grid(records, epoch))
            if diagnostic_loader is not None:
                fit_auprc, fit_loss = fit_ap(exp, model, diagnostic_loader, ez_weight)
                write_json(folder / f"epoch_{epoch:02d}_trajectory.json",
                           {"epoch": epoch, "fit_auprc": fit_auprc, "fit_bce": fit_loss,
                            "validation_macro_f1_at_0_5": float(val_metrics["patient_macro_f1"]),
                            "validation_bce": float(val_loss),
                            "train_objective_loss": float(train_metrics["loss"])})
        print(f"[{variant}] fold={fold} epoch={epoch}/30 loss={train_metrics['loss']:.6f}", flush=True)
    write_json(folder / "complete.json", {"variant": variant, "fold": fold, "epochs": 30,
               "parameters": sum(parameter.numel() for parameter in model.parameters()),
               "capacity_base": capacity_base, "outer_test_accessed": False})
    print(f"[DONE] {variant} fold={fold}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", required=True, choices=VARIANTS)
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5))
    parser.add_argument("--capacity-base", choices=("original", "balanced"), default="original")
    options = parser.parse_args()
    ensure_source()
    if not json.loads((EXPERIMENT / "SOURCE_REPRODUCTION.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    if not json.loads((EXPERIMENT / "CLASS_BALANCED_LOSS_UNIT_TEST.json").read_text(encoding="utf-8"))["pass"]:
        raise RuntimeError("CLASS_BALANCED_LOSS_UNIT_TEST_FAILED")
    if options.variant.startswith("CAP"):
        gate = json.loads((EXPERIMENT / "audit_a_class_geometry" / "CLASS_GEOMETRY_GATE.json").read_text(encoding="utf-8"))
        if options.capacity_base != ("balanced" if gate["pass"] else "original"):
            raise RuntimeError("Capacity BCE base conflicts with frozen Audit A gate")
    if options.variant == "FP":
        gate = json.loads((EXPERIMENT / "audit_b_top_heavy" / "HARD_NEGATIVE_GATE.json").read_text(encoding="utf-8"))
        if not gate["first_positive_triggered"] or options.capacity_base != gate["first_positive_base"]:
            raise RuntimeError("First-positive diagnostic not predeclared-triggered")
    exp = make_experiment()
    for split in exp.outer_splits:
        fold, train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
        assert_no_outer_loader(test_loader)
        if options.fold and fold != options.fold:
            continue
        train_one(exp, fold, train_set, train_loader, val_loader, normalizer, options.variant, options.capacity_base)


if __name__ == "__main__":
    main()
