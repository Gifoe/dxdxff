"""Reproduce 150 A1 validation grids, VLOO-select target epochs, extract exact R3/R4 privately."""
from __future__ import annotations

import argparse
import json
import math
import pickle

import numpy as np
import torch

from common import (EXPECTED_FOLDS, ROOT, RUNTIME, build_fold, checkpoint, core, epoch_grid,
                    experiment, finalize_fold, preflight, source_grid, write_csv, write_json)


def grid_error(actual, reference):
    if actual["epoch"] != reference["epoch"] or [r["subject_id"] for r in actual["patients"]] != [r["subject_id"] for r in reference["patients"]]:
        raise RuntimeError("A1 grid epoch/patient identity changed")
    error = 0.0
    for a, b in zip(actual["patients"], reference["patients"], strict=True):
        if a["n_channels"] != b["n_channels"]:
            raise RuntimeError("A1 grid channel count changed")
        for group in ("grid", "fixed"):
            if set(a[group]) != set(b[group]):
                raise RuntimeError("A1 grid metric schema changed")
            for metric in a[group]:
                error = max(error, float(np.max(np.abs(np.asarray(a[group][metric]) - np.asarray(b[group][metric])))))
    return error


def capture(model, loader, device):
    hooks = {}

    def r3_hook(_module, inputs):
        hooks["r3"] = inputs[0].detach()

    def r4_hook(_module, inputs):
        hooks["r4"] = inputs[0].detach()

    h3 = model.channel_classifier.channel_attn.register_forward_pre_hook(r3_hook)
    h4 = model.channel_classifier.classifier.register_forward_pre_hook(r4_hook)
    result = {}
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                hooks.clear()
                out = model(core._move_tensors_to_device(batch, device))
                if set(hooks) != {"r3", "r4"}:
                    raise RuntimeError("Exact R3/R4 hook failed")
                mask = batch["channel_mask"].to(device)
                r3, r4 = hooks["r3"], hooks["r4"]
                if r3.ndim != 3 or r3.shape != r4.shape or r3.shape[:2] != mask.shape:
                    raise RuntimeError("Exact representation shape mismatch")
                replay = model.channel_classifier.classifier(r4).squeeze(-1)
                if not torch.allclose(replay[mask], out["logits"][mask], atol=1e-6, rtol=0):
                    raise RuntimeError("R4 original-classifier replay failed")
                for i, subject in enumerate(batch["subject_id"]):
                    valid = batch["channel_mask"][i].numpy().astype(bool)
                    x3 = r3[i][mask[i]].cpu().numpy().astype(np.float32)
                    x4 = r4[i][mask[i]].cpu().numpy().astype(np.float32)
                    y = batch["labels_ez"][i].numpy()[valid].astype(np.int8)
                    if set(np.unique(y)) != {0, 1}:
                        raise RuntimeError("Patient without both classes")
                    result[str(subject)] = {"R3": x3, "R4": x4, "y": y,
                                            "source_ez": (-out["logits"][i][mask[i]]).cpu().numpy().astype(np.float32)}
    finally:
        h3.remove()
        h4.remove()
    return result


def run_fold(exp, split):
    fold, train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
    if test_loader is not None:
        raise RuntimeError("Outer loader constructed")
    fit_ids, val_ids = set(split["fit_subjects"]), set(split["validation_subjects"])
    if fit_ids & val_ids or len(fit_ids) != (51, 51, 50, 52, 51)[fold - 1] or len(val_ids) != 13:
        raise RuntimeError("FIT/validation membership mismatch")
    folder = RUNTIME / "private" / f"fold_{fold}"
    folder.mkdir(parents=True, exist_ok=True)
    fit_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
    model = exp.runtime["model_cls"](exp.args).to(exp.device)
    exp._dry_initialize_lazy_layers(model, train_loader)
    errors = []
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    # Recheck every historical validation grid rather than trusting a reference summary alone.
    for epoch in range(1, 31):
        state = torch.load(checkpoint(fold, epoch), map_location=exp.device, weights_only=False)
        if (state["variant"], state["fold"], state["epoch"]) != ("A1", fold, epoch):
            raise RuntimeError("A1 checkpoint identity changed")
        if not np.array_equal(state["normalizer_mean"], normalizer.mean) or not np.array_equal(state["normalizer_std"], normalizer.std):
            raise RuntimeError("A1 FIT normalizer changed")
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        _, _, records = exp._evaluate(model, val_loader, ez_weight, split_name="val")
        error = grid_error(epoch_grid(records, epoch), source_grid(fold, epoch))
        if error > 1e-6:
            raise RuntimeError(f"SOURCE_A1_REPRODUCTION_FAILED fold={fold} epoch={epoch} error={error}")
        errors.append(error)
    public, _ = finalize_fold([source_grid(fold, e) for e in range(1, 31)], "A1", fold,
                              folder / "A1_VLOO_PRIVATE.csv")
    if not math.isclose(public["patient_macro_f1"], EXPECTED_FOLDS[fold - 1], abs_tol=1e-6):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED VLOO")
    import csv
    with (folder / "A1_VLOO_PRIVATE.csv").open(encoding="utf-8", newline="") as f:
        selected = list(csv.DictReader(f))
    if set(r["subject_id"] for r in selected) != val_ids:
        raise RuntimeError("VLOO selected patients changed")
    dim = None
    for epoch in sorted({int(r["selected_epoch"]) for r in selected}):
        target = folder / f"epoch_{epoch:02d}_representations.pkl"
        if target.exists():
            with target.open("rb") as f:
                payload = pickle.load(f)
            if (payload["fold"], payload["epoch"]) != (fold, epoch) or set(payload["fit"]) != fit_ids or set(payload["val"]) != val_ids:
                raise RuntimeError("Private representation cache invalid")
            dim = payload["dim"]
            continue
        state = torch.load(checkpoint(fold, epoch), map_location=exp.device, weights_only=False)
        model.load_state_dict(state["model_state_dict"], strict=True)
        model.eval()
        fit, val = capture(model, fit_loader, exp.device), capture(model, val_loader, exp.device)
        if set(fit) != fit_ids or set(val) != val_ids:
            raise RuntimeError("Representation membership changed")
        dim = int(next(iter(fit.values()))["R4"].shape[1])
        if dim != int(model.channel_classifier.classifier[0].in_features) or dim != 64:
            raise RuntimeError("R4 actual D differs from locked expected D")
        if any(row["R3"].shape[1] != dim for row in list(fit.values()) + list(val.values())):
            raise RuntimeError("R3 dimension differs from R4")
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as f:
            pickle.dump({"fold": fold, "epoch": epoch, "dim": dim, "fit": fit, "val": val}, f, protocol=5)
        tmp.replace(target)
        print(f"[EXTRACT] fold={fold} epoch={epoch} R3/R4 complete", flush=True)
    write_json(folder / "extract_status.json", {"fold": fold, "source_replay_count": 30,
               "max_grid_error": max(errors), "selected_epoch_count": len({int(r["selected_epoch"]) for r in selected}),
               "target_cells": 13, "r3_r4_dim": dim, "outer_loader_constructed": False})
    return public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    opt = parser.parse_args()
    preflight()
    exp = experiment()
    results = []
    for split in exp.outer_splits:
        if opt.fold is None or int(split["fold_idx"]) == opt.fold:
            results.append(run_fold(exp, split))
    if opt.fold is None:
        if len(results) != 5 or not math.isclose(np.mean([r["patient_macro_f1"] for r in results]),
                                                  0.6259962097139906, abs_tol=1e-6):
            raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED mean")
        write_json(ROOT / "SOURCE_REPRODUCTION.json", {"pass": True, "terminal": "SOURCE_A1_REPRODUCED",
                   "checkpoints": 150, "max_grid_error": max(json.loads((RUNTIME / "private" / f"fold_{f}" / "extract_status.json").read_text(encoding="utf-8"))["max_grid_error"] for f in range(1, 6)),
                   "fold_macro_f1": [r["patient_macro_f1"] for r in results],
                   "mean_macro_f1": float(np.mean([r["patient_macro_f1"] for r in results])),
                   "r3_hook": "channel_attn forward pre-hook", "r4_hook": "original classifier forward pre-hook",
                   "r4_original_classifier_replay": True, "outer_prediction_or_metric": False,
                   "legacy_monolithic_cache_materializes_all_80_labels": True})


if __name__ == "__main__":
    main()
