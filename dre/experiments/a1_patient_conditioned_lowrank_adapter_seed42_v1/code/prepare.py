"""Replay 150 A1 grids and cache exact frozen classifier-input R4 arrays privately."""
from __future__ import annotations

import argparse
import json
import math
import pickle

import numpy as np
import torch

from common import (EXPECTED, ROOT, RUNTIME, build_fold, checkpoint, core, epoch_grid,
                    experiment, finalize_fold, preflight, source_grid, write_csv, write_json)


def grid_error(actual, reference):
    if actual["epoch"] != reference["epoch"] or [r["subject_id"] for r in actual["patients"]] != [r["subject_id"] for r in reference["patients"]]:
        raise RuntimeError("A1 grid patient or epoch changed")
    maximum = 0.0
    for a, b in zip(actual["patients"], reference["patients"], strict=True):
        if a["n_channels"] != b["n_channels"]:
            raise RuntimeError("A1 channel count changed")
        for group in ("grid", "fixed"):
            if set(a[group]) != set(b[group]):
                raise RuntimeError("A1 metric schema changed")
            for metric in a[group]:
                maximum = max(maximum, float(np.max(np.abs(np.asarray(a[group][metric]) - np.asarray(b[group][metric])))))
    return maximum


def capture_r4(model, loader, device):
    captured = {}
    def hook(_module, inputs):
        captured["r4"] = inputs[0].detach()
    handle = model.channel_classifier.classifier.register_forward_pre_hook(hook)
    rows = {}
    model.eval()
    try:
        with torch.no_grad():
            for batch in loader:
                captured.clear()
                out = model(core._move_tensors_to_device(batch, device))
                if "r4" not in captured:
                    raise RuntimeError("Classifier-input R4 not captured")
                mask = batch["channel_mask"].to(device)
                h = captured["r4"]
                if h.ndim != 3 or h.shape[:2] != mask.shape:
                    raise RuntimeError("R4 interface is not [B,C,D]")
                replay = model.channel_classifier.classifier(h).squeeze(-1)
                if not torch.allclose(replay[mask], out["logits"][mask], atol=1e-6, rtol=0):
                    raise RuntimeError("R4 classifier replay differs from source")
                for i, subject in enumerate(batch["subject_id"]):
                    valid = batch["channel_mask"][i].numpy().astype(bool)
                    rows[str(subject)] = {"h": h[i][mask[i]].cpu().numpy().astype(np.float32),
                                          "y_ez": batch["labels_ez"][i].numpy()[valid].astype(np.int8),
                                          "source_nez_logit": out["logits"][i][mask[i]].cpu().numpy().astype(np.float32)}
                    if set(np.unique(rows[str(subject)]["y_ez"])) != {0, 1}:
                        raise RuntimeError("FIT/validation patient lacks a class")
    finally:
        handle.remove()
    return rows


def fold_run(exp, split):
    fold, train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
    if test_loader is not None:
        raise RuntimeError("Outer loader constructed")
    fit_ids, val_ids = set(split["fit_subjects"]), set(split["validation_subjects"])
    if fit_ids & val_ids or len(val_ids) != 13:
        raise RuntimeError("FIT/validation role overlap or membership changed")
    outdir = RUNTIME / "private" / f"fold_{fold}"
    outdir.mkdir(parents=True, exist_ok=True)
    fit_loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
    model = exp.runtime["model_cls"](exp.args).to(exp.device)
    exp._dry_initialize_lazy_layers(model, train_loader)
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    errors = []
    for epoch in range(1, 31):
        target = outdir / f"source_epoch_{epoch:02d}.pkl"
        if target.exists():
            with target.open("rb") as stream:
                stored = pickle.load(stream)
            if stored["fold"] != fold or stored["epoch"] != epoch or stored["error"] > 1e-6:
                raise RuntimeError("Bad source replay cache")
            errors.append(stored["error"])
            dim = int(stored["dim"])
            continue
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
        fit, val = capture_r4(model, fit_loader, exp.device), capture_r4(model, val_loader, exp.device)
        if set(fit) != fit_ids or set(val) != val_ids:
            raise RuntimeError("Frozen R4 extraction membership changed")
        dim = int(next(iter(fit.values()))["h"].shape[1])
        if dim != int(model.channel_classifier.classifier[0].in_features):
            raise RuntimeError("R4 D differs from source classifier input D")
        payload = {"fold": fold, "epoch": epoch, "error": error, "dim": dim, "fit": fit, "val": val}
        tmp = target.with_suffix(".tmp")
        with tmp.open("wb") as stream:
            pickle.dump(payload, stream, protocol=5)
        tmp.replace(target)
        errors.append(error)
        print(f"[PREPARE] fold={fold} epoch={epoch} grid_error={error:.1g} D={dim}", flush=True)
    grids = [source_grid(fold, epoch) for epoch in range(1, 31)]
    public, _ = finalize_fold(grids, "P0", fold, outdir / "P0_VLOO_PRIVATE.csv")
    if not math.isclose(public["patient_macro_f1"], EXPECTED[fold - 1], abs_tol=1e-6):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED selected VLOO")
    write_json(outdir / "prepare_status.json", {"fold": fold, "checkpoints": 30, "max_error": max(errors),
                                               "r4_dim": dim, "classifier_replay_pass": True,
                                               "outer_loader_constructed": False})
    return public


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, choices=range(1, 6))
    options = parser.parse_args()
    preflight()
    exp = experiment()
    results = []
    for split in exp.outer_splits:
        if options.fold is None or int(split["fold_idx"]) == options.fold:
            results.append(fold_run(exp, split))
    if options.fold is None:
        if len(results) != 5 or not math.isclose(np.mean([r["patient_macro_f1"] for r in results]),
                                                  0.6259962097139906, abs_tol=1e-6):
            raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
        write_csv(ROOT / "p0_source" / "P0_VLOO_BY_FOLD.csv", results)
        write_json(ROOT / "SOURCE_REPRODUCTION.json", {"pass": True, "terminal": "SOURCE_A1_REPRODUCED",
                   "checkpoints": 150, "max_grid_error": max(json.loads((RUNTIME / "private" / f"fold_{f}" / "prepare_status.json").read_text())["max_error"] for f in range(1, 6)),
                   "mean_macro_f1": float(np.mean([r["patient_macro_f1"] for r in results])),
                   "r4_dim": 64, "r4_obtained_from_exact_classifier_input_hook": True,
                   "ordinary_a1_forward_has_contextual_channel_embedding_key": False,
                   "outer_predictions_or_metrics": False,
                   "legacy_monolithic_loader_materializes_all_80_labels": True})


if __name__ == "__main__":
    main()
