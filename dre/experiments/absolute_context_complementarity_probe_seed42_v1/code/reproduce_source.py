"""Reproduce all frozen A1 validation grids and VLOO before any probe training."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, assert_sources, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import epoch_grid, finalize_fold  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402

LOCK_SHA256 = "76a9a94cbf364878088ed89370e062e6fe92687adafcd4f95b923e67ce1a36a3"
RUNTIME = Path(os.environ.get("ABS_CONTEXT_RUNTIME", ""))
A1_RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def validation_grid(exp, model, loader, epoch: int) -> dict:
    records = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            outputs = model(core._move_tensors_to_device(batch, exp.device))
            p = outputs["score_nez"].detach().cpu().numpy()
            q = outputs["score_ez"].detach().cpu().numpy()
            mask = batch["channel_mask"].numpy()
            for i, subject_id in enumerate(batch["subject_id"]):
                valid = mask[i].astype(bool)
                records.append({
                    "subject_id": str(subject_id),
                    "labels": batch["labels_nez"][i].numpy()[valid],
                    "labels_nez": batch["labels_nez"][i].numpy()[valid],
                    "labels_ez": batch["labels_ez"][i].numpy()[valid],
                    "score_nez": p[i][valid], "score_ez": q[i][valid],
                    "channel_mask": np.ones(int(valid.sum()), dtype=bool),
                })
    return epoch_grid(records, epoch)


def compare_grids(current: dict, original: dict) -> float:
    if [r["subject_id"] for r in current["patients"]] != [r["subject_id"] for r in original["patients"]]:
        raise RuntimeError("A1 validation patient order changed")
    maximum = 0.0
    for new, old in zip(current["patients"], original["patients"], strict=True):
        if new["n_channels"] != old["n_channels"]:
            raise RuntimeError("A1 validation channel count changed")
        for category in ("grid", "fixed"):
            if set(new[category]) != set(old[category]):
                raise RuntimeError("A1 validation metric fields changed")
            for metric in new[category]:
                maximum = max(maximum, float(np.max(np.abs(np.asarray(new[category][metric]) -
                                                       np.asarray(old[category][metric])))))
    return maximum


def main() -> None:
    if not os.environ.get("ABS_CONTEXT_RUNTIME") or not RUNTIME.is_absolute() or \
            not os.environ.get("A1_A2_RUNTIME") or not A1_RUNTIME.is_absolute():
        raise RuntimeError("Set absolute ABS_CONTEXT_RUNTIME and A1_A2_RUNTIME")
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != LOCK_SHA256:
        raise RuntimeError("Probe protocol lock changed")
    assert_sources()
    lock = json.loads((EXPERIMENT / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if sha256(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "PROTOCOL_LOCK.json") != lock["source_protocol_sha256"]:
        raise RuntimeError("A1 source lock changed")
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    exp = core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5 or len(exp.run_records) != 256 or \
            sum(len(meta["canonical_channels"]) for meta in exp.patient_index.values()) != 7635:
        raise RuntimeError("Frozen source cohort changed")
    folds, max_grid_error = [], 0.0
    for split in exp.outer_splits:
        fold, _, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
        if test_loader is not None:
            raise RuntimeError("Outer-test loader must not be built")
        model = exp.runtime["model_cls"](args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        epochs = []
        for epoch in range(1, 31):
            checkpoint = torch.load(A1_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}.pt",
                                    map_location=exp.device, weights_only=False)
            if checkpoint["variant"] != "A1" or int(checkpoint["fold"]) != fold or int(checkpoint["epoch"]) != epoch:
                raise RuntimeError("A1 checkpoint identity changed")
            if not np.array_equal(normalizer.mean, checkpoint["normalizer_mean"]) or \
                    not np.array_equal(normalizer.std, checkpoint["normalizer_std"]):
                raise RuntimeError("A1 normalization changed")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            grid = validation_grid(exp, model, val_loader, epoch)
            original = json.loads((A1_RUNTIME / "A1" / f"fold_{fold}" /
                                   f"epoch_{epoch:02d}_validation_grid.json").read_text(encoding="utf-8"))
            error = compare_grids(grid, original)
            if error > 1e-8:
                raise RuntimeError(f"A1 validation grid changed, max_abs={error}")
            max_grid_error = max(max_grid_error, error)
            epochs.append(grid)
        aggregate, _ = finalize_fold(epochs, "P0", fold,
                                     RUNTIME / "private" / f"fold_{fold}_P0_VLOO_PATIENT.csv")
        observed = aggregate["patient_macro_f1"]
        reference = lock["source_vloo_macro_f1_by_fold"][fold - 1]
        folds.append({"fold": fold, "observed": observed, "reference": reference,
                      "absolute_error": abs(observed - reference)})
        print(f"[SOURCE] fold={fold} VLOO={observed:.10f}", flush=True)
    mean = float(np.mean([row["observed"] for row in folds]))
    passed = all(row["absolute_error"] <= lock["source_tolerance"] for row in folds) and \
        abs(mean - lock["source_vloo_macro_f1_mean"]) <= lock["source_tolerance"]
    result = {"pass": passed, "terminal": "SOURCE_REPRODUCED" if passed else "SOURCE_A1_REPRODUCTION_FAILED",
              "frozen_checkpoints_evaluated": 150, "validation_patients_per_fold": 13,
              "max_grid_error_vs_original": max_grid_error, "folds": folds, "observed_mean": mean,
              "reference_mean": lock["source_vloo_macro_f1_mean"], "outer_test_accessed": False}
    write_json(EXPERIMENT / "SOURCE_REPRODUCTION.json", result)
    if not passed:
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")


if __name__ == "__main__":
    main()
