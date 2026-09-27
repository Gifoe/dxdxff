"""Re-evaluate every frozen A1 checkpoint on validation; never build outer test."""

from __future__ import annotations

import json
import math

import numpy as np
import torch

from common import (A1_RUNTIME, EXPERIMENT, RUNTIME, assert_no_outer_loader, build_fold, core,
                    ensure_source, epoch_grid, finalize_fold, make_experiment, source_checkpoint,
                    source_grid, write_json)

EXPECTED = (0.6549997115717437, 0.6410958089865288, 0.6101185971670673,
            0.6329791804569156, 0.5907877503876969)


def grid_error(actual: dict, reference: dict) -> float:
    if actual["epoch"] != reference["epoch"] or [r["subject_id"] for r in actual["patients"]] != [r["subject_id"] for r in reference["patients"]]:
        raise RuntimeError("Source validation grid membership or epoch changed")
    maximum = 0.0
    for a, b in zip(actual["patients"], reference["patients"], strict=True):
        if a["n_channels"] != b["n_channels"]:
            raise RuntimeError("Source validation channel count changed")
        for group in ("grid", "fixed"):
            if set(a[group]) != set(b[group]):
                raise RuntimeError("Source metric schema changed")
            for metric in a[group]:
                maximum = max(maximum, float(np.max(np.abs(np.asarray(a[group][metric]) - np.asarray(b[group][metric])))))
    return maximum


def main() -> None:
    ensure_source()
    exp = make_experiment()
    folds, errors = [], []
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    for split in exp.outer_splits:
        fold, _train_set, train_loader, val_loader, test_loader, normalizer = build_fold(exp, split, "validation")
        assert_no_outer_loader(test_loader)
        if len(val_loader.dataset) != 13:
            raise RuntimeError("A1 VLOO requires 13 validation patients")
        model = exp.runtime["model_cls"](exp.args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        for epoch in range(1, 31):
            checkpoint = torch.load(source_checkpoint(fold, epoch), map_location=exp.device, weights_only=False)
            if (checkpoint["variant"], checkpoint["fold"], checkpoint["epoch"]) != ("A1", fold, epoch):
                raise RuntimeError("A1 checkpoint identity changed")
            if not np.array_equal(checkpoint["normalizer_mean"], normalizer.mean) or not np.array_equal(checkpoint["normalizer_std"], normalizer.std):
                raise RuntimeError("A1 normalization changed")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            model.eval()
            _, _, records = exp._evaluate(model, val_loader, ez_weight, split_name="val")
            error = grid_error(epoch_grid(records, epoch), source_grid(fold, epoch))
            errors.append(error)
            if error > 1e-8:
                raise RuntimeError(f"A1 replay grid differed: fold={fold}, epoch={epoch}, error={error}")
        grids = [source_grid(fold, epoch) for epoch in range(1, 31)]
        public, _ = finalize_fold(grids, "A1", fold, RUNTIME / "private" / f"fold_{fold}_A1_SF1_PATIENT.csv")
        if not math.isclose(public["patient_macro_f1"], EXPECTED[fold - 1], abs_tol=1e-8):
            raise RuntimeError(f"A1 VLOO changed in fold {fold}")
        folds.append({"fold": fold, "observed_macro_f1": public["patient_macro_f1"], "expected_macro_f1": EXPECTED[fold - 1]})
        print(f"[SOURCE] fold={fold} exact=True", flush=True)
    mean = float(np.mean([row["observed_macro_f1"] for row in folds]))
    if not math.isclose(mean, 0.6259962097139906, abs_tol=1e-8):
        raise RuntimeError("SOURCE_A1_REPRODUCTION_FAILED")
    write_json(EXPERIMENT / "SOURCE_REPRODUCTION.json", {"pass": True, "terminal": "SOURCE_A1_REPRODUCED",
               "checkpoints": 150, "patients_per_fold": 13, "max_validation_grid_error": max(errors),
               "folds": folds, "mean_macro_f1": mean, "outer_test_accessed": False})


if __name__ == "__main__":
    main()
