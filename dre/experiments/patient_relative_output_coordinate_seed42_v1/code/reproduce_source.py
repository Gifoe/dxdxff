"""Rebuild frozen A1 validation logits; reproduce source VLOO before C1/C2.

No training and no outer-test examples, predictions or result files are used.
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "code"))
from development_metrics import epoch_grid, finalize_fold  # noqa: E402
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import SOURCE_ROOT, assert_sources, build_fold, install_interleaved_hlv_view, make_args, sha256  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402


EXPECTED_LOCK_SHA256 = "879a585e7a6ae8df3b69faaa6efedc29a02711f5c5a07adcd64fed0b305b5805"
SOURCE_RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))
RUNTIME = Path(os.environ.get("COORD_RUNTIME", ""))


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def collect_validation_logits(exp, model, loader) -> tuple[list[dict], float]:
    model.eval()
    patients = []
    max_probability_error = 0.0
    with torch.no_grad():
        for batch in loader:
            device_batch = core._move_tensors_to_device(batch, exp.device)
            outputs = model(device_batch)
            # The frozen A1 classifier emits EZ-positive logits. The locked
            # coordinate experiment uses NEZ-positive logits, so negate them.
            logits = (-outputs["logits"]).detach().cpu().numpy()
            score_nez = outputs["score_nez"].detach().cpu().numpy()
            score_ez = outputs["score_ez"].detach().cpu().numpy()
            labels_nez = batch["labels_nez"].numpy()
            labels_ez = batch["labels_ez"].numpy()
            mask = batch["channel_mask"].numpy()
            for index, subject_id in enumerate(batch["subject_id"]):
                valid = mask[index].astype(bool)
                a = logits[index][valid].astype(np.float32)
                p = score_nez[index][valid].astype(np.float32)
                q = score_ez[index][valid].astype(np.float32)
                error = np.max(np.abs(torch.sigmoid(torch.from_numpy(a)).numpy() - p))
                max_probability_error = max(max_probability_error, float(error))
                if not np.isfinite(a).all() or not np.isfinite(p).all() or not np.isfinite(q).all():
                    raise RuntimeError("Nonfinite A1 validation logits or scores")
                patients.append({
                    "subject_id": str(subject_id),
                    "logits_nez": a.astype(float).tolist(),
                    "score_nez_core": p.astype(float).tolist(),
                    "score_ez_core": q.astype(float).tolist(),
                    "labels_nez": labels_nez[index][valid].astype(int).tolist(),
                    "labels_ez": labels_ez[index][valid].astype(int).tolist(),
                })
    if len(patients) != 13 or len({row["subject_id"] for row in patients}) != 13:
        raise RuntimeError("Expected 13 unique validation patients")
    return sorted(patients, key=lambda row: row["subject_id"]), max_probability_error


def to_c0_records(patients: list[dict]) -> list[dict]:
    records = []
    for row in patients:
        labels_nez = np.asarray(row["labels_nez"], dtype=np.float32)
        labels_ez = np.asarray(row["labels_ez"], dtype=np.float32)
        p = np.asarray(row["score_nez_core"], dtype=np.float32)
        q = np.asarray(row["score_ez_core"], dtype=np.float32)
        records.append({"subject_id": row["subject_id"], "labels": labels_nez,
                        "labels_nez": labels_nez, "labels_ez": labels_ez,
                        "score_nez": p, "score_ez": q,
                        "channel_mask": np.ones(len(p), dtype=bool)})
    return records


def compare_epoch_grid(rebuilt: dict, prior: dict) -> float:
    if [row["subject_id"] for row in rebuilt["patients"]] != [row["subject_id"] for row in prior["patients"]]:
        raise RuntimeError("Rebuilt A1 validation patient order changed")
    worst = 0.0
    for new, old in zip(rebuilt["patients"], prior["patients"], strict=True):
        if new["n_channels"] != old["n_channels"]:
            raise RuntimeError("A1 validation channel count changed")
        for category in ("grid", "fixed"):
            if set(new[category]) != set(old[category]):
                raise RuntimeError("A1 validation metric fields changed")
            for metric in new[category]:
                worst = max(worst, float(np.max(np.abs(np.asarray(new[category][metric]) -
                                                      np.asarray(old[category][metric])))))
    return worst


def main() -> None:
    if not os.environ.get("A1_A2_RUNTIME") or not SOURCE_RUNTIME.is_absolute() or not os.environ.get("COORD_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("Set absolute A1_A2_RUNTIME and COORD_RUNTIME")
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Output-coordinate protocol lock changed")
    assert_sources()
    lock = json.loads((EXPERIMENT / "PROTOCOL_LOCK.json").read_text(encoding="utf-8"))
    if sha256(HERE.parents[2] / "a1_a2_patient_equal_objective_seed42_v1" / "PROTOCOL_LOCK.json") != lock["source_protocol_sha256"]:
        raise RuntimeError("Source A1 protocol changed")
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME)
    exp = core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("Source cohort changed")
    folds = []
    overall_probability_error = overall_grid_error = 0.0
    for split in exp.outer_splits:
        fold, _, train_loader, val_loader, _, normalizer = build_fold(exp, split, "validation")
        model = exp.runtime["model_cls"](args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        epochs = []
        for epoch in range(1, 31):
            checkpoint_path = SOURCE_RUNTIME / "A1" / f"fold_{fold}" / f"epoch_{epoch:02d}.pt"
            checkpoint = torch.load(checkpoint_path, map_location=exp.device, weights_only=False)
            if checkpoint["variant"] != "A1" or int(checkpoint["fold"]) != fold or int(checkpoint["epoch"]) != epoch:
                raise RuntimeError("A1 checkpoint identity mismatch")
            if not np.array_equal(normalizer.mean, checkpoint["normalizer_mean"]) or not np.array_equal(normalizer.std, checkpoint["normalizer_std"]):
                raise RuntimeError("A1 fit normalizer changed")
            model.load_state_dict(checkpoint["model_state_dict"], strict=True)
            patients, probability_error = collect_validation_logits(exp, model, val_loader)
            if probability_error > 1e-7:
                raise RuntimeError("sigmoid(raw logits) does not reproduce frozen model score_nez")
            overall_probability_error = max(overall_probability_error, probability_error)
            payload = {"epoch": epoch, "patients": patients}
            write_json(RUNTIME / "validation_logits_private" / f"fold_{fold}" / f"epoch_{epoch:02d}.json", payload)
            grid = epoch_grid(to_c0_records(patients), epoch)
            prior = json.loads((SOURCE_RUNTIME / "A1" / f"fold_{fold}" /
                                f"epoch_{epoch:02d}_validation_grid.json").read_text(encoding="utf-8"))
            grid_error = compare_epoch_grid(grid, prior)
            if grid_error > 1e-8:
                raise RuntimeError("Recomputed A1 validation grid disagrees with stored original")
            overall_grid_error = max(overall_grid_error, grid_error)
            epochs.append(grid)
        aggregate, _ = finalize_fold(epochs, "C0_RAW", fold,
                                     RUNTIME / "private" / f"fold_{fold}_C0_VLOO_PATIENT.csv")
        reproduced = aggregate["patient_macro_f1"]
        reference = lock["source_vloo_macro_f1_by_fold"][fold - 1]
        folds.append({"fold": fold, "reference": reference, "reproduced": reproduced,
                      "absolute_error": abs(reproduced - reference)})
        print(f"[SOURCE] fold {fold}: reproduced VLOO Macro-F1={reproduced:.10f}", flush=True)
    mean = float(np.mean([row["reproduced"] for row in folds]))
    passed = all(row["absolute_error"] <= lock["source_reproduction_tolerance"] for row in folds) and \
        abs(mean - lock["source_vloo_macro_f1_mean"]) <= lock["source_reproduction_tolerance"]
    result = {"pass": passed, "terminal": "SOURCE_REPRODUCED" if passed else "SOURCE_A1_VLOO_REPRODUCTION_FAILED",
              "frozen_A1_checkpoints_evaluated": 150, "validation_patients_per_fold": 13,
              "raw_logit_vs_core_probability_max_abs_error": overall_probability_error,
              "recomputed_grid_vs_stored_max_abs_error": overall_grid_error,
              "folds": folds, "reference_mean": lock["source_vloo_macro_f1_mean"],
              "reproduced_mean": mean, "mean_absolute_error": abs(mean - lock["source_vloo_macro_f1_mean"]),
              "outer_test_accessed": False}
    write_json(EXPERIMENT / "SOURCE_REPRODUCTION.json", result)
    print(json.dumps(result, indent=2), flush=True)
    if not passed:
        raise RuntimeError("SOURCE_A1_VLOO_REPRODUCTION_FAILED")


if __name__ == "__main__":
    main()
