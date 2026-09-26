"""One-shot outer evaluation; refuses to run unless the locked gate passed."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from run_matched import (
    EXPERIMENT, RUNTIME, SOURCE_ROOT, GateTracker, assert_sources, build_fold,
    install_interleaved_hlv_view, make_args, numeric_summary,
)


PUBLIC = EXPERIMENT / "outer"
FIELDS = ("patient_macro_f1", "patient_macro_ez_f1", "patient_macro_balanced_accuracy", "patient_macro_auprc_ez")


def csv_write(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def patient_rows(records, fold, variant):
    rows = []
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels = np.asarray(record["labels_ez"], dtype=int)[valid]
        scores = np.asarray(record["score_ez"], dtype=float)[valid]
        if len(np.unique(labels)) != 2:
            raise RuntimeError("Outer patient lacks both classes for EZ-AUPRC")
        rows.append({
            "subject_id": str(record["subject_id"]), "fold": fold, "variant": variant,
            "patient_macro_f1": float(record["patient_macro_f1"]),
            "patient_macro_ez_f1": float(record["patient_ez_f1"]),
            "patient_macro_balanced_accuracy": float(record["patient_balanced_accuracy"]),
            "patient_macro_auprc_ez": float(average_precision_score(labels, scores)),
        })
    return rows


def evaluate_variant(variant: str):
    import exp_ez_hybrid as core

    args = make_args(variant, RUNTIME / variant)
    exp = core.Exp_EZHybridLocalization(args)
    fold_rows, patients, diagnostics = [], [], []
    for split in exp.outer_splits:
        fold = int(split["fold_idx"])
        checkpoint_path = RUNTIME / variant / f"fold_{fold}" / "selected.pt"
        if not checkpoint_path.exists():
            raise RuntimeError(f"Missing validation-selected checkpoint: {checkpoint_path}")
        fold, _, train_loader, _, test_loader, normalizer = build_fold(exp, split, "outer")
        if test_loader is None:
            raise RuntimeError("Outer loader was not constructed")
        model = exp.runtime["model_cls"](args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        checkpoint = torch.load(checkpoint_path, map_location=exp.device, weights_only=False)
        if checkpoint["fold"] != fold or checkpoint["variant"] != variant:
            raise RuntimeError("Checkpoint fold/variant mismatch")
        for attr, value in (("mean", checkpoint["normalizer_mean"]), ("std", checkpoint["normalizer_std"])):
            if not np.array_equal(getattr(normalizer, attr), value):
                raise RuntimeError("Fit-only normalizer changed since validation training")
        if not np.array_equal(normalizer.physics.mean, checkpoint["normalizer_physics_mean"]):
            raise RuntimeError("Physics normalizer changed since validation training")
        model.load_state_dict(checkpoint["model_state_dict"])
        tracker = GateTracker(model) if variant == "R1" else None
        ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
        _, _, raw_records = exp._evaluate(model, test_loader, ez_weight, split_name="test")
        threshold = float(checkpoint["threshold"])
        summary, records = core._summarize_prediction_records(raw_records, classification_threshold=threshold)
        row = numeric_summary(summary, records, fold, variant, int(checkpoint["selected_epoch"]), threshold)
        if tracker is not None:
            diag = {"fold": fold, **tracker.summary()}
            diagnostics.append(diag)
            tracker.close()
        fold_rows.append(row)
        patients.extend(patient_rows(records, fold, variant))
        print(f"[{variant}] outer fold {fold} Macro-F1={row['patient_macro_f1']:.6f}", flush=True)
    if len(patients) != 80 or len({row["subject_id"] for row in patients}) != 80:
        raise RuntimeError("Outer evaluation did not cover 80 patients exactly once")
    private = RUNTIME / variant / "outer_patient_metrics.json"
    private.write_text(json.dumps(patients), encoding="utf-8")
    csv_write(PUBLIC / f"{variant}_OUTER_RESULTS.csv", fold_rows)
    if variant == "R1":
        csv_write(PUBLIC / "HLV_GATE_DIAGNOSTICS.csv", diagnostics)
    return fold_rows, patients


def main():
    gate_path = EXPERIMENT / "validation" / "VALIDATION_GATE.json"
    if not gate_path.is_file() or json.loads(gate_path.read_text(encoding="utf-8")).get("pass") is not True:
        raise RuntimeError("R1 validation gate did not pass; outer evaluation forbidden")
    if (PUBLIC / "OUTER_COMPARISON.csv").exists():
        raise RuntimeError("Outer evaluation already completed; refusing a second run")
    assert_sources()
    sys.path.insert(0, str(SOURCE_ROOT))
    install_interleaved_hlv_view()
    r0_folds, r0_patients = evaluate_variant("R0")
    r1_folds, r1_patients = evaluate_variant("R1")
    comparisons = []
    for a, b in zip(r0_folds, r1_folds, strict=True):
        if a["fold"] != b["fold"]:
            raise RuntimeError("Mismatched outer folds")
        comparisons.append({"fold": a["fold"], "n_patients": a["n_patients"], **{
            f"delta_{key}": b[key] - a[key] for key in (
                "patient_macro_f1", "patient_macro_ez_f1", "patient_macro_nez_f1",
                "patient_macro_balanced_accuracy", "patient_macro_auprc_ez", "patient_macro_auroc_ez",
                "patient_macro_ez_mrr", "top1_is_ez_rate",
            )
        }})
    csv_write(PUBLIC / "OUTER_COMPARISON.csv", comparisons)
    by_id_r0 = {row["subject_id"]: row for row in r0_patients}
    by_id_r1 = {row["subject_id"]: row for row in r1_patients}
    if set(by_id_r0) != set(by_id_r1):
        raise RuntimeError("R0/R1 outer patient identities differ")
    ids = sorted(by_id_r0)
    rng = np.random.default_rng(42042)
    boot = []
    for key in FIELDS:
        deltas = np.asarray([by_id_r1[sid][key] - by_id_r0[sid][key] for sid in ids])
        draws = rng.integers(0, len(ids), size=(2000, len(ids)))
        replicates = deltas[draws].mean(axis=1)
        boot.append({"metric": key, "R0": float(np.mean([by_id_r0[sid][key] for sid in ids])),
                     "R1": float(np.mean([by_id_r1[sid][key] for sid in ids])),
                     "delta": float(deltas.mean()), "ci95_low": float(np.quantile(replicates, 0.025)),
                     "ci95_high": float(np.quantile(replicates, 0.975)), "bootstrap_replicates": 2000})
    if any(not math.isfinite(float(value)) for row in boot for value in row.values() if isinstance(value, (int, float))):
        raise RuntimeError("Nonfinite outer bootstrap result")
    csv_write(PUBLIC / "PAIRED_BOOTSTRAP.csv", boot)
    print(json.dumps({"status": "OUTER_EVALUATED_ONCE", "bootstrap": boot}, indent=2), flush=True)


if __name__ == "__main__":
    main()
