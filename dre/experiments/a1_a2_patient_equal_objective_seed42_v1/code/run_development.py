"""Thirty-epoch A0/A1/A2 objective-only development on fixed fit/validation roles.

This module never builds an outer-test loader or reads prior outer result files.
All patient-level predictions, optimizer states, and checkpoints stay private.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import sys
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve()
EXPERIMENT = HERE.parents[1]
sys.path.insert(0, str(HERE.parents[2] / "r1_hlv_ictal_dynamics_seed42_v1" / "code"))
from run_matched import (SOURCE_ROOT, assert_sources, build_fold, install_interleaved_hlv_view,
                         make_args, sha256)  # noqa: E402
sys.path.insert(0, str(SOURCE_ROOT))
import exp_ez_hybrid as core  # noqa: E402
from development_metrics import THRESHOLDS, epoch_grid, finalize_fold  # noqa: E402
from objectives import beta_at_epoch, patient_equal_weighted_bce_loss, patient_soft_macro_f1_loss, per_patient_weighted_bce  # noqa: E402


VARIANTS = ("A0", "A1", "A2")
EXPECTED_LOCK_SHA256 = "6694da1b351d7130015661fcce1ec91bec643821392e409f1d5728f472b3aba0"
RUNTIME = Path(os.environ.get("A1_A2_RUNTIME", ""))


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def preflight() -> None:
    if not os.environ.get("A1_A2_RUNTIME") or not RUNTIME.is_absolute():
        raise RuntimeError("A1_A2_RUNTIME must be an absolute private runtime directory")
    if sha256(EXPERIMENT / "PROTOCOL_LOCK.json") != EXPECTED_LOCK_SHA256:
        raise RuntimeError("Development protocol lock changed")
    assert_sources()


def assert_architecture_args(args) -> None:
    if bool(args.use_physics_dynamics) or str(args.loss_mode) != "masked_bce" or str(args.class_weight_mode) != "ez_negative":
        raise RuntimeError("R0 architecture/objective arguments changed")
    if float(args.ez_negative_weight) != 2.0 or int(args.model_dim) != 32 or int(args.num_heads) != 2:
        raise RuntimeError("R0 backbone settings changed")
    if int(args.patient_batch_size) != 2 or str(args.positive_label).lower() != "nez":
        raise RuntimeError("R0 batch/label settings changed")
    prohibited = (
        "use_v3_qbc", "use_v3_rcc", "use_diffusion_residual", "use_negative_anchor_head",
        "use_two_expert_router", "use_n6_dual_view_ema", "use_view_gated_fusion",
        "use_ez_ranking_loss", "use_hard_topk_loss", "use_broad_ez_mil_loss",
        "use_a9v8_lcbo", "use_edf_quality_weighting", "use_raw_window_encoder",
    )
    if any(bool(getattr(args, key, False)) for key in prohibited):
        raise RuntimeError("Prohibited auxiliary mechanism enabled")


def assert_cohort(exp) -> None:
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("Expected 80 patients and five fixed folds")
    channels = sum(len(meta["canonical_channels"]) for meta in exp.patient_index.values())
    if channels != 7635 or len(exp.run_records) != 256:
        raise RuntimeError(f"Current data cohort mismatch: channels={channels}, records={len(exp.run_records)}")
    groups = [set(split["test_subjects"]) for split in exp.outer_splits]
    if len(set().union(*groups)) != 80 or sum(map(len, groups)) != 80:
        raise RuntimeError("Fixed test memberships do not partition 80 patients")


def class_support_for_fold(train_set, val_set, fold: int) -> dict:
    output = {"fold": fold}
    for name, dataset in (("fit", train_set), ("validation", val_set)):
        missing_ez = missing_nez = total_channels = 0
        for example in dataset.patient_examples:
            valid = np.asarray(example["channel_mask"], dtype=bool)
            yez = np.asarray(example["labels_ez"], dtype=float)[valid]
            total_channels += int(valid.sum())
            missing_ez += int(not np.any(yez == 1))
            missing_nez += int(not np.any(yez == 0))
        output[f"{name}_patients"] = len(dataset)
        output[f"{name}_channels"] = total_channels
        output[f"{name}_missing_ez"] = missing_ez
        output[f"{name}_missing_nez"] = missing_nez
    if output["validation_patients"] != 13:
        raise RuntimeError("VLOO requires exactly 13 validation patients per fold")
    return output


def audit_class_support(exp) -> dict:
    rows = []
    for split in exp.outer_splits:
        fold, train_set, _, val_loader, _, _ = build_fold(exp, split, "validation")
        rows.append(class_support_for_fold(train_set, val_loader.dataset, fold))
    payload = {
        "cohort_patients": 80, "no_outer_test_examples_built": True,
        "folds": rows,
        "all_active_fit_validation_patients_have_both_classes": all(
            row["fit_missing_ez"] == row["fit_missing_nez"] == row["validation_missing_ez"] == row["validation_missing_nez"] == 0
            for row in rows
        ),
        "missing_class_soft_f1_rule": "true missing class gets soft F1=0, matching zero_division=0",
    }
    atomic_json(EXPERIMENT / "CLASS_SUPPORT_AUDIT.json", payload)
    if not payload["all_active_fit_validation_patients_have_both_classes"]:
        raise RuntimeError("Missing class found; stop before training and audit metric semantics")
    return payload


def custom_compute_loss(variant: str):
    def compute(self, outputs, batch, ez_negative_weight, *, split_name="train"):
        if variant not in ("A1", "A2"):
            raise RuntimeError("Custom objective is only for A1/A2")
        if float(ez_negative_weight) != 2.0:
            raise RuntimeError("R0 EZ class weight changed")
        logits = outputs["logits"]
        labels_nez, labels_ez, mask = batch["labels"], batch["labels_ez"], batch["channel_mask"]
        bce = patient_equal_weighted_bce_loss(logits, labels_nez, labels_ez, mask)
        parts = {"bce": float(bce.detach()), "patient_equal_bce": float(bce.detach())}
        if variant == "A1":
            return bce, parts
        soft, soft_nez, soft_ez = patient_soft_macro_f1_loss(logits, labels_nez, labels_ez, mask)
        beta = beta_at_epoch(int(self.current_epoch))
        loss = bce + beta * soft
        parts.update({"soft_macro_f1_loss": float(soft.detach()), "soft_nez_f1": float(soft_nez.detach()),
                      "soft_ez_f1": float(soft_ez.detach()), "beta": beta})
        return loss, parts
    return compute


def check_grid_against_core(raw_records: list[dict], payload: dict) -> None:
    midpoint = 9
    summary, enriched = core._summarize_prediction_records(raw_records, classification_threshold=float(THRESHOLDS[midpoint]))
    by_id = {str(row["subject_id"]): row for row in enriched}
    for row in payload["patients"]:
        reference = by_id[row["subject_id"]]
        for field, core_field in (("patient_macro_f1", "patient_macro_f1"),
                                  ("patient_ez_f1", "patient_ez_f1"),
                                  ("patient_nez_f1", "patient_nez_f1"),
                                  ("patient_balanced_accuracy", "patient_balanced_accuracy")):
            if not math.isclose(row["grid"][field][midpoint], float(reference[core_field]), abs_tol=1e-8):
                raise RuntimeError(f"Patient metric implementation disagrees with core: {field}")
    mean = np.mean([row["grid"]["patient_macro_f1"][midpoint] for row in payload["patients"]])
    if not math.isclose(float(mean), float(summary["patient_macro_f1"]), abs_tol=1e-8):
        raise RuntimeError("Validation metric grid disagrees with core mean")


def fit_loss_diagnostics(exp, model, train_set, fold: int, variant: str) -> dict:
    from exp_ez_hybrid import _move_tensors_to_device
    loader = exp._make_loader(train_set, shuffle=False, batch_size=2)
    model.eval()
    losses, counts = [], []
    with torch.no_grad():
        for batch in loader:
            batch = _move_tensors_to_device(batch, exp.device)
            logits = model(batch)["logits"]
            per_patient, active = per_patient_weighted_bce(logits, batch["labels"], batch["labels_ez"], batch["channel_mask"])
            losses.extend(per_patient[active].detach().cpu().numpy().tolist())
            counts.extend(batch["channel_mask"].sum(dim=1)[active].detach().cpu().numpy().tolist())
    if not losses:
        raise RuntimeError("No active fit patients for diagnostics")
    correlation = float(np.corrcoef(counts, losses)[0, 1]) if np.std(counts) > 0 and np.std(losses) > 0 else 0.0
    return {"variant": variant, "fold": fold, "fit_patients": len(losses),
            "per_patient_weighted_bce_mean": float(np.mean(losses)),
            "per_patient_weighted_bce_std": float(np.std(losses)),
            "channel_count_vs_patient_loss_correlation": correlation}


def train_variant(exp, args, fold: int, train_set, train_loader, val_loader, normalizer,
                  initial_state: dict, original_compute_loss, variant: str) -> None:
    variant_dir = RUNTIME / variant / f"fold_{fold}"
    variant_dir.mkdir(parents=True, exist_ok=True)
    if (variant_dir / "development_summary.json").exists():
        print(f"[{variant}] fold={fold}: 30 epochs and VLOO already complete", flush=True)
        return
    exp._compute_loss = (original_compute_loss if variant == "A0" else
                         types.MethodType(custom_compute_loss(variant), exp))
    model = exp.runtime["model_cls"](args).to(exp.device)
    exp._dry_initialize_lazy_layers(model, train_loader)
    model.load_state_dict(initial_state, strict=True)
    if any(not torch.equal(model.state_dict()[key].detach().cpu(), value) for key, value in initial_state.items()):
        raise RuntimeError("Initial model state differs across variants")
    count = sum(parameter.numel() for parameter in model.parameters())
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
    ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
    for epoch in range(1, 31):
        exp.current_epoch = epoch
        ckpt = variant_dir / f"epoch_{epoch:02d}.pt"
        metrics_path = variant_dir / f"epoch_{epoch:02d}_validation_grid.json"
        if ckpt.exists():
            state = torch.load(ckpt, map_location=exp.device, weights_only=False)
            if int(state["epoch"]) != epoch or state["variant"] != variant or int(state["fold"]) != fold:
                raise RuntimeError("Resume checkpoint identity mismatch")
            model.load_state_dict(state["model_state_dict"], strict=True)
            optimizer.load_state_dict(state["optimizer_state_dict"])
            train_metrics = state["train_metrics"]
        else:
            core._set_random_seed(42 * 100000 + fold * 1000 + epoch)
            train_metrics = exp._train_one_epoch(model, train_loader, optimizer, ez_weight)
            if not math.isfinite(float(train_metrics["loss"])):
                raise RuntimeError("Nonfinite training loss")
            torch.save({"epoch": epoch, "fold": fold, "variant": variant,
                        "model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics, "normalizer_mean": normalizer.mean,
                        "normalizer_std": normalizer.std,
                        "normalizer_physics_mean": normalizer.physics.mean,
                        "normalizer_physics_std": normalizer.physics.std}, ckpt)
        if not metrics_path.exists():
            _, _, raw_records = exp._evaluate(model, val_loader, ez_weight, split_name="val")
            grid = epoch_grid(raw_records, epoch)
            check_grid_against_core(raw_records, grid)
            atomic_json(metrics_path, grid)
        print(f"[{variant}] fold={fold} epoch={epoch}/30 train_loss={train_metrics['loss']:.6f}", flush=True)
    payloads = [json.loads((variant_dir / f"epoch_{epoch:02d}_validation_grid.json").read_text(encoding="utf-8"))
                for epoch in range(1, 31)]
    private_patient_path = RUNTIME / "development_private" / f"{variant}_fold_{fold}_VLOO_PATIENT_PRIVATE.csv"
    public_fold, fullval = finalize_fold(payloads, variant, fold, private_patient_path)
    selected_path = variant_dir / f"epoch_{fullval['selected_epoch']:02d}.pt"
    fullval["checkpoint_sha256"] = sha256(selected_path)
    diagnostics = fit_loss_diagnostics(exp, model, train_set, fold, variant)
    atomic_json(variant_dir / "development_summary.json",
                {"vloo": public_fold, "fullval": fullval, "loss_diagnostics": diagnostics,
                 "parameter_count": count, "all_30_epochs_completed": True})
    print(f"[{variant}] fold={fold} VLOO Macro-F1={public_fold['patient_macro_f1']:.6f} "
          f"fullval_epoch={fullval['selected_epoch']}", flush=True)


def train_all(exp, only_fold: int | None) -> None:
    unit_path = EXPERIMENT / "A1_LOSS_UNIT_TEST.json"
    class_path = EXPERIMENT / "CLASS_SUPPORT_AUDIT.json"
    if not unit_path.is_file() or not json.loads(unit_path.read_text(encoding="utf-8")).get("pass"):
        raise RuntimeError("A1 loss unit tests have not passed")
    if not class_path.is_file() or not json.loads(class_path.read_text(encoding="utf-8")).get(
        "all_active_fit_validation_patients_have_both_classes"
    ):
        raise RuntimeError("Class-support audit has not passed")
    for split in exp.outer_splits:
        fold = int(split["fold_idx"])
        if only_fold is not None and fold != only_fold:
            continue
        fold, train_set, train_loader, val_loader, _, normalizer = build_fold(exp, split, "validation")
        args = exp.args
        initial_path = RUNTIME / "initial" / f"fold_{fold}_initial.pt"
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        core._set_random_seed(42 + fold)
        reference_model = exp.runtime["model_cls"](args).to(exp.device)
        exp._dry_initialize_lazy_layers(reference_model, train_loader)
        fresh_state = {key: value.detach().cpu().clone() for key, value in reference_model.state_dict().items()}
        if initial_path.exists():
            initial_state = torch.load(initial_path, map_location="cpu", weights_only=True)
            if set(initial_state) != set(fresh_state) or any(initial_state[key].shape != fresh_state[key].shape for key in fresh_state):
                raise RuntimeError("Saved initial architecture differs from current architecture")
        else:
            initial_state = fresh_state
            torch.save(initial_state, initial_path)
        original_compute_loss = exp._compute_loss
        for variant in VARIANTS:
            train_variant(exp, args, fold, train_set, train_loader, val_loader, normalizer,
                          initial_state, original_compute_loss, variant)
        exp._compute_loss = original_compute_loss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--fold", type=int, choices=(1, 2, 3, 4, 5))
    options = parser.parse_args()
    preflight()
    install_interleaved_hlv_view()
    args = make_args("R0", RUNTIME / "A0")
    assert_architecture_args(args)
    exp = core.Exp_EZHybridLocalization(args)
    assert_cohort(exp)
    if options.audit_only:
        print(json.dumps(audit_class_support(exp), indent=2), flush=True)
        return
    train_all(exp, options.fold)


if __name__ == "__main__":
    main()
