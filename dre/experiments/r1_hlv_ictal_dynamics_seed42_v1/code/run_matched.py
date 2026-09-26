"""Matched R0/R1 window-level experiment on the frozen 80-patient protocol.

This runner deliberately separates validation training from outer evaluation.
Private checkpoints and patient records stay outside the Git repository.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch


EXPERIMENT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = Path(r"E:\DRE-nips\new-pipeline\7-11")
CACHE = Path(r"D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl")
SPLIT = Path(r"D:\nips-temp\task1_aaai_completion_training\audit\fixed_partition_manifest.csv")
RUNTIME = Path(r"D:\nips-temp\r1_hlv_ictal_dynamics_seed42_v1")
EXPECTED_CACHE_SHA = "9b5bb58a0175aba494ad79e30c5a65c7cee33c7d464273f9bcf62b9b280b0087"
EXPECTED_SPLIT_SHA = "fd897fa7eed2c521fd5b14c1ae95d91b5d43b2ae85822317dcda08a878f58278"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_sources() -> None:
    for path, expected in ((CACHE, EXPECTED_CACHE_SHA), (SPLIT, EXPECTED_SPLIT_SHA)):
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"Missing or changed frozen input: {path}")


def install_interleaved_hlv_view() -> None:
    """Interleave delta/zdelta by feature without changing their values."""
    from neuroez_c import dataset

    original = dataset.physics_state_features

    def interleaved(features, centers, args=None):
        output = original(features, centers, args)
        if output.shape[-1] != 6:
            raise RuntimeError(f"HLV must contain exactly six columns, got {output.shape[-1]}")
        # Source implementation with zdelta,delta gives [Hz,Lz,Vz,Hd,Ld,Vd].
        # The locked R1 order is [Hd,Hz,Ld,Lz,Vd,Vz].
        return output[..., [3, 0, 4, 1, 5, 2]]

    dataset.physics_state_features = interleaved


def make_args(variant: str, output_dir: Path):
    from neuroez_c.config import apply_pruned_defaults
    from run_neuroez_c import build_parser

    args = build_parser().parse_args([])
    args.window_cache_path = str(CACHE)
    args.fixed_split_manifest = str(SPLIT)
    args.allowed_subjects_ledger = str(SPLIT)
    args.require_n_patients = 80
    args.output_dir = str(output_dir)
    args.random_seed = 42
    args.positive_label = "nez"
    args.outcome_subset = "all"
    args.drop_high_ez_fraction_lzu = False
    args.use_patient_relative_z = True
    args.use_physics_dynamics = variant == "R1"
    args.physics_state_features = "log_bp_high_gamma,line_length_per_sec,variance"
    args.physics_feature_parts = "zdelta,delta"
    args.physics_gate_init = -4.0
    args.physics_loss_weight = 0.0
    args.physics_source_sparse_weight = 0.0
    args.physics_velocity_l2_weight = 0.0
    args.epochs = 30
    args.patience = 6
    args.min_epochs_before_early_stop = 0
    args.early_stop_metric = "patient_macro_f1"
    args.learning_rate = 1e-4
    args.weight_decay = 1e-3
    args.patient_batch_size = 2
    args.batch_size = 2
    args.model_dim = 32
    args.num_heads = 2
    args.dropout = 0.4
    args.temporal_pooling = "mean"
    args.record_pooling = "mean"
    args.class_weight_mode = "ez_negative"
    args.ez_negative_weight = "2"
    args.patient_loss_weighting = "uniform"
    args.loss_mode = "masked_bce"
    apply_pruned_defaults(args)
    if any(bool(getattr(args, flag, False)) for flag in (
        "use_v3_qbc", "use_v3_rcc", "use_diffusion_residual", "use_two_expert_router",
        "use_view_gated_fusion", "use_n6_dual_view_ema", "use_negative_anchor_head",
    )):
        raise RuntimeError("A prohibited additional branch is enabled")
    return args


class GateTracker:
    def __init__(self, model):
        self._mask = None
        self.values: list[np.ndarray] = []
        self.abs_residual: list[float] = []
        self.norm_ratio: list[float] = []
        self.pre_handle = model.register_forward_pre_hook(self._pre)
        self.handle = model.physics_gate.register_forward_hook(self._post)

    def _pre(self, _module, inputs):
        batch = inputs[0]
        self._mask = batch["window_mask"][:, :, :, None] & batch["seizure_channel_mask"][:, :, None, :]

    def _post(self, _module, inputs, output):
        with torch.no_grad():
            gate = torch.sigmoid(output.detach())
            h_b0, h_dyn = inputs[0].detach().chunk(2, dim=-1)
            mask = self._mask
            if mask is None or not bool(mask.any()):
                raise RuntimeError("No valid HLV positions for gate diagnostic")
            selected_gate = gate[mask.expand_as(gate)].float()
            residual = (gate * h_dyn)[mask.expand_as(gate)].float()
            base = h_b0[mask.expand_as(gate)].float()
            self.values.append(selected_gate.cpu().numpy())
            self.abs_residual.append(float(residual.abs().mean().cpu()))
            self.norm_ratio.append(float(torch.linalg.vector_norm(residual).div(
                torch.linalg.vector_norm(base).clamp_min(1e-12)).cpu()))

    def summary(self):
        values = np.concatenate(self.values) if self.values else np.asarray([], dtype=np.float32)
        if not len(values):
            raise RuntimeError("HLV gate diagnostics were not captured")
        return {
            "gate_mean": float(values.mean()),
            "gate_median": float(np.median(values)),
            "gate_q10": float(np.quantile(values, 0.1)),
            "gate_q90": float(np.quantile(values, 0.9)),
            "mean_abs_gated_residual": float(np.mean(self.abs_residual)),
            "gated_residual_to_base_norm_ratio": float(np.mean(self.norm_ratio)),
        }

    def close(self):
        self.handle.remove()
        self.pre_handle.remove()


def numeric_summary(summary, records, fold, variant, epoch, threshold):
    fields = (
        "patient_macro_f1", "patient_macro_ez_f1", "patient_macro_nez_f1",
        "patient_macro_balanced_accuracy", "patient_macro_auprc_ez", "patient_macro_auroc_ez",
        "patient_macro_ez_mrr", "top1_is_ez_rate",
    )
    result = {"fold": fold, "variant": variant, "selected_epoch": epoch,
              "validation_selected_threshold": threshold, "n_patients": len(records)}
    result.update({key: float(summary[key]) for key in fields})
    result["predicted_ez_fraction"] = float(np.mean([
        float(record["predicted_ez_count"]) / max(int(record["n_channels"]), 1)
        for record in records
    ]))
    if any(not math.isfinite(float(value)) for value in result.values() if isinstance(value, (int, float))):
        raise RuntimeError("Nonfinite evaluation summary")
    return result


def build_fold(exp, split, stage):
    import exp_ez_hybrid as core

    fold = int(split["fold_idx"])
    core._set_random_seed(42 + fold)
    fit, val = list(split["fit_subjects"]), list(split["validation_subjects"])
    test = list(split["test_subjects"]) if stage == "outer" else []
    if set(fit) & set(val) or set(fit) & set(split["test_subjects"]) or set(val) & set(split["test_subjects"]):
        raise RuntimeError("Fixed fold contains overlapping roles")
    train_set, val_set, test_set, normalizer = exp._build_datasets(fit, val, test)
    if len(train_set) != len(fit) or len(val_set) != len(val):
        raise RuntimeError("Expected one dataset example per fit/validation patient")
    train_loader = exp._make_loader(train_set, shuffle=True, batch_size=2)
    val_loader = exp._make_loader(val_set, shuffle=False, batch_size=2)
    test_loader = exp._make_loader(test_set, shuffle=False, batch_size=2) if stage == "outer" else None
    return fold, train_set, train_loader, val_loader, test_loader, normalizer


def load_shared_initialization(model, fold_dir, variant):
    init_path = RUNTIME / "R0" / f"fold_{fold_dir}" / "shared_initial_state.pt"
    if variant == "R0":
        torch.save({key: value.detach().cpu() for key, value in model.state_dict().items()}, init_path)
        return
    if not init_path.exists():
        raise RuntimeError(f"R0 initialization is missing: {init_path}")
    reference = torch.load(init_path, map_location="cpu", weights_only=True)
    target = model.state_dict()
    common = {key: value for key, value in reference.items() if key in target and target[key].shape == value.shape}
    if not common or "b0_encoder" not in " ".join(common):
        raise RuntimeError("Matched shared initialization is empty")
    target.update(common)
    model.load_state_dict(target)
    if any(not torch.equal(model.state_dict()[key].cpu(), value) for key, value in common.items()):
        raise RuntimeError("R0/R1 common initial state mismatch")


def run_validation(variant):
    import exp_ez_hybrid as core

    output_dir = RUNTIME / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    args = make_args(variant, output_dir)
    exp = core.Exp_EZHybridLocalization(args)
    if len(exp.patient_index) != 80 or len(exp.outer_splits) != 5:
        raise RuntimeError("Active cohort/splits are not 80 patients/five folds")
    channels = sum(len(meta["canonical_channels"]) for meta in exp.patient_index.values())
    if channels != 7635:
        raise RuntimeError(f"Active cohort has {channels} rather than 7,635 channels")
    test_groups = [set(split["test_subjects"]) for split in exp.outer_splits]
    if set().union(*test_groups) != set(exp.patient_index) or sum(map(len, test_groups)) != 80:
        raise RuntimeError("Frozen outer-test membership is not a disjoint 80-patient partition")
    for split in exp.outer_splits:
        fold = int(split["fold_idx"])
        fold_dir = output_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        result_path = fold_dir / "validation_summary.json"
        if result_path.exists():
            print(f"[{variant}] fold {fold}: completed; preserving result", flush=True)
            continue
        fold, train_set, train_loader, val_loader, _, normalizer = build_fold(exp, split, "validation")
        model = exp.runtime["model_cls"](args).to(exp.device)
        exp._dry_initialize_lazy_layers(model, train_loader)
        load_shared_initialization(model, fold, variant)
        core._set_random_seed(42 + fold)
        ez_weight = torch.tensor(2.0, dtype=torch.float32, device=exp.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
        best_score = -float("inf")
        best_epoch = 0
        best_threshold = 0.5
        stale = 0
        ckpt_path = fold_dir / "selected.pt"
        for epoch in range(1, 31):
            exp.current_epoch = epoch
            train_metrics = exp._train_one_epoch(model, train_loader, optimizer, ez_weight)
            _, _, raw_records = exp._evaluate(model, val_loader, ez_weight, split_name="val")
            threshold = core._select_validation_classification_threshold(raw_records, positive_label="nez")
            summary, _ = core._summarize_prediction_records(raw_records, classification_threshold=threshold)
            score = float(summary["patient_macro_f1"])
            if not math.isfinite(score) or not math.isfinite(float(train_metrics["loss"])):
                raise RuntimeError(f"Nonfinite training/validation at {variant} fold={fold} epoch={epoch}")
            print(f"[{variant}] fold={fold} epoch={epoch} train_loss={train_metrics['loss']:.6f} val_f1={score:.6f} threshold={threshold:.3f}", flush=True)
            if score > best_score + 1e-12:
                best_score, best_epoch, best_threshold, stale = score, epoch, threshold, 0
                torch.save({
                    "model_state_dict": model.state_dict(), "selected_epoch": epoch,
                    "threshold": threshold, "normalizer_mean": normalizer.mean,
                    "normalizer_std": normalizer.std,
                    "normalizer_physics_mean": normalizer.physics.mean,
                    "normalizer_physics_std": normalizer.physics.std,
                    "variant": variant, "fold": fold,
                }, ckpt_path)
            else:
                stale += 1
            if stale >= 6:
                break
        checkpoint = torch.load(ckpt_path, map_location=exp.device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        tracker = GateTracker(model) if variant == "R1" else None
        _, _, raw_records = exp._evaluate(model, val_loader, ez_weight, split_name="val")
        summary, records = core._summarize_prediction_records(raw_records, classification_threshold=best_threshold)
        row = numeric_summary(summary, records, fold, variant, best_epoch, best_threshold)
        if tracker is not None:
            row.update(tracker.summary())
            tracker.close()
        result_path.write_text(json.dumps(row, indent=2, sort_keys=True), encoding="utf-8")
        print(f"[{variant}] fold {fold}: selected epoch {best_epoch}, validation Macro-F1={row['patient_macro_f1']:.6f}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=("R0", "R1"), required=True)
    parser.add_argument("--stage", choices=("validation",), default="validation")
    cli = parser.parse_args()
    assert_sources()
    sys.path.insert(0, str(SOURCE_ROOT))
    install_interleaved_hlv_view()
    run_validation(cli.variant)


if __name__ == "__main__":
    main()
