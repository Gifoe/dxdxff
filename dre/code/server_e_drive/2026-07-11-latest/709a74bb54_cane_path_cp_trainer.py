"""Leak-free nested training orchestration for N8F CANE-PATH-CP."""

from __future__ import annotations

import copy
import gc
import json
import math
import pickle
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader

from data_factory import split_train_val_subjects

from .cane_path_cp_heads import initialize_clean_nez_prototypes
from .cane_path_cp_loss import CenterBalancedPatientBatchSampler
from .cane_path_threshold import (
    PATH_SUMMARY_FIELDS,
    PatientAdaptiveThresholdHead,
    apply_patient_adaptive_threshold,
    build_path_summary,
    oracle_standardized_threshold,
    path_training_loss,
)


def _move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def build_inner_crossfit_splits(subjects: Sequence[str], n_splits: int = 4, seed: int = 42) -> list[dict[str, Any]]:
    ordered = np.asarray(sorted(map(str, subjects)), dtype=object)
    splitter = KFold(n_splits=int(n_splits), shuffle=True, random_state=int(seed))
    rows = []
    heldout_union: list[str] = []
    for fold, (fit_idx, heldout_idx) in enumerate(splitter.split(ordered), start=1):
        fit = ordered[fit_idx].tolist()
        heldout = ordered[heldout_idx].tolist()
        if set(fit) & set(heldout):
            raise RuntimeError("Inner cross-fitting patient leakage")
        heldout_union.extend(heldout)
        rows.append({"inner_fold": fold, "fit_subjects": fit, "heldout_subjects": heldout})
    if sorted(heldout_union) != sorted(ordered.tolist()) or len(set(heldout_union)) != len(ordered):
        raise RuntimeError("Inner OOF union must cover every outer-train patient exactly once")
    return rows


def _loader(experiment: Any, dataset: Any, *, train: bool, seed: int) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    if train and bool(getattr(experiment.args, "center_balanced_batches", False)):
        centers = [str(item["center"]) for item in dataset.patient_examples]
        sampler = CenterBalancedPatientBatchSampler(
            centers,
            seed=seed,
            shuffle=True,
            batch_size=int(getattr(experiment.args, "patient_batch_size", 4)),
        )
        return DataLoader(
            dataset, batch_sampler=sampler, num_workers=int(getattr(experiment.args, "num_workers", 0)),
            collate_fn=experiment.runtime["collate_fn"], pin_memory=experiment.device.type == "cuda", generator=generator,
        )
    return DataLoader(
        dataset, batch_size=int(getattr(experiment.args, "patient_batch_size", 4)), shuffle=train,
        num_workers=int(getattr(experiment.args, "num_workers", 0)),
        collate_fn=experiment.runtime["collate_fn"], pin_memory=experiment.device.type == "cuda", generator=generator,
    )


def _optimizer(model: torch.nn.Module, args: Any) -> torch.optim.Optimizer:
    head_names = ("clean_nez_anchor", "multi_seizure_evidence", "causal_propagation_residual", "p2_scope_boundary", "p2_scope_cardinality")
    backbone, heads = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (heads if any(token in name for token in head_names) else backbone).append(parameter)
    return torch.optim.AdamW(
        [
            {"params": backbone, "lr": float(getattr(args, "cane_backbone_lr_stage1", 1e-4)), "group_name": "backbone"},
            {"params": heads, "lr": float(getattr(args, "cane_head_lr", 1e-4)), "group_name": "cane_heads"},
        ],
        weight_decay=float(getattr(args, "weight_decay", 1e-3)),
    )


@torch.no_grad()
def collect_ranking_records(
    experiment: Any,
    model: torch.nn.Module,
    loader: DataLoader,
    *,
    source: str,
    p23_disable_temporal: bool = False,
    p23_teacher_model: torch.nn.Module | None = None,
) -> list[dict[str, Any]]:
    model.eval()
    records: list[dict[str, Any]] = []
    for batch in loader:
        device_batch = _move(batch, experiment.device)
        if p23_disable_temporal:
            # This flag is consumed only by the P23 forward path.  It never
            # changes labels, thresholds, or the fitted checkpoint.
            device_batch = dict(device_batch)
            device_batch["p23_disable_temporal"] = True
        outputs = model(device_batch)
        # P23 reliability is a training-only target construction.  Persisting
        # it for held-out diagnostics must not change the formal prediction;
        # use the selected student checkpoint as the detached reference.
        if bool(getattr(experiment, "use_p23_trn_nez", False)) and not bool(getattr(experiment.args, "p23_use_p2_loss", False)):
            from .p23_noisy_ez_loss import compute_p23_loss

            teacher_score = None
            if p23_teacher_model is not None:
                p23_teacher_model.eval()
                teacher_score = p23_teacher_model(device_batch)["final_score_nez"]
            compute_p23_loss(
                outputs,
                device_batch,
                experiment.args,
                epoch=int(getattr(experiment, "current_epoch", 1)),
                teacher_score_nez=teacher_score if teacher_score is not None else outputs["final_score_nez"].detach(),
            )
        if bool(getattr(experiment, "use_p23_trn_nez", False)):
            # P23 explicitly forbids the PATH threshold head.  Its ledgers
            # still retain compatible placeholder diagnostics for shared CSV
            # writers, but must never invoke PATH's 23-D finite contract.
            direct_logits = outputs["direct_nez_logit"]
            summary = torch.zeros(
                (*direct_logits.shape, 23),
                dtype=direct_logits.dtype,
                device=direct_logits.device,
            )
            standardized = outputs.get("r_direct", torch.zeros_like(direct_logits))
        else:
            summary, standardized = build_path_summary(outputs, device_batch)
        for patient_idx, subject_id in enumerate(batch["subject_id"]):
            # The collator pads all channel tensors to the largest patient in
            # a batch.  Persist only this patient's canonical channel extent;
            # otherwise output ledgers mix padded labels with shorter names.
            canonical_channels = list(batch["canonical_channels"][patient_idx])
            n_channels = len(canonical_channels)
            channel_mask = device_batch["channel_mask"][patient_idx, :n_channels].detach().cpu().numpy().astype(bool)
            labels_ez = device_batch["labels_ez"][patient_idx, :n_channels].detach().cpu().numpy()
            labels_nez = np.where(labels_ez >= 0, 1.0 - labels_ez, -1.0).astype(np.float32)
            cp = outputs["causal_propagation_features"][patient_idx, :n_channels].detach().cpu().numpy()
            record = {
                "subject_id": str(subject_id), "center": str(batch["center"][patient_idx]),
                "center_id": int(device_batch["center_id"][patient_idx].item()),
                "canonical_channels": canonical_channels,
                "channel_mask": channel_mask, "labels_nez": labels_nez, "labels_ez": labels_ez,
                "score_nez": outputs["final_score_nez"][patient_idx, :n_channels].detach().cpu().numpy(),
                "score_ez": outputs["final_score_ez"][patient_idx, :n_channels].detach().cpu().numpy(),
                "direct_score_nez": outputs["direct_score_nez"][patient_idx, :n_channels].detach().cpu().numpy(),
                "direct_nez_logit": outputs["direct_nez_logit"][patient_idx, :n_channels].detach().cpu().numpy(),
                "final_nez_logit": outputs["final_nez_logit"][patient_idx, :n_channels].detach().cpu().numpy(),
                # Persist the frozen channel representation for downstream
                # adapters. It is never used as a center-conditioned input.
                "contextual_channel_embedding": outputs["contextual_channel_embedding"][patient_idx, :n_channels].detach().cpu().numpy(),
                "standardized_nez_logit": standardized[patient_idx, :n_channels].detach().cpu().numpy(),
                "anchor_residual": outputs["anchor_residual"][patient_idx, :n_channels].detach().cpu().numpy(),
                "anchor_distance": outputs["anchor_distance"][patient_idx, :n_channels].detach().cpu().numpy(),
                "anchor_distance_z": outputs["anchor_distance_z"][patient_idx, :n_channels].detach().cpu().numpy(),
                "anchor_nez_evidence": outputs["anchor_nez_evidence"][patient_idx, :n_channels].detach().cpu().numpy(),
                "seizure_residual": outputs["seizure_residual"][patient_idx, :n_channels].detach().cpu().numpy(),
                "seizure_nez_probability_mean": outputs["seizure_nez_probability_mean"][patient_idx, :n_channels].detach().cpu().numpy(),
                "seizure_nez_probability_std": outputs["seizure_nez_probability_std"][patient_idx, :n_channels].detach().cpu().numpy(),
                "seizure_nez_agreement": outputs["seizure_nez_agreement"][patient_idx, :n_channels].detach().cpu().numpy(),
                "causal_propagation_features": cp,
                "causal_feature_valid": outputs["causal_feature_valid"][patient_idx, :n_channels].detach().cpu().numpy().astype(bool),
                "causal_residual": outputs["causal_residual"][patient_idx, :n_channels].detach().cpu().numpy(),
                "path_summary": summary[patient_idx].detach().cpu().numpy(),
                "n_seizures": int(device_batch["seizure_mask"][patient_idx].sum().item()),
                "inner_fold_source": source,
            }
            for name in (
                "scope_boundary_delta", "scope_nez_logit", "scope_score_nez", "scope_score_ez", "scope_ez_score", "patient_relative_direct_rank",
                "v3_anchor_standardized", "v3_anchor_valid", "direct_standardized",
                "u_anchor", "u_seizure", "seizure_term", "raw_u_causal", "u_causal", "q_cp",
                "q_coverage", "q_seizure", "q_stability", "w_noop", "w_anchor",
                "w_seizure", "w_causal", "fusion_entropy", "delta",
                "ez_reliability", "reliability_component_count", "seizure_evidence_valid",
                "cp_early_source_rank", "correction_saturation", "r_direct",
                "temporal_gate", "temporal_delta_norm", "delta_onset_norm", "delta_spread_norm",
                "delta_late_norm", "slope_norm", "temporal_bin_valid_fraction", "valid_pre",
                "valid_onset", "valid_spread", "valid_late", "slope_valid",
                "seizure_nez_probability_q10", "seizure_agreement", "valid_seizure_count",
                "seizure_nez_logit_mean", "seizure_nez_probability_raw_q10",
                "seizure_nez_robust_tail_logit", "seizure_nez_robust_tail_probability",
                "seizure_nez_soft_tail_logit", "seizure_nez_bounded_tail_gap",
                "seizure_nez_tail_gap", "tail_shrinkage_alpha", "tail_valid", "tail_reliability",
                "tail_residual_suppressed",
                "ema_teacher_score_nez", "observed_ez_reliability", "soft_target_nez",
                "noise_aware_ramp",
            ):
                value = outputs.get(name)
                if torch.is_tensor(value) and value.ndim >= 2:
                    # Channel is the last axis.  Temporal diagnostics can have
                    # shape [patient, seizure, channel], so slicing axis one
                    # would silently corrupt their ledger representation.
                    record[name] = value[patient_idx, ..., :n_channels].detach().cpu().numpy()
            for name in ("scope_count_prior_mu0", "scope_count_mu_delta", "scope_count_predicted_mu", "scope_count_predicted_kappa", "scope_count_alpha", "scope_count_beta", "scope_expected_ez_fraction", "scope_expected_ez_count"):
                value = outputs.get(name)
                if torch.is_tensor(value): record[name] = float(value[patient_idx].detach().cpu())
            if "v3_score_nez" in device_batch:
                record["v3_score_nez"] = device_batch["v3_score_nez"][patient_idx, :n_channels].detach().cpu().numpy()
                record["v3_logit_nez"] = device_batch["v3_logit_nez"][patient_idx, :n_channels].detach().cpu().numpy()
            if bool(getattr(experiment, "use_p21_v3_asrr_nez", False)):
                record["v3_anchor_mode"] = str(getattr(experiment.args, "v3_anchor_mode", "none"))
                record["p21_profile"] = str(getattr(experiment.args, "p21_profile", "R5_FULL"))
            records.append(record)
    return records


def ranking_validation_summary(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    nez_ap, ez_ap, oracle_f1 = [], [], []
    center_values: dict[str, list[tuple[float, float]]] = {}
    for record in records:
        valid = record["channel_mask"]
        labels = record["labels_nez"][valid].astype(int)
        score = record["score_nez"][valid]
        if np.unique(labels).size > 1:
            n_ap = float(average_precision_score(labels, score))
            e_ap = float(average_precision_score(1 - labels, 1 - score))
            nez_ap.append(n_ap); ez_ap.append(e_ap)
            center_values.setdefault(record["center"], []).append((n_ap, e_ap))
        oracle_f1.append(oracle_standardized_threshold(record["standardized_nez_logit"][valid], labels)["oracle_macro_f1"])
    mean_nez = float(np.mean(nez_ap)) if nez_ap else 0.0
    mean_ez = float(np.mean(ez_ap)) if ez_ap else 0.0
    hmean = 2 * mean_nez * mean_ez / max(mean_nez + mean_ez, 1e-12)
    center_hmeans = []
    for values in center_values.values():
        a, b = np.mean(values, axis=0)
        center_hmeans.append(2 * a * b / max(a + b, 1e-12))
    threshold_rows: list[tuple[float, float, float, float, float]] = []
    for threshold in np.linspace(0.05, 0.95, 37):
        patient_macro_f1, patient_nez_f1, patient_ez_f1, balanced = [], [], [], []
        for record in records:
            valid = np.asarray(record["channel_mask"], dtype=bool)
            labels = np.asarray(record["labels_nez"], dtype=np.float32)[valid].astype(int)
            scores = np.asarray(record["score_nez"], dtype=np.float32)[valid]
            prediction = (scores >= float(threshold)).astype(int)
            patient_macro_f1.append(float(f1_score(labels, prediction, average="macro", zero_division=0)))
            patient_nez_f1.append(float(f1_score(labels, prediction, pos_label=1, zero_division=0)))
            patient_ez_f1.append(float(f1_score(labels, prediction, pos_label=0, zero_division=0)))
            balanced.append(float(balanced_accuracy_score(labels, prediction)))
        threshold_rows.append((
            float(np.mean(patient_macro_f1)) if patient_macro_f1 else 0.0,
            float(np.mean(patient_nez_f1)) if patient_nez_f1 else 0.0,
            float(np.mean(patient_ez_f1)) if patient_ez_f1 else 0.0,
            float(np.mean(balanced)) if balanced else 0.0,
            float(threshold),
        ))
    best_macro_f1, best_nez_f1, best_ez_f1, best_balanced_accuracy, best_threshold = max(
        threshold_rows,
        key=lambda row: (row[0], row[1], -abs(row[4] - 0.5)),
    )
    return {
        "patient_macro_auprc_nez": mean_nez, "patient_macro_auprc_ez": mean_ez,
        "balanced_patient_auprc_hmean": hmean, "min_patient_macro_auprc": min(mean_nez, mean_ez),
        "patient_ranking_oracle_macro_f1": float(np.mean(oracle_f1)) if oracle_f1 else 0.0,
        "worst_center_auprc_hmean": min(center_hmeans) if center_hmeans else 0.0,
        "validation_patient_macro_f1": best_macro_f1,
        "validation_patient_nez_f1": best_nez_f1,
        "validation_patient_ez_f1": best_ez_f1,
        "validation_min_class_f1": min(best_nez_f1, best_ez_f1),
        "validation_patient_macro_auprc_ez": mean_ez,
        "validation_patient_macro_auprc_nez": mean_nez,
        "validation_patient_ranking_oracle_macro_f1": float(np.mean(oracle_f1)) if oracle_f1 else 0.0,
        "validation_worst_center_auprc_hmean": min(center_hmeans) if center_hmeans else 0.0,
        "validation_balanced_accuracy": best_balanced_accuracy,
        "validation_selected_threshold": best_threshold,
    }


def _selection_key(summary: dict[str, float], epoch: int, objective: str) -> tuple[float, ...]:
    if objective == "ez_aware_f1":
        return (
            summary["validation_patient_macro_f1"], summary["validation_min_class_f1"],
            summary["validation_patient_ez_f1"], summary["validation_patient_macro_auprc_ez"],
            summary["validation_patient_ranking_oracle_macro_f1"], summary["validation_worst_center_auprc_hmean"],
            summary["validation_balanced_accuracy"], -float(epoch),
        )
    if objective == "f1":
        return (
            summary["validation_patient_macro_f1"],
            summary["validation_patient_nez_f1"],
            summary["balanced_patient_auprc_hmean"],
            summary["min_patient_macro_auprc"],
            -float(epoch),
        )
    return (
        summary["balanced_patient_auprc_hmean"], summary["min_patient_macro_auprc"],
        summary["patient_ranking_oracle_macro_f1"], summary["worst_center_auprc_hmean"], -float(epoch),
    )


def train_ranking_model(
    experiment: Any, fit_subjects: Sequence[str], validation_subjects: Sequence[str], *,
    fold_tag: str, max_epochs: int, select_checkpoint: bool,
) -> tuple[torch.nn.Module, int, float, list[dict[str, Any]], dict[str, Any], Any]:
    if experiment.device.type == "cuda":
        # Improve small dense matmul throughput on Ampere and newer GPUs.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    fit_ds, val_ds, _, normalizer = experiment._build_datasets(fit_subjects, validation_subjects, validation_subjects)
    # The formal ensemble seed is an explicit independent control. Fold tags
    # identify ledgers only and must not silently perturb model initialization.
    seed = int(getattr(experiment.args, "model_seed", 42))
    train_loader = _loader(experiment, fit_ds, train=True, seed=seed)
    val_loader = _loader(experiment, val_ds, train=False, seed=seed)
    torch.manual_seed(seed)
    model = experiment.runtime["model_cls"](experiment.args).to(experiment.device)
    experiment._dry_initialize_lazy_layers(model, train_loader)
    optimizer = _optimizer(model, experiment.args)
    unit_weight = torch.tensor(1.0, device=experiment.device)
    best_state = copy.deepcopy(model.state_dict())
    best_key: tuple[float, ...] | None = None
    best_epoch = 0
    best_threshold = float(getattr(experiment.args, "classification_threshold", 0.5))
    no_improve = 0
    history: list[dict[str, Any]] = []
    prototype_audit: dict[str, Any] = {}
    for epoch in range(1, int(max_epochs) + 1):
        experiment.current_epoch = epoch
        for group in optimizer.param_groups:
            group["lr"] = float(getattr(experiment.args, "cane_head_lr", 1e-4)) if group["group_name"] == "cane_heads" else (
                float(getattr(experiment.args, "cane_backbone_lr_stage1", 1e-4)) if epoch <= 5 else float(getattr(experiment.args, "cane_backbone_lr_after_warmup", 2e-5))
            )
        if hasattr(train_loader.batch_sampler, "set_epoch"):
            train_loader.batch_sampler.set_epoch(epoch)
        train_metrics = experiment._train_one_epoch(model, train_loader, optimizer, unit_weight)
        if epoch == 5:
            prototype_audit = initialize_clean_nez_prototypes(
                model, train_loader, experiment.device, int(getattr(experiment.args, "model_seed", 42)),
                max_samples=int(getattr(experiment.args, "cane_prototype_max_samples", 20000)),
            )
        val_records = collect_ranking_records(experiment, model, val_loader, source=fold_tag)
        val_summary = ranking_validation_summary(val_records)
        row = {"fold_tag": fold_tag, "epoch": epoch, **train_metrics, **{f"val_{k}": v for k, v in val_summary.items()}}
        history.append(row)
        experiment._log(
            f"[CANE-PATH] {fold_tag} epoch {epoch}/{max_epochs} | "
            f"train_loss={float(train_metrics.get('loss', 0.0)):.4f} | "
            f"val_macro_f1={val_summary['validation_patient_macro_f1']:.4f} | "
            f"val_nez_f1={val_summary['validation_patient_nez_f1']:.4f} | "
            f"val_threshold={val_summary['validation_selected_threshold']:.3f} | "
            f"val_balanced_auprc_hmean={val_summary['balanced_patient_auprc_hmean']:.4f}"
        )
        selection_objective = str(getattr(experiment.args, "cane_selection_objective", "auprc")).lower()
        key = _selection_key(val_summary, epoch, selection_objective)
        eligible = epoch >= 5
        min_delta = max(0.0, float(getattr(experiment.args, "cane_early_stop_min_delta", 0.0)))
        meaningful_improvement = (
            best_key is None
            or key[0] > best_key[0] + min_delta
        )
        if eligible and (not select_checkpoint or meaningful_improvement):
            best_key, best_epoch, best_threshold, best_state, no_improve = (
                key, epoch, float(val_summary["validation_selected_threshold"]),
                copy.deepcopy(model.state_dict()), 0,
            )
        else:
            no_improve += 1
        if select_checkpoint and epoch >= int(getattr(experiment.args, "min_epochs_before_early_stop", 18)) and no_improve >= int(getattr(experiment.args, "patience", 10)):
            experiment._log(
                f"[CANE-PATH] {fold_tag} early stop at epoch {epoch} | "
                f"patience={no_improve} | min_delta={min_delta:.4f}"
            )
            break
    if best_epoch < 5:
        raise RuntimeError("Ranking training ended before fit-only prototype initialization")
    model.load_state_dict(best_state)
    return model, best_epoch, best_threshold, history, prototype_audit, val_loader


def train_path_head(records: Sequence[dict[str, Any]], args: Any, device: torch.device) -> tuple[PatientAdaptiveThresholdHead, list[dict[str, float]]]:
    summaries = torch.as_tensor(np.stack([record["path_summary"] for record in records]), dtype=torch.float32, device=device)
    oracle_rows = [oracle_standardized_threshold(record["standardized_nez_logit"][record["channel_mask"]], record["labels_nez"][record["channel_mask"]].astype(int)) for record in records]
    targets = torch.as_tensor([row["oracle_standardized_threshold"] for row in oracle_rows], dtype=torch.float32, device=device)
    score_rows = [torch.as_tensor(record["standardized_nez_logit"][record["channel_mask"]], dtype=torch.float32, device=device) for record in records]
    label_rows = [torch.as_tensor(record["labels_nez"][record["channel_mask"]], dtype=torch.long, device=device) for record in records]
    rng = np.random.default_rng(int(getattr(args, "inner_split_seed", 42)))
    order = rng.permutation(len(records))
    n_val = max(1, int(round(0.20 * len(order))))
    val_idx, train_idx = order[:n_val], order[n_val:]
    torch.manual_seed(int(getattr(args, "model_seed", 42)))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(getattr(args, "model_seed", 42)))
    head = PatientAdaptiveThresholdHead(int(getattr(args, "path_hidden_dim", 16)), float(getattr(args, "path_max_abs_threshold", 2.5))).to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=float(getattr(args, "path_learning_rate", 1e-3)), weight_decay=float(getattr(args, "path_weight_decay", 1e-2)))
    best_state, best_loss, no_improve = copy.deepcopy(head.state_dict()), float("inf"), 0
    history: list[dict[str, float]] = []

    def validation_metrics(predicted_thresholds: torch.Tensor) -> dict[str, float]:
        macro_rows, ez_ap_rows = [], []
        center_rows: dict[str, list[float]] = {}
        for local_idx, record_idx in enumerate(val_idx):
            scores = score_rows[record_idx].detach().cpu().numpy()
            labels = label_rows[record_idx].detach().cpu().numpy().astype(int)
            threshold = float(predicted_thresholds[local_idx].detach().cpu())
            prediction = (scores >= threshold).astype(int)
            macro = float(f1_score(labels, prediction, average="macro", zero_division=0))
            macro_rows.append(macro)
            center_rows.setdefault(str(records[record_idx]["center"]), []).append(macro)
            if np.unique(labels).size > 1:
                ez_ap_rows.append(float(average_precision_score(1 - labels, -scores)))
        metrics = {
            "validation_legal_path_macro_f1": float(np.mean(macro_rows)) if macro_rows else 0.0,
            "validation_inner_oof_ez_auprc": float(np.mean(ez_ap_rows)) if ez_ap_rows else 0.0,
            "validation_n_patients": float(len(macro_rows)),
        }
        for center in ("hup", "lzu", "multicenter", "pediatric"):
            values = center_rows.get(center, [])
            metrics[f"validation_{center}_legal_path_macro_f1"] = float(np.mean(values)) if values else float("nan")
        return metrics

    for epoch in range(1, int(getattr(args, "path_epochs", 300)) + 1):
        head.train(); optimizer.zero_grad(set_to_none=True)
        predicted = head(summaries[train_idx])
        loss, parts = path_training_loss(
            predicted, targets[train_idx], [score_rows[i] for i in train_idx], [label_rows[i] for i in train_idx],
            threshold_weight=float(getattr(args, "path_threshold_loss_weight", 0.10)),
            soft_f1_weight=float(getattr(args, "path_soft_f1_weight", 0.20)),
            threshold_l2_weight=float(getattr(args, "path_threshold_l2_weight", 0.005)),
        )
        loss.backward(); optimizer.step()
        head.eval()
        with torch.no_grad():
            val_predicted = head(summaries[val_idx])
            val_loss, _ = path_training_loss(
                val_predicted, targets[val_idx], [score_rows[i] for i in val_idx], [label_rows[i] for i in val_idx],
                threshold_weight=float(getattr(args, "path_threshold_loss_weight", 0.10)),
                soft_f1_weight=float(getattr(args, "path_soft_f1_weight", 0.20)),
                threshold_l2_weight=float(getattr(args, "path_threshold_l2_weight", 0.005)),
            )
        history.append({
            "epoch": float(epoch), "train_loss": float(loss.detach().cpu()),
            "validation_loss": float(val_loss.detach().cpu()),
            **validation_metrics(val_predicted),
            **{key: float(value.detach().cpu()) for key, value in parts.items()},
        })
        if float(val_loss) < best_loss - 1e-8:
            best_loss, best_state, no_improve = float(val_loss), copy.deepcopy(head.state_dict()), 0
        else:
            no_improve += 1
        if no_improve >= int(getattr(args, "path_patience", 30)):
            break
    head.load_state_dict(best_state)
    return head, history


def apply_path_head(records: Sequence[dict[str, Any]], head: PatientAdaptiveThresholdHead, device: torch.device) -> list[dict[str, Any]]:
    result = []
    head.eval()
    with torch.no_grad():
        for record in records:
            summary = torch.as_tensor(record["path_summary"], dtype=torch.float32, device=device)[None]
            threshold = head(summary)
            standardized = torch.as_tensor(record["standardized_nez_logit"], dtype=torch.float32, device=device)[None]
            mask = torch.as_tensor(record["channel_mask"], dtype=torch.bool, device=device)[None]
            decision = apply_patient_adaptive_threshold(standardized, threshold, mask)
            enriched = dict(record)
            enriched.update({
                "predicted_patient_threshold": float(threshold.item()),
                "predicted_nez_mask": decision["predicted_nez_mask"][0].cpu().numpy(),
                "predicted_ez_mask": decision["predicted_ez_mask"][0].cpu().numpy(),
                "decision_probability_nez": decision["decision_probability_nez"][0].cpu().numpy(),
                "decision_rule": "patient_adaptive_standardized_nez_threshold",
                "threshold_source": "cross_fitted_patient_adaptive_head",
                "classification_threshold": float("nan"), "true_count_used_for_prediction": False,
                "oracle_threshold_used_for_prediction": False, "center_used_for_prediction": False,
                "positive_label": "nez",
            })
            result.append(enriched)
    return result


def _write_ledgers(records: Sequence[dict[str, Any]], output_dir: Path, fold: int, seed: int) -> None:
    patient_rows, channel_rows = [], []
    for record in records:
        subject_id = str(record.get("subject_id", "<unknown>"))
        channel_names = list(record["canonical_channels"])
        n_channels = len(channel_names)

        def channel_array(name: str, dtype: Any | None = None) -> np.ndarray:
            value = np.asarray(record[name], dtype=dtype)
            if value.ndim != 1 or value.shape[0] != n_channels:
                raise ValueError(
                    f"Ledger field {name!r} for {subject_id} must have shape "
                    f"({n_channels},), got {value.shape}"
                )
            return value

        valid = channel_array("channel_mask", bool)
        labels = channel_array("labels_nez", np.float32)
        labels_ez = channel_array("labels_ez", np.float32)
        predicted_nez_mask = channel_array("predicted_nez_mask", bool)
        predicted_ez_mask = channel_array("predicted_ez_mask", bool)
        if np.any(predicted_nez_mask & predicted_ez_mask) or not np.array_equal(
            predicted_nez_mask | predicted_ez_mask, valid
        ):
            raise ValueError(
                f"Ledger prediction masks for {subject_id} must be disjoint complements on valid channels"
            )

        vector_fields = (
            "direct_nez_logit", "final_nez_logit", "standardized_nez_logit",
            "direct_score_nez", "score_nez", "score_ez", "anchor_residual",
            "anchor_distance", "anchor_nez_evidence", "seizure_residual",
            "seizure_nez_probability_mean", "seizure_nez_probability_std",
            "seizure_nez_agreement", "causal_feature_valid", "causal_residual",
        )
        vectors = {name: channel_array(name) for name in vector_fields}
        causal_features = np.asarray(record["causal_propagation_features"], dtype=np.float32)
        if causal_features.shape != (n_channels, 6):
            raise ValueError(
                f"Ledger field 'causal_propagation_features' for {subject_id} must have "
                f"shape ({n_channels}, 6), got {causal_features.shape}"
            )
        patient_rows.append({
            "outer_fold": fold, "model_seed": seed, "subject_id": record["subject_id"], "center": record["center"],
            "n_channels": int(valid.sum()), "n_seizures": record["n_seizures"],
            "true_nez_count": int(((labels == 1) & valid).sum()), "true_ez_count": int(((labels == 0) & valid).sum()),
            "predicted_nez_count": int(predicted_nez_mask.sum()), "predicted_ez_count": int(predicted_ez_mask.sum()),
            "predicted_threshold": record["predicted_patient_threshold"], "decision_rule": record["decision_rule"],
            "true_count_used_for_prediction": False, "oracle_threshold_used_for_prediction": False,
        })
        for channel_id, channel_name in enumerate(channel_names):
            if not valid[channel_id]:
                continue
            cp = causal_features[channel_id]
            channel_rows.append({
                "outer_fold": fold, "inner_fold_source": record["inner_fold_source"], "model_seed": seed,
                "subject_id": record["subject_id"], "center": record["center"], "channel_name": channel_name, "channel_id": channel_id,
                "label_nez": int(labels[channel_id]), "label_ez": int(labels_ez[channel_id]), "valid": True,
                "direct_nez_logit": vectors["direct_nez_logit"][channel_id], "final_nez_logit": vectors["final_nez_logit"][channel_id],
                "standardized_nez_logit": vectors["standardized_nez_logit"][channel_id], "direct_score_nez": vectors["direct_score_nez"][channel_id],
                "final_score_nez": vectors["score_nez"][channel_id], "final_score_ez": vectors["score_ez"][channel_id],
                "anchor_residual": vectors["anchor_residual"][channel_id], "anchor_distance": vectors["anchor_distance"][channel_id],
                "anchor_nez_evidence": vectors["anchor_nez_evidence"][channel_id], "seizure_residual": vectors["seizure_residual"][channel_id],
                "seizure_nez_probability_mean": vectors["seizure_nez_probability_mean"][channel_id],
                "seizure_nez_probability_std": vectors["seizure_nez_probability_std"][channel_id], "seizure_nez_agreement": vectors["seizure_nez_agreement"][channel_id],
                **{name: float(cp[index]) for index, name in enumerate((
                    "cp_preictal_suppression_rank", "cp_suppression_release_shift", "cp_causal_early_activation_rank",
                    "cp_early_source_rank", "cp_propagation_persistence", "cp_cross_seizure_source_consistency"))},
                "cp_feature_valid": bool(vectors["causal_feature_valid"][channel_id]), "causal_residual": vectors["causal_residual"][channel_id],
                "predicted_patient_threshold": record["predicted_patient_threshold"], "predicted_nez": int(predicted_nez_mask[channel_id]),
                "predicted_ez": int(predicted_ez_mask[channel_id]), "decision_rule": record["decision_rule"],
                "true_count_used_for_prediction": False, "oracle_threshold_used_for_prediction": False,
            })
    pd.DataFrame(patient_rows).to_csv(output_dir / f"test_patient_predictions_neuroez_v2_fold_{fold}.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(output_dir / f"test_channel_predictions_neuroez_v2_fold_{fold}.csv", index=False)


def apply_fixed_nez_probability_threshold(
    records: Sequence[dict[str, Any]], threshold: float,
    threshold_source: str = "fixed_predeclared_probability_threshold",
) -> list[dict[str, Any]]:
    threshold = float(threshold)
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"Fixed NEZ probability threshold must be in [0,1], got {threshold}")
    result: list[dict[str, Any]] = []
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        score_nez = np.asarray(record["score_nez"], dtype=np.float32)
        if score_nez.shape != valid.shape:
            raise ValueError(
                f"score_nez/channel_mask shape mismatch for {record.get('subject_id', '<unknown>')}: "
                f"{score_nez.shape} versus {valid.shape}"
            )
        predicted_nez = (score_nez >= threshold) & valid
        enriched = dict(record)
        enriched.update({
            "predicted_nez_mask": predicted_nez,
            "predicted_ez_mask": (~predicted_nez) & valid,
            "predicted_patient_threshold": threshold,
            "classification_threshold": threshold,
            "decision_rule": "fixed_nez_probability_threshold",
            "threshold_source": str(threshold_source),
            "true_count_used_for_prediction": False,
            "oracle_threshold_used_for_prediction": False,
            "center_used_for_prediction": False,
            "positive_label": "nez",
        })
        result.append(enriched)
    return result


def run_cane_path_cp_outer_only(experiment: Any) -> list[dict[str, Any]]:
    """Run one leak-free fit/validation model per outer fold without PATH cross-fitting."""
    from exp_ez_hybrid import _summarize_prediction_records

    args = experiment.args
    output = Path(getattr(args, "output_dir", "outputs"))
    output.mkdir(parents=True, exist_ok=True)
    seed = int(getattr(args, "model_seed", 42))
    threshold = float(getattr(args, "classification_threshold", 0.5))
    run_signature = {
        "training_mode": "direct_outer_only",
        "model_seed": seed,
        "cane_profile": str(getattr(args, "cane_profile", "P1")),
        "batch_size": int(getattr(args, "batch_size", 2)),
        "patient_batch_size": int(getattr(args, "patient_batch_size", 2)),
        "center_balanced_batches": bool(getattr(args, "center_balanced_batches", False)),
        "classification_threshold": threshold,
        "epochs": int(getattr(args, "epochs", 45)),
        "patience": int(getattr(args, "patience", 5)),
        "early_stop_min_delta": float(getattr(args, "cane_early_stop_min_delta", 0.0)),
        "selection_objective": str(getattr(args, "cane_selection_objective", "auprc")),
    }
    splits = list(experiment.outer_splits)
    max_outer = int(getattr(args, "max_outer_folds", 0) or 0)
    if max_outer > 0:
        splits = splits[:max_outer]

    experiment._log(
        f"[CANE-PATH][OuterOnly] start | outer_folds={len(splits)} | "
        f"models_to_train={len(splits)} | selection_objective={run_signature['selection_objective']} | "
        "inner_crossfit=disabled | path_head=disabled"
    )
    all_test_records: list[dict[str, Any]] = []
    fold_summaries: list[dict[str, Any]] = []
    training_history: list[dict[str, Any]] = []
    prototype_rows: list[dict[str, Any]] = []

    for split in splits:
        outer_fold = int(split["fold_idx"])
        outer_train = list(split["train_subjects"])
        outer_test = list(split["test_subjects"])
        checkpoint_path = output / f"outer_only_fold_{outer_fold}_checkpoint.pkl"
        fold_outputs = (
            output / f"test_patient_predictions_neuroez_v2_fold_{outer_fold}.csv",
            output / f"test_channel_predictions_neuroez_v2_fold_{outer_fold}.csv",
            output / f"cane_path_fold_{outer_fold}_seed_{seed}_audit.json",
        )
        if checkpoint_path.is_file() and all(path.is_file() for path in fold_outputs):
            with checkpoint_path.open("rb") as handle:
                checkpoint = pickle.load(handle)
            if checkpoint.get("run_signature") != run_signature:
                raise RuntimeError(
                    f"Outer-only fold {outer_fold} checkpoint was created with different run parameters. "
                    "Use a new OutputRoot when changing batch size, threshold, seed, or training settings."
                )
            checkpoint_records = list(checkpoint.get("records", []))
            checkpoint_subjects = {str(record["subject_id"]) for record in checkpoint_records}
            if checkpoint_subjects != set(map(str, outer_test)) or len(checkpoint_records) != len(outer_test):
                raise RuntimeError(
                    f"Outer-only fold {outer_fold} checkpoint does not match the frozen test cohort"
                )
            all_test_records.extend(checkpoint_records)
            fold_summaries.append(dict(checkpoint["summary"]))
            training_history.extend(list(checkpoint.get("history", [])))
            prototype_rows.append(dict(checkpoint.get("prototype", {})))
            experiment._log(
                f"[CANE-PATH][OuterOnly] fold {outer_fold}/{len(splits)} restored from checkpoint"
            )
            continue
        if checkpoint_path.is_file():
            experiment._log(
                f"[CANE-PATH][OuterOnly] fold {outer_fold} checkpoint is incomplete; rerunning fold"
            )
        fit_subjects, validation_subjects = split_train_val_subjects(
            outer_train,
            val_ratio=float(getattr(args, "val_ratio", 0.2)),
            random_seed=int(getattr(args, "outer_split_seed", 42)),
            fold_idx=outer_fold,
        )
        if set(fit_subjects) & set(validation_subjects) or (
            set(fit_subjects) | set(validation_subjects)
        ) & set(outer_test):
            raise RuntimeError(f"Outer-only patient leakage detected in fold {outer_fold}")

        tag = f"outer{outer_fold}_direct"
        experiment._log(
            f"[CANE-PATH][OuterOnly] fold {outer_fold}/{len(splits)} start | "
            f"fit_patients={len(fit_subjects)} | val_patients={len(validation_subjects)} | "
            f"test_patients={len(outer_test)}"
        )
        model, best_epoch, selected_threshold, history, prototype_audit, _ = train_ranking_model(
            experiment,
            fit_subjects,
            validation_subjects,
            fold_tag=tag,
            max_epochs=int(getattr(args, "epochs", 45)),
            select_checkpoint=True,
        )
        training_history.extend(
            {"outer_fold": outer_fold, "model_seed": seed, **row} for row in history
        )
        prototype_rows.append({
            "outer_fold": outer_fold, "model_seed": seed, "best_epoch": best_epoch,
            **prototype_audit,
        })

        _, _, test_dataset, _ = experiment._build_datasets(
            fit_subjects, validation_subjects, outer_test
        )
        test_loader = _loader(experiment, test_dataset, train=False, seed=seed)
        frozen_test = collect_ranking_records(
            experiment, model, test_loader, source=f"outer{outer_fold}_test"
        )
        predicted = apply_fixed_nez_probability_threshold(
            frozen_test, selected_threshold, threshold_source="fold_validation_macro_f1"
        )
        summary, enriched = _summarize_prediction_records(
            predicted
        )
        summary.update({
            "outer_fold": outer_fold,
            "model_seed": seed,
            "best_epoch": best_epoch,
            "training_mode": "direct_outer_only",
            "inner_crossfit_used": False,
            "path_head_used": False,
            "validation_selected_threshold": selected_threshold,
        })
        fold_summaries.append(summary)
        all_test_records.extend(enriched)
        _write_ledgers(enriched, output, outer_fold, seed)
        audit = {
            "method": "N8F_CANE_PATH_CP_NEZ_80_DIRECT_OUTER_ONLY",
            "training_mode": "direct_outer_only",
            "outer_fold": outer_fold,
            "model_seed": seed,
            "fit_subjects": sorted(fit_subjects),
            "validation_subjects": sorted(validation_subjects),
            "test_subjects": sorted(outer_test),
            "fit_validation_overlap": sorted(set(fit_subjects) & set(validation_subjects)),
            "train_test_overlap": sorted(
                (set(fit_subjects) | set(validation_subjects)) & set(outer_test)
            ),
            "inner_crossfit_used": False,
            "path_head_used": False,
            "decision_rule": "fixed_nez_probability_threshold",
            "classification_threshold": selected_threshold,
            "true_count_used_for_prediction": False,
            "oracle_threshold_used_for_prediction": False,
            "best_epoch": best_epoch,
        }
        (output / f"cane_path_fold_{outer_fold}_seed_{seed}_audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8"
        )
        checkpoint = {
            # Records-only historical checkpoints remain readable. New checkpoints
            # also support explicit P2.1 warm starts without changing P2 inference.
            "model_state_dict": {
                key: value.detach().cpu() for key, value in model.state_dict().items()
            },
            "run_signature": run_signature,
            "records": enriched,
            "summary": summary,
            "history": [
                row for row in training_history if int(row.get("outer_fold", -1)) == outer_fold
            ],
            "prototype": prototype_rows[-1],
        }
        checkpoint_tmp = checkpoint_path.with_suffix(".tmp")
        with checkpoint_tmp.open("wb") as handle:
            pickle.dump(checkpoint, handle, protocol=pickle.HIGHEST_PROTOCOL)
        checkpoint_tmp.replace(checkpoint_path)
        experiment._log(
            f"[CANE-PATH][OuterOnly] fold {outer_fold}/{len(splits)} complete | "
            f"best_epoch={best_epoch}"
        )
        del model, test_loader, frozen_test, predicted, enriched
        gc.collect()
        if experiment.device.type == "cuda":
            torch.cuda.empty_cache()

    if not all_test_records:
        raise RuntimeError("Outer-only CANE-PATH produced no held-out test records")
    expected_subjects = {
        str(subject) for split in splits for subject in split["test_subjects"]
    }
    observed_subjects = {str(record["subject_id"]) for record in all_test_records}
    if observed_subjects != expected_subjects or len(all_test_records) != len(expected_subjects):
        raise RuntimeError(
            "Outer-only held-out ledger must contain exactly one row per expected test patient"
        )

    overall, _ = _summarize_prediction_records(all_test_records)
    fold_thresholds = [float(row["validation_selected_threshold"]) for row in fold_summaries]
    overall.update({
        "analysis_status": "posthoc_sensitivity_not_primary",
        "training_mode": "direct_outer_only",
        "inner_crossfit_used": False,
        "path_head_used": False,
        "decision_rule": "fixed_nez_probability_threshold",
        "classification_threshold": float(np.mean(fold_thresholds)),
        "fold_validation_thresholds": ",".join(f"{value:.6f}" for value in fold_thresholds),
        "threshold_source": "fold_validation_macro_f1",
        "selection_objective": run_signature["selection_objective"],
    })
    pd.DataFrame(training_history).to_csv(
        output / "ranking_train_history_by_fold_seed.csv", index=False
    )
    pd.DataFrame(training_history).to_csv(
        output / "ranking_validation_history_by_fold_seed.csv", index=False
    )
    pd.DataFrame(prototype_rows).to_csv(
        output / "prototype_initialization_by_fold_seed.csv", index=False
    )
    pd.DataFrame([overall]).to_csv(output / "heldout_summary_cane_path_cp.csv", index=False)
    (output / "heldout_summary_cane_path_cp.json").write_text(
        json.dumps(overall, indent=2, allow_nan=True), encoding="utf-8"
    )
    pd.DataFrame(fold_summaries).to_csv(
        output / "heldout_fold_summary_cane_path_cp.csv", index=False
    )
    return all_test_records


def run_cane_path_cp_nested(experiment: Any) -> list[dict[str, Any]]:
    from exp_ez_hybrid import _summarize_prediction_records

    args = experiment.args
    output = Path(getattr(args, "output_dir", "outputs")); output.mkdir(parents=True, exist_ok=True)
    all_test_records, inner_oof_records = [], []
    ranking_history, prototype_rows, path_history_rows, fold_summaries, inner_summary_rows = [], [], [], [], []
    seed = int(getattr(args, "model_seed", 42))
    splits = list(experiment.outer_splits)
    max_folds = int(getattr(args, "max_outer_folds", 0) or 0)
    if max_folds > 0:
        splits = splits[:max_folds]
    for split in splits:
        outer_fold = int(split["fold_idx"])
        outer_train, outer_test = list(split["train_subjects"]), list(split["test_subjects"])
        experiment._log(
            f"[CANE-PATH] outer {outer_fold}/{len(splits)} start | "
            f"train_patients={len(outer_train)} | test_patients={len(outer_test)}"
        )
        inner_splits = build_inner_crossfit_splits(outer_train, int(args.inner_splits), int(args.inner_split_seed))
        fold_oof, best_epochs = [], []
        for inner in inner_splits:
            tag = f"outer{outer_fold}_inner{inner['inner_fold']}"
            experiment._log(
                f"[CANE-PATH] {tag} start | fit_patients={len(inner['fit_subjects'])} | "
                f"heldout_patients={len(inner['heldout_subjects'])}"
            )
            model, best_epoch, _, history, proto_audit, heldout_loader = train_ranking_model(
                experiment, inner["fit_subjects"], inner["heldout_subjects"], fold_tag=tag,
                max_epochs=int(args.epochs), select_checkpoint=True,
            )
            inner_records = collect_ranking_records(experiment, model, heldout_loader, source=tag)
            for record in inner_records:
                record["outer_fold"] = outer_fold
                record["inner_fold"] = int(inner["inner_fold"])
                record["model_seed"] = seed
            fold_oof.extend(inner_records)
            best_epochs.append(best_epoch)
            ranking_history.extend({"outer_fold": outer_fold, "inner_fold": inner["inner_fold"], "model_seed": seed, **row} for row in history)
            prototype_rows.append({"outer_fold": outer_fold, "inner_fold": inner["inner_fold"], "model_seed": seed, **proto_audit})
            experiment._log(f"[CANE-PATH] {tag} complete | best_epoch={best_epoch}")
            del model, heldout_loader, inner_records
            gc.collect()
            if experiment.device.type == "cuda":
                torch.cuda.empty_cache()
        if {record["subject_id"] for record in fold_oof} != set(outer_train) or len(fold_oof) != len(outer_train):
            raise RuntimeError("PATH training source is not a complete one-row-per-patient inner OOF ledger")
        inner_oof_records.extend(fold_oof)
        path_head, path_history = train_path_head(fold_oof, args, experiment.device)
        path_history_rows.extend({"outer_fold": outer_fold, "model_seed": seed, **row} for row in path_history)
        best_path_row = min(path_history, key=lambda row: (row["validation_loss"], row["epoch"]))
        inner_summary_rows.append({
            "outer_fold": outer_fold, "model_seed": seed,
            **ranking_validation_summary(fold_oof),
            "legal_path_macro_f1": best_path_row["validation_legal_path_macro_f1"],
            "legal_path_ez_auprc": best_path_row["validation_inner_oof_ez_auprc"],
            "legal_path_validation_n_patients": best_path_row["validation_n_patients"],
            "legal_path_best_epoch": best_path_row["epoch"],
            **{
                f"{center}_legal_path_macro_f1": best_path_row[f"validation_{center}_legal_path_macro_f1"]
                for center in ("hup", "lzu", "multicenter", "pediatric")
            },
        })
        final_epochs = max(5, int(round(median(best_epochs))))
        final_model, _, _, final_history, final_proto, _ = train_ranking_model(
            experiment, outer_train, outer_train, fold_tag=f"outer{outer_fold}_final",
            max_epochs=final_epochs, select_checkpoint=False,
        )
        ranking_history.extend({"outer_fold": outer_fold, "inner_fold": 0, "model_seed": seed, **row} for row in final_history)
        prototype_rows.append({"outer_fold": outer_fold, "inner_fold": 0, "model_seed": seed, **final_proto})
        _, _, test_ds, _ = experiment._build_datasets(outer_train, outer_train, outer_test)
        test_loader = _loader(experiment, test_ds, train=False, seed=seed)
        frozen_test = collect_ranking_records(experiment, final_model, test_loader, source=f"outer{outer_fold}_test")
        predicted = apply_path_head(frozen_test, path_head, experiment.device)
        summary, enriched = _summarize_prediction_records(predicted)
        summary.update({"outer_fold": outer_fold, "model_seed": seed, "final_outer_epochs": final_epochs})
        fold_summaries.append(summary); all_test_records.extend(enriched)
        _write_ledgers(enriched, output, outer_fold, seed)
        del final_model, test_loader, frozen_test, predicted, enriched
        gc.collect()
        if experiment.device.type == "cuda":
            torch.cuda.empty_cache()
        audit = {
            "method": "N8F_CANE_PATH_CP_NEZ_80", "outer_fold": outer_fold, "model_seed": seed,
            "outer_split_seed": int(args.outer_split_seed), "inner_split_seed": int(args.inner_split_seed),
            "outer_train_subjects": sorted(outer_train), "outer_test_subjects": sorted(outer_test),
            "outer_train_test_overlap": sorted(set(outer_train) & set(outer_test)),
            "inner_oof_subjects": sorted(record["subject_id"] for record in fold_oof),
            "inner_oof_complete": True, "path_training_source": "outer_train_inner_oof_only",
            "final_outer_epochs": final_epochs, "inner_best_epochs": best_epochs,
            "prototype_used_validation": False, "prototype_used_test": False,
            "test_evaluations_before_model_selection": 0, "test_used_for_threshold_training": False,
            "center_used_as_model_input": False, "true_count_used_for_prediction": False,
            "oracle_threshold_used_for_prediction": False,
        }
        (output / f"cane_path_fold_{outer_fold}_seed_{seed}_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True), encoding="utf-8")
    pd.DataFrame(ranking_history).to_csv(output / "ranking_train_history_by_fold_seed.csv", index=False)
    pd.DataFrame(ranking_history).to_csv(output / "ranking_validation_history_by_fold_seed.csv", index=False)
    pd.DataFrame(prototype_rows).to_csv(output / "prototype_initialization_by_fold_seed.csv", index=False)
    (output / "cane_prototype_initialization_audit.json").write_text(
        json.dumps(prototype_rows, indent=2, sort_keys=True), encoding="utf-8"
    )
    pd.DataFrame(path_history_rows).to_csv(output / "path_training_history.csv", index=False)
    pd.DataFrame(inner_summary_rows).to_csv(output / "inner_oof_summary_by_outer_fold.csv", index=False)
    oof_patient_rows = []
    oof_channel_rows = []
    for record in inner_oof_records:
        valid = record["channel_mask"]
        oracle = oracle_standardized_threshold(record["standardized_nez_logit"][valid], record["labels_nez"][valid].astype(int))
        oof_patient_rows.append({
            "outer_fold": record["outer_fold"], "inner_fold": record["inner_fold"],
            "model_seed": record["model_seed"], "subject_id": record["subject_id"],
            "center": record["center"], "inner_fold_source": record["inner_fold_source"],
            **oracle, **dict(zip(PATH_SUMMARY_FIELDS, record["path_summary"])),
        })
        for idx, channel in enumerate(record["canonical_channels"]):
            if valid[idx]:
                oof_channel_rows.append({
                    "outer_fold": record["outer_fold"], "inner_fold": record["inner_fold"],
                    "model_seed": record["model_seed"], "subject_id": record["subject_id"],
                    "channel_name": channel, "inner_fold_source": record["inner_fold_source"],
                    "label_nez": int(record["labels_nez"][idx]),
                    "final_nez_logit": record["final_nez_logit"][idx],
                    "standardized_nez_logit": record["standardized_nez_logit"][idx],
                })
    pd.DataFrame(oof_patient_rows).to_csv(output / "inner_oof_patient_threshold_targets.csv", index=False)
    pd.DataFrame(oof_channel_rows).to_csv(output / "inner_oof_channel_predictions.csv", index=False)
    overall, _ = _summarize_prediction_records(all_test_records)
    overall.update({"analysis_status": "posthoc_sensitivity_not_primary", "decision_rule": "patient_adaptive_standardized_nez_threshold", "classification_threshold": float("nan"), "threshold_source": "cross_fitted_patient_adaptive_head"})
    pd.DataFrame([overall]).to_csv(output / "heldout_summary_cane_path_cp.csv", index=False)
    (output / "heldout_summary_cane_path_cp.json").write_text(json.dumps(overall, indent=2, allow_nan=True), encoding="utf-8")
    pd.DataFrame(fold_summaries).to_csv(output / "heldout_fold_summary_cane_path_cp.csv", index=False)
    return all_test_records


__all__ = [
    "apply_fixed_nez_probability_threshold", "apply_path_head", "build_inner_crossfit_splits",
    "collect_ranking_records", "ranking_validation_summary", "run_cane_path_cp_nested",
    "run_cane_path_cp_outer_only", "train_path_head",
]
