"""P2.1 training orchestration with checkpoint/threshold separation."""

from __future__ import annotations

import copy
import gc
import json
import pickle
import shutil
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.nn.parameter import UninitializedParameter
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from data_factory import split_train_val_subjects

from .cane_path_cp_heads import initialize_clean_nez_prototypes
from .cane_path_cp_trainer import (
    _loader, _move, apply_fixed_nez_probability_threshold,
    build_inner_crossfit_splits, collect_ranking_records,
)
from .cane_path_threshold import oracle_standardized_threshold
from .p21_profiles import get_p21_profile
from .p21_ranking_loss import compute_observed_ez_reliability


def configure_p21_training_stage(model: torch.nn.Module, epoch: int, args: Any) -> str:
    stage1_end = int(getattr(args, "p21_stage1_end", 8))
    stage2_end = int(getattr(args, "p21_stage2_end", 20))
    stage = "stage1" if int(epoch) <= stage1_end else "stage2" if int(epoch) <= stage2_end else "stage3"
    head_tokens = (
        "p21_selective_fusion", "p21_evidence", "clean_nez_anchor",
        "multi_seizure_evidence", "causal_propagation_residual",
    )
    for name, parameter in model.named_parameters():
        trainable = any(token in name for token in head_tokens)
        if stage in {"stage2", "stage3"} and any(token in name for token in ("channel_classifier", "temporal_encoder", "physics_gate")):
            trainable = True
        if stage == "stage3" and bool(getattr(args, "p21_unfreeze_b0_stage3", False)) and "b0_encoder" in name:
            trainable = True
        parameter.requires_grad_(trainable)
    return stage


def _optimizer(model: torch.nn.Module, args: Any) -> torch.optim.Optimizer:
    head_tokens = ("p21_", "clean_nez_anchor", "multi_seizure_evidence", "causal_propagation_residual")
    heads, backbone = [], []
    for name, parameter in model.named_parameters():
        (heads if any(token in name for token in head_tokens) else backbone).append(parameter)
    return torch.optim.AdamW([
        {"params": heads, "lr": float(getattr(args, "p21_head_lr", 5e-5)), "group_name": "p21_heads"},
        {"params": backbone, "lr": float(getattr(args, "p21_backbone_lr", 1e-5)), "group_name": "p21_backbone"},
    ], weight_decay=float(getattr(args, "weight_decay", 1e-3)))


def _warm_start(model: torch.nn.Module, args: Any, outer_fold: int) -> dict[str, Any]:
    mode = str(getattr(args, "p21_init_mode", "from_current_p2_checkpoint"))
    if mode == "from_scratch":
        return {"init_mode": mode, "loaded": False, "reason": "explicit_from_scratch"}
    root = Path(getattr(args, "current_p2_checkpoint_root", ""))
    candidates = sorted(root.rglob(f"outer_only_fold_{outer_fold}_checkpoint.pkl")) if root.is_dir() else []
    if not candidates:
        raise FileNotFoundError(f"P2.1 warm start cannot find outer fold {outer_fold} checkpoint under {root}")
    with candidates[0].open("rb") as handle:
        payload = pickle.load(handle)
    state = payload.get("model_state_dict") if isinstance(payload, dict) else None
    if not isinstance(state, dict):
        raise ValueError(
            "Current P2 checkpoint has no model_state_dict. Re-run that P2 fold with the updated checkpoint writer "
            "or use --p21-init-mode from_scratch; records-only checkpoints cannot warm-start a model."
        )
    target_state = model.state_dict()
    compatible = {
        key: value for key, value in state.items()
        if key in target_state and (
            isinstance(target_state[key], UninitializedParameter)
            or target_state[key].shape == value.shape
        )
    }
    result = model.load_state_dict(compatible, strict=False)
    return {
        "init_mode": mode, "loaded": True, "checkpoint": str(candidates[0]),
        "compatible_tensor_count": len(compatible),
        "missing_keys": list(result.missing_keys), "unexpected_keys": list(result.unexpected_keys),
    }


def _ranking_metrics(records: Sequence[dict[str, Any]], checkpoint_threshold: float = 0.5) -> dict[str, float]:
    nez_ap, ez_ap, oracle, ez_mrr = [], [], [], []
    macro_f1, nez_f1, ez_f1, balanced = [], [], [], []
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels = np.asarray(record["labels_nez"])[valid].astype(int)
        score = np.asarray(record["score_nez"])[valid]
        prediction = (score >= float(checkpoint_threshold)).astype(int)
        macro_f1.append(float(f1_score(labels, prediction, average="macro", labels=[0, 1], zero_division=0)))
        nez_f1.append(float(f1_score(labels, prediction, pos_label=1, zero_division=0)))
        ez_f1.append(float(f1_score(labels, prediction, pos_label=0, zero_division=0)))
        recalls = [np.mean(prediction[labels == value] == value) for value in (0, 1) if np.any(labels == value)]
        balanced.append(float(np.mean(recalls)) if recalls else 0.0)
        if np.unique(labels).size == 2:
            nez_ap.append(average_precision_score(labels, score))
            ez_ap.append(average_precision_score(1 - labels, 1 - score))
        oracle.append(oracle_standardized_threshold(np.asarray(record["standardized_nez_logit"])[valid], labels)["oracle_macro_f1"])
        order = np.argsort(score, kind="mergesort")
        positions = np.flatnonzero(labels[order] == 0)
        ez_mrr.append(1.0 / (int(positions[0]) + 1) if positions.size else 0.0)
    mean_nez = float(np.mean(nez_ap)) if nez_ap else 0.0
    mean_ez = float(np.mean(ez_ap)) if ez_ap else 0.0
    return {
        "checkpoint_patient_macro_f1": float(np.mean(macro_f1)) if macro_f1 else 0.0,
        "checkpoint_patient_nez_f1": float(np.mean(nez_f1)) if nez_f1 else 0.0,
        "checkpoint_patient_ez_f1": float(np.mean(ez_f1)) if ez_f1 else 0.0,
        "checkpoint_balanced_accuracy": float(np.mean(balanced)) if balanced else 0.0,
        "balanced_patient_auprc_hmean": 2 * mean_nez * mean_ez / max(mean_nez + mean_ez, 1e-12),
        "patient_macro_auprc_nez": mean_nez,
        "patient_macro_auprc_ez": mean_ez,
        "patient_oracle_macro_f1": float(np.mean(oracle)) if oracle else 0.0,
        "patient_macro_ez_mrr": float(np.mean(ez_mrr)) if ez_mrr else 0.0,
    }


def _checkpoint_key(metrics: dict[str, float], epoch: int, objective: str = "f1") -> tuple[float, ...]:
    if str(objective).lower() == "f1":
        return (
            metrics["checkpoint_patient_macro_f1"], metrics["checkpoint_patient_nez_f1"],
            metrics["checkpoint_patient_ez_f1"], metrics["balanced_patient_auprc_hmean"],
            metrics["patient_macro_ez_mrr"], -float(epoch),
        )
    return (
        metrics["balanced_patient_auprc_hmean"], metrics["patient_macro_auprc_ez"],
        metrics["patient_macro_ez_mrr"], -float(epoch),
    )


def _threshold(records: Sequence[dict[str, Any]]) -> tuple[float, dict[str, float]]:
    scores = np.concatenate([np.asarray(row["score_nez"])[np.asarray(row["channel_mask"], dtype=bool)] for row in records])
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], scores)))
    evaluated = []
    for threshold in candidates:
        macro, nez, ez, balanced = [], [], [], []
        for row in records:
            valid = np.asarray(row["channel_mask"], dtype=bool)
            labels = np.asarray(row["labels_nez"])[valid].astype(int)
            prediction = (np.asarray(row["score_nez"])[valid] >= threshold).astype(int)
            macro.append(f1_score(labels, prediction, average="macro", labels=[0, 1], zero_division=0))
            nez.append(f1_score(labels, prediction, pos_label=1, zero_division=0))
            ez.append(f1_score(labels, prediction, pos_label=0, zero_division=0))
            recalls = [np.mean(prediction[labels == value] == value) for value in (0, 1) if np.any(labels == value)]
            balanced.append(float(np.mean(recalls)))
        evaluated.append((float(np.mean(macro)), float(min(np.mean(nez), np.mean(ez))), float(np.mean(balanced)), -abs(float(threshold) - 0.5), -float(threshold), float(threshold)))
    best = max(evaluated)
    return best[-1], {"patient_macro_f1": best[0], "min_class_f1": best[1], "balanced_accuracy": best[2]}


def _train_model(
    experiment: Any,
    fit_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    *, outer_fold: int, fold_tag: str, max_epochs: int, select_checkpoint: bool,
) -> tuple[torch.nn.Module, int, list[dict[str, Any]], dict[str, Any]]:
    mode = str(getattr(experiment.args, "v3_anchor_mode", "precomputed_oof_screening"))
    contexts = None
    if mode == "nested_fold_safe":
        contexts = {name: (outer_fold, "inner_oof") for name in ("fit", "val", "test")}
    fit_ds, val_ds, _, _ = experiment._build_datasets(
        fit_subjects, validation_subjects, validation_subjects, v3_contexts=contexts
    )
    seed = int(getattr(experiment.args, "model_seed", 42))
    train_loader = _loader(experiment, fit_ds, train=True, seed=seed)
    val_loader = _loader(experiment, val_ds, train=False, seed=seed)
    torch.manual_seed(seed)
    model = experiment.runtime["model_cls"](experiment.args).to(experiment.device)
    warm = _warm_start(model, experiment.args, outer_fold)
    prototype = initialize_clean_nez_prototypes(model, train_loader, experiment.device, seed)
    optimizer = _optimizer(model, experiment.args)
    best_state, best_key, best_epoch, history, no_improve = None, None, 0, [], 0
    for epoch in range(1, int(max_epochs) + 1):
        experiment.current_epoch = epoch
        stage = configure_p21_training_stage(model, epoch, experiment.args)
        sampler = getattr(train_loader, "batch_sampler", None)
        if hasattr(sampler, "set_epoch"):
            sampler.set_epoch(epoch)
        model.train(); losses = []; batch_diagnostics: list[dict[str, Any]] = []
        for batch in train_loader:
            device_batch = _move(batch, experiment.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(device_batch)
            loss, loss_diagnostics = experiment._compute_loss(outputs, device_batch, torch.tensor(1.0, device=experiment.device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(getattr(experiment.args, "grad_clip", 1.0)))
            optimizer.step(); losses.append(float(loss.detach().cpu())); batch_diagnostics.append(loss_diagnostics)
        validation = collect_ranking_records(experiment, model, val_loader, source=f"{fold_tag}_validation")
        checkpoint_threshold = float(getattr(experiment.args, "p21_checkpoint_threshold", 0.5))
        metrics = _ranking_metrics(validation, checkpoint_threshold)
        key = _checkpoint_key(metrics, epoch, getattr(experiment.args, "p21_checkpoint_objective", "f1"))
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch, best_state, no_improve = key, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        mean_loss_diagnostics = {
            key: float(np.mean([row[key] for row in batch_diagnostics if isinstance(row.get(key), (int, float))]))
            for key in sorted({key for row in batch_diagnostics for key in row})
            if any(isinstance(row.get(key), (int, float)) for row in batch_diagnostics)
        }
        history.append({"epoch": epoch, "stage": stage, "train_loss": float(np.mean(losses)), **metrics, **mean_loss_diagnostics, "improved": improved})
        experiment._log(
            f"[P2.1] {fold_tag} epoch {epoch}/{max_epochs} | train_loss={np.mean(losses):.4f} | "
            f"val_balanced_auprc_hmean={metrics['balanced_patient_auprc_hmean']:.4f} | "
            f"val_checkpoint_macro_f1={metrics['checkpoint_patient_macro_f1']:.4f} | "
            f"val_ez_auprc={metrics['patient_macro_auprc_ez']:.4f}"
        )
        if select_checkpoint and epoch >= int(getattr(experiment.args, "min_epochs_before_early_stop", 20)) and no_improve >= int(getattr(experiment.args, "patience", 8)):
            break
    if best_state is None:
        raise RuntimeError("P2.1 training produced no checkpoint")
    model.load_state_dict(best_state)
    return model, best_epoch, history, {"warm_start": warm, "prototype": prototype}


def _collect_for_subjects(experiment: Any, model: torch.nn.Module, fit: Sequence[str], subjects: Sequence[str], *, outer_fold: int, split_role: str, source: str):
    mode = str(getattr(experiment.args, "v3_anchor_mode", "precomputed_oof_screening"))
    contexts = None
    if mode == "nested_fold_safe":
        contexts = {"fit": (outer_fold, "inner_oof"), "val": (outer_fold, split_role), "test": (outer_fold, split_role)}
    _, _, dataset, _ = experiment._build_datasets(fit, subjects, subjects, v3_contexts=contexts)
    loader = _loader(experiment, dataset, train=False, seed=int(getattr(experiment.args, "model_seed", 42)))
    return collect_ranking_records(experiment, model, loader, source=source)


def _flatten_channels(records: Sequence[dict[str, Any]], outer_fold: int) -> pd.DataFrame:
    rows = []
    arrays = (
        "v3_score_nez", "v3_logit_nez", "v3_anchor_standardized", "v3_anchor_valid",
        "direct_nez_logit", "direct_score_nez", "direct_standardized", "anchor_nez_evidence",
        "u_anchor", "seizure_nez_probability_mean", "u_seizure", "raw_u_causal", "u_causal",
        "q_cp", "q_coverage", "q_seizure", "q_stability", "w_noop", "w_anchor", "w_seizure",
        "w_causal", "fusion_entropy", "delta", "final_nez_logit", "score_nez", "score_ez",
        "final_score_nez", "final_score_ez", "ez_reliability", "reliability_component_count",
        "seizure_evidence_valid",
        "correction_saturation", "cp_early_source_rank", "causal_feature_valid",
    )
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        for index, channel in enumerate(record["canonical_channels"]):
            if not valid[index]:
                continue
            row = {
                "outer_fold": outer_fold, "subject_id": record["subject_id"], "center": record["center"],
                "channel_id": index, "channel_name": channel, "label_nez": int(record["labels_nez"][index]),
                "predicted_threshold": float(record["predicted_patient_threshold"]),
                "predicted_nez": int(record["predicted_nez_mask"][index]),
                "predicted_ez": int(record["predicted_ez_mask"][index]),
                "threshold_source": record["threshold_source"], "true_count_used": False,
                "analysis_status": record.get("analysis_status", ""),
            }
            for name in arrays:
                if name in record and index < len(np.asarray(record[name])):
                    row[name] = float(np.asarray(record[name])[index])
            rows.append(row)
    return pd.DataFrame(rows)


def _summaries(records: Sequence[dict[str, Any]], group_field: str | None = None) -> pd.DataFrame:
    from exp_ez_hybrid import _summarize_prediction_records
    groups = [("overall", records)] if group_field is None else []
    if group_field:
        values = sorted({record[group_field] for record in records})
        groups = [(value, [record for record in records if record[group_field] == value]) for value in values]
    rows = []
    for key, selected in groups:
        summary, _ = _summarize_prediction_records(selected)
        rows.append(({group_field: key} if group_field else {}) | summary)
    return pd.DataFrame(rows)


def _attach_reliability(records: Sequence[dict[str, Any]]) -> None:
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels = torch.as_tensor(np.asarray(record["labels_nez"])[None, :], dtype=torch.float32)
        mask = torch.as_tensor(valid[None, :])
        # Older P2.1 collectors did not persist this diagnostic. Treat it as
        # unavailable instead of failing after a completed held-out inference.
        seizure_available = np.asarray(record.get("seizure_evidence_valid", valid), dtype=bool)
        causal_available = np.asarray(record.get("causal_feature_valid", np.zeros_like(valid)), dtype=bool)
        result = compute_observed_ez_reliability(
            labels, mask,
            torch.as_tensor(np.asarray(record["direct_score_nez"])[None, :]),
            torch.as_tensor(np.asarray(record["anchor_nez_evidence"])[None, :]),
            torch.as_tensor(np.asarray(record["seizure_nez_probability_mean"])[None, :]),
            torch.as_tensor(np.asarray(record["cp_early_source_rank"])[None, :]),
            seizure_available=torch.as_tensor(seizure_available[None, :]),
            causal_available=torch.as_tensor(causal_available[None, :]),
        )
        for key, value in result.items():
            record[key] = value[0].cpu().numpy()


def _bootstrap_summary(records: Sequence[dict[str, Any]], seed: int, n_bootstrap: int = 1000) -> pd.DataFrame:
    rng = np.random.default_rng(int(seed))
    rows = []
    for replicate in range(int(n_bootstrap)):
        selected = [records[index] for index in rng.integers(0, len(records), size=len(records))]
        summary = _summaries(selected).iloc[0]
        rows.append({"replicate": replicate, **summary.to_dict()})
    frame = pd.DataFrame(rows)
    metrics = [name for name in ("patient_macro_f1", "patient_macro_nez_f1", "patient_macro_ez_f1", "patient_macro_auprc_nez", "patient_macro_auprc_ez", "patient_macro_ez_mrr") if name in frame]
    return pd.DataFrame([{
        "metric": name, "mean": float(frame[name].mean()),
        "ci_lower_2p5": float(frame[name].quantile(0.025)),
        "ci_upper_97p5": float(frame[name].quantile(0.975)), "n_bootstrap": int(n_bootstrap),
    } for name in metrics])


def _counterfactual_ablation(records: Sequence[dict[str, Any]], max_correction: float) -> pd.DataFrame:
    variants = ("C0_FULL", "C1_NO_CORRECTION", "C2_NO_ANCHOR_EVIDENCE", "C3_NO_SEIZURE_EVIDENCE", "C4_NO_CAUSAL_EVIDENCE", "C5_EQUAL_EVIDENCE_GATE")
    rows = []
    for variant in variants:
        altered = copy.deepcopy(list(records))
        for record in altered:
            valid = np.asarray(record["channel_mask"], dtype=bool)
            base = np.asarray(record["v3_anchor_standardized" if np.asarray(record.get("v3_anchor_valid", [])).any() else "direct_standardized"], dtype=float)
            anchor = np.asarray(record["u_anchor"], dtype=float); seizure = np.asarray(record["u_seizure"], dtype=float); causal = np.asarray(record["u_causal"], dtype=float)
            wa = np.asarray(record["w_anchor"], dtype=float); ws = np.asarray(record["w_seizure"], dtype=float); wc = np.asarray(record["w_causal"], dtype=float)
            if variant == "C0_FULL": delta = np.asarray(record["delta"], dtype=float)
            elif variant == "C1_NO_CORRECTION": delta = np.zeros_like(base)
            elif variant == "C2_NO_ANCHOR_EVIDENCE": delta = max_correction * (ws * seizure + wc * causal)
            elif variant == "C3_NO_SEIZURE_EVIDENCE": delta = max_correction * (wa * anchor + wc * causal)
            elif variant == "C4_NO_CAUSAL_EVIDENCE": delta = max_correction * (wa * anchor + ws * seizure)
            else:
                available = np.stack((np.ones_like(valid), np.asarray(record["seizure_evidence_valid"], dtype=bool), np.asarray(record["causal_feature_valid"], dtype=bool)))
                evidence = np.stack((anchor, seizure, causal))
                delta = max_correction * (evidence * available).sum(axis=0) / available.sum(axis=0).clip(min=1)
            score = 1.0 / (1.0 + np.exp(-np.clip(base + delta, -30.0, 30.0)))
            score[~valid] = 0.0
            record["score_nez"] = score; record["score_ez"] = np.where(valid, 1.0 - score, 0.0)
            threshold = float(record["predicted_patient_threshold"])
            record["predicted_nez_mask"] = (score >= threshold) & valid
            record["predicted_ez_mask"] = (score < threshold) & valid
        summary = _summaries(altered).iloc[0].to_dict()
        rows.append({"variant": variant, "analysis_status": "COUNTERFACTUAL_INFERENCE_ABLATION", "not_retrained": True, **summary})
    return pd.DataFrame(rows)


def run_p21_v3_asrr(experiment: Any) -> list[dict[str, Any]]:
    args = experiment.args
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    profile = get_p21_profile(args.p21_profile)
    if profile.legacy_p2:
        from .cane_path_cp_trainer import run_cane_path_cp_outer_only

        # R0 is an actual historical P2 execution, not merely the same forward.
        args.cane_direct_outer_only = True
        args.cane_selection_objective = "f1"
        records = run_cane_path_cp_outer_only(experiment)
        aliases = {
            "heldout_summary_cane_path_cp.csv": "p21_overall_summary.csv",
            "heldout_fold_summary_cane_path_cp.csv": "p21_fold_summary.csv",
            "ranking_train_history_by_fold_seed.csv": "p21_train_history.csv",
            "ranking_validation_history_by_fold_seed.csv": "p21_validation_history.csv",
        }
        for source_name, destination_name in aliases.items():
            source = output / source_name
            if source.is_file():
                shutil.copyfile(source, output / destination_name)
        for pattern, destination in (
            ("test_channel_predictions_neuroez_v2_fold_*.csv", "p21_channel_predictions.csv"),
            ("test_patient_predictions_neuroez_v2_fold_*.csv", "p21_patient_predictions.csv"),
        ):
            frames = [pd.read_csv(path) for path in sorted(output.glob(pattern))]
            if frames:
                pd.concat(frames, ignore_index=True).to_csv(output / destination, index=False)
        protocol = {
            "method": "P21_V3_ASRR_NEZ_80", "profile": profile.name,
            "delegated_historical_trainer": "run_cane_path_cp_outer_only",
            "forward_loss_threshold_checkpoint_legacy_p2": True,
            "positive_label": "nez", "score_semantics": "P(NEZ)",
            "true_count_used": False, "patient_oracle_used_for_prediction": False,
        }
        (output / "p21_protocol_audit.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        (output / "run_args_p21.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
        return records
    mode = str(args.v3_anchor_mode)
    analysis_status = "PRECOMPUTED_GLOBAL_OOF_SCREENING_ONLY" if mode == "precomputed_oof_screening" else "STRICT_NESTED_FOLD_SAFE"
    all_records, histories, fold_rows, stage_rows = [], [], [], []
    splits = list(experiment.outer_splits)
    if int(getattr(args, "max_outer_folds", 0) or 0) > 0:
        splits = splits[: int(args.max_outer_folds)]
    for split in splits:
        outer_fold = int(split["fold_idx"]); outer_train = list(split["train_subjects"]); outer_test = list(split["test_subjects"])
        if mode == "nested_fold_safe":
            oof_records, best_epochs = [], []
            for inner in build_inner_crossfit_splits(outer_train, int(args.inner_splits), int(args.inner_split_seed)):
                inner_fit, inner_val = split_train_val_subjects(inner["fit_subjects"], val_ratio=float(args.val_ratio), random_seed=int(args.inner_split_seed), fold_idx=int(inner["inner_fold"]))
                model, best_epoch, history, audit = _train_model(experiment, inner_fit, inner_val, outer_fold=outer_fold, fold_tag=f"outer{outer_fold}_inner{inner['inner_fold']}", max_epochs=int(args.epochs), select_checkpoint=True)
                heldout = _collect_for_subjects(experiment, model, inner["fit_subjects"], inner["heldout_subjects"], outer_fold=outer_fold, split_role="inner_oof", source=f"outer{outer_fold}_inner{inner['inner_fold']}_oof")
                oof_records.extend(heldout); best_epochs.append(best_epoch); histories.extend({"outer_fold": outer_fold, "inner_fold": inner["inner_fold"], **row} for row in history)
                del model; gc.collect()
            if {row["subject_id"] for row in oof_records} != set(map(str, outer_train)):
                raise RuntimeError("P2.1 strict inner OOF union does not exactly cover outer train")
            threshold, threshold_diag = _threshold(oof_records)
            final_epochs = max(1, int(round(median(best_epochs))))
            model, best_epoch, history, init_audit = _train_model(experiment, outer_train, outer_train, outer_fold=outer_fold, fold_tag=f"outer{outer_fold}_final", max_epochs=final_epochs, select_checkpoint=False)
            test_records = _collect_for_subjects(experiment, model, outer_train, outer_test, outer_fold=outer_fold, split_role="outer_test", source=f"outer{outer_fold}_test")
            threshold_source = "outer_train_inner_oof_macro_f1"
        else:
            fit, val = split_train_val_subjects(outer_train, val_ratio=float(args.val_ratio), random_seed=int(args.outer_split_seed), fold_idx=outer_fold)
            model, best_epoch, history, init_audit = _train_model(experiment, fit, val, outer_fold=outer_fold, fold_tag=f"outer{outer_fold}_screen", max_epochs=int(args.epochs), select_checkpoint=True)
            validation = _collect_for_subjects(experiment, model, fit, val, outer_fold=outer_fold, split_role="inner_oof", source=f"outer{outer_fold}_validation")
            threshold, threshold_diag = _threshold(validation)
            test_records = _collect_for_subjects(experiment, model, fit, outer_test, outer_fold=outer_fold, split_role="outer_test", source=f"outer{outer_fold}_test")
            threshold_source = "fold_validation_macro_f1"
            histories.extend({"outer_fold": outer_fold, **row} for row in history)
        predicted = apply_fixed_nez_probability_threshold(test_records, threshold, threshold_source=threshold_source)
        for record in predicted:
            record["outer_fold"] = outer_fold; record["analysis_status"] = analysis_status
        all_records.extend(predicted)
        ranking = _ranking_metrics(predicted, float(getattr(args, "p21_checkpoint_threshold", 0.5)))
        fold_rows.append({"outer_fold": outer_fold, "best_epoch": best_epoch, "selected_threshold": threshold, "threshold_source": threshold_source, **threshold_diag, **ranking})
        stage_rows.append({"outer_fold": outer_fold, "init_audit": json.dumps(init_audit, default=str), "analysis_status": analysis_status})
        torch.save({"model_state_dict": model.state_dict(), "outer_fold": outer_fold, "profile": profile.name, "analysis_status": analysis_status}, output / f"p21_best_model_fold_{outer_fold}.pt")
        _flatten_channels(predicted, outer_fold).to_csv(output / f"p21_channel_predictions_fold_{outer_fold}.csv", index=False)
        del model; gc.collect()
        if experiment.device.type == "cuda": torch.cuda.empty_cache()
    if not all_records:
        raise RuntimeError("P2.1 produced no held-out records")
    _attach_reliability(all_records)
    pd.concat([_flatten_channels([record], int(record["outer_fold"])) for record in all_records], ignore_index=True).to_csv(output / "p21_channel_predictions.csv", index=False)
    patient_rows = []
    from exp_ez_hybrid import _summarize_prediction_records
    for record in all_records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels = np.asarray(record["labels_nez"])[valid].astype(int)
        reference = np.asarray(record["v3_anchor_standardized"] if np.asarray(record.get("v3_anchor_valid", [])).any() else record["direct_standardized"])[valid]
        final = np.asarray(record["final_nez_logit"])[valid]
        safe_pairs = (reference[labels == 1, None] - reference[labels == 0][None, :]) >= float(args.p21_preserve_safe_margin) if np.any(labels == 1) and np.any(labels == 0) else np.zeros((0, 0), dtype=bool)
        final_pairs = final[labels == 1, None] - final[labels == 0][None, :] if safe_pairs.size else np.zeros((0, 0))
        patient_metrics, _ = _summarize_prediction_records([record])
        patient_rows.append({
            "outer_fold": record["outer_fold"], "subject_id": record["subject_id"], "center": record["center"],
            "n_channels": int(valid.sum()), "predicted_threshold": record["predicted_patient_threshold"],
            "mean_abs_delta": float(np.mean(np.abs(record.get("delta", np.zeros_like(valid))[valid]))),
            "mean_w_noop": float(np.mean(record.get("w_noop", np.ones_like(valid, dtype=float))[valid])),
            "mean_q_cp": float(np.mean(record.get("q_cp", np.zeros_like(valid, dtype=float))[valid])),
            "delta_p95": float(np.quantile(np.abs(np.asarray(record["delta"])[valid]), 0.95)),
            "correction_saturation_rate": float(np.mean(np.asarray(record["correction_saturation"])[valid])),
            "mean_w_anchor": float(np.mean(np.asarray(record["w_anchor"])[valid])),
            "mean_w_seizure": float(np.mean(np.asarray(record["w_seizure"])[valid])),
            "mean_w_causal": float(np.mean(np.asarray(record["w_causal"])[valid])),
            "causal_valid_fraction": float(np.mean(np.asarray(record["causal_feature_valid"])[valid])),
            "mean_ez_reliability": float(np.mean(np.asarray(record["ez_reliability"])[valid & (np.asarray(record["labels_nez"]) < 0.5)])) if np.any(valid & (np.asarray(record["labels_nez"]) < 0.5)) else 0.0,
            "preserve_pair_count": int(safe_pairs.sum()),
            "preserve_violation_rate": float(np.mean(final_pairs[safe_pairs] < float(args.p21_preserve_beta) * (reference[labels == 1, None] - reference[labels == 0][None, :])[safe_pairs])) if safe_pairs.any() else 0.0,
            "analysis_status": analysis_status,
            **patient_metrics,
        })
    patient_frame = pd.DataFrame(patient_rows)
    patient_frame.to_csv(output / "p21_patient_predictions.csv", index=False)
    _summaries(all_records).assign(analysis_status=analysis_status).to_csv(output / "p21_overall_summary.csv", index=False)
    _summaries(all_records, "outer_fold").assign(analysis_status=analysis_status).to_csv(output / "p21_fold_summary.csv", index=False)
    _summaries(all_records, "center").assign(analysis_status=analysis_status).to_csv(output / "p21_center_summary.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "p21_train_history.csv", index=False)
    pd.DataFrame(histories).to_csv(output / "p21_validation_history.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "p21_v3_vs_final_oracle.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output / "p21_training_stage_audit.csv", index=False)
    channel_frame = pd.read_csv(output / "p21_channel_predictions.csv")
    channel_frame.groupby(["outer_fold", "center"], as_index=False)[["w_noop", "w_anchor", "w_seizure", "w_causal", "fusion_entropy"]].mean().to_csv(output / "p21_gate_diagnostics.csv", index=False)
    patient_frame[["outer_fold", "subject_id", "center", "mean_abs_delta", "delta_p95", "correction_saturation_rate"]].to_csv(output / "p21_delta_diagnostics.csv", index=False)
    patient_frame[["outer_fold", "subject_id", "center", "mean_ez_reliability", "causal_valid_fraction"]].to_csv(output / "p21_reliability_diagnostics.csv", index=False)
    patient_frame[["outer_fold", "subject_id", "center", "preserve_pair_count", "preserve_violation_rate"]].to_csv(output / "p21_preserve_diagnostics.csv", index=False)
    _bootstrap_summary(all_records, int(args.model_seed)).to_csv(output / "p21_bootstrap_ci.csv", index=False)
    _counterfactual_ablation(all_records, float(args.p21_max_total_correction)).to_csv(output / "p21_counterfactual_component_ablation.csv", index=False)
    protocol = {
        "method": "P21_V3_ASRR_NEZ_80", "profile": profile.name, "v3_anchor_mode": mode,
        "analysis_status": analysis_status, "formal_deployable": mode == "nested_fold_safe",
        "positive_label": "nez", "score_semantics": "P(NEZ)", "true_count_used": False,
        "patient_oracle_used_for_prediction": False,
        "checkpoint_primary": str(getattr(args, "p21_checkpoint_objective", "f1")),
        "checkpoint_threshold": float(getattr(args, "p21_checkpoint_threshold", 0.5)),
        "checkpoint_uses_test_labels": False,
        "threshold_source": fold_rows[0]["threshold_source"],
    }
    (output / "p21_protocol_audit.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    (output / "p21_training_stage_audit.json").write_text(json.dumps(stage_rows, indent=2), encoding="utf-8")
    if experiment.v3_anchor_store is not None:
        experiment.v3_anchor_store.write_audit(output / "p21_v3_anchor_audit.json")
        experiment.v3_anchor_store.write_audit_bundle(output / "audit")
    (output / "run_args_p21.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")
    return all_records


__all__ = ["configure_p21_training_stage", "run_p21_v3_asrr"]
