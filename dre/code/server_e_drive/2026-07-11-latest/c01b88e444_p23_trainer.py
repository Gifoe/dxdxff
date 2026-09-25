"""Leak-free P23 nested-CV trainer with an inner-OOF global threshold."""

from __future__ import annotations

import copy
import gc
import json
from pathlib import Path
from statistics import median
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

from data_factory import split_train_val_subjects
from .cane_path_cp_heads import initialize_clean_nez_prototypes
from .cane_path_cp_trainer import (
    _loader, _move, apply_fixed_nez_probability_threshold, build_inner_crossfit_splits,
    collect_ranking_records, ranking_validation_summary,
    _optimizer as _p2_optimizer, _selection_key as _p2_selection_key,
)
from .p23_noisy_ez_loss import compute_p23_loss


_P23_RESUME_VERSION = 1


def _rtc_resume_contract(args: Any) -> dict[str, Any]:
    """Fields whose changes invalidate an RTC/ATC resume artifact."""
    return {
        "outer_split_seed": int(getattr(args, "outer_split_seed", 42)),
        "inner_split_seed": int(getattr(args, "inner_split_seed", 42)),
        "inner_splits": int(getattr(args, "inner_splits", 4)),
        "rtc_tail_tau": float(getattr(args, "rtc_tail_tau", 0.25)),
        "tail_shrinkage_formula_version": "softmin_logit_v1",
        "reliability_formula_version": "count_agreement_v1",
        "tail_rank": (
            bool(getattr(args, "rtc_enable_tail_rank", False)),
            float(getattr(args, "rtc_hc_ez_fraction", 0.20)),
            float(getattr(args, "rtc_tail_rank_margin", 0.10)),
            float(getattr(args, "rtc_tail_rank_weight", 0.03)),
            int(getattr(args, "rtc_tail_rank_start_epoch", 8)),
            int(getattr(args, "rtc_tail_rank_ramp_epochs", 5)),
            float(getattr(args, "rtc_clean_tail_consistency_weight", 0.01)),
        ),
        "checkpoint_objective": str(getattr(args, "cane_selection_objective", "f1")),
        "shift_contract": (
            float(getattr(args, "rtc_shift_b_max", 0.12)),
            str(getattr(args, "rtc_shift_lambda_grid", "0.1,1,10,100")),
            "unlabeled_distribution_15_v1",
        ),
        "feature_cache_hash": str(getattr(args, "rtc_feature_cache_hash", "")),
        "source_hash": str(getattr(args, "rtc_source_hash", "")),
        "atc_contract": (
            bool(getattr(args, "use_p2_atc", False)), str(getattr(args, "p2_atc_profile", "")),
            float(getattr(args, "p2_atc_robust_tail_tau", 0.25)),
            float(getattr(args, "p2_atc_clean_tail_margin", 0.10)),
            float(getattr(args, "p2_atc_clean_tail_weight", 0.020)),
            float(getattr(args, "p2_atc_pair_margin", 0.15)),
            float(getattr(args, "p2_atc_pair_weight", 0.020)),
            float(getattr(args, "p2_atc_trusted_ez_fraction", 0.20)),
            int(getattr(args, "p2_atc_loss_start_epoch", 8)), int(getattr(args, "p2_atc_loss_ramp_epochs", 5)),
            int(getattr(args, "p2_atc_max_pairs_per_patient", 256)),
            str(getattr(args, "p2_atc_feature_cache_hash", "")), str(getattr(args, "p2_atc_source_hash", "")),
        ),
    }

def _inner_resume_path(output: Path, outer_fold: int, inner_fold: int) -> Path:
    return output / "p23_resume" / f"outer_{outer_fold}" / f"inner_{inner_fold}.pt"


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    """Persist a completed stage atomically so a crash cannot create a false resume point."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_inner_resume(
    path: Path,
    *,
    args: Any,
    outer_fold: int,
    inner_fold: int,
    fit_subjects: Sequence[str],
    heldout_subjects: Sequence[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"P23 resume artifact is unreadable: {path}") from error
    expected = {
        "version": _P23_RESUME_VERSION,
        "profile": str(args.p23_profile),
        "rtc_profile": str(getattr(args, "p2_rtc_profile", "")),
        "rtc_cohort": str(getattr(args, "cohort_mode", "")),
        "rtc_subject_set_hash": str(getattr(args, "rtc_cohort_audit", {}).get("subject_set_hash", "")),
        "rtc_fold_ledger_hash": str(getattr(args, "rtc_cohort_audit", {}).get("outer_fold_ledger_hash", "")),
        "model_seed": int(args.model_seed),
        "rtc_resume_contract": _rtc_resume_contract(args),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_subjects": sorted(map(str, fit_subjects)),
        "heldout_subjects": sorted(map(str, heldout_subjects)),
    }
    actual = {name: payload.get(name) for name in expected}
    if actual != expected:
        raise RuntimeError(
            f"P23 resume artifact does not match this experiment: {path}. "
            "Use a new output directory or remove only that incompatible resume artifact."
        )
    if not isinstance(payload.get("records"), list) or not isinstance(payload.get("history"), list):
        raise RuntimeError(f"P23 resume artifact is incomplete: {path}")
    observed = sorted(str(row.get("subject_id")) for row in payload["records"])
    if observed != expected["heldout_subjects"]:
        raise RuntimeError(f"P23 resume artifact has an invalid held-out subject ledger: {path}")
    return payload


def _save_inner_resume(
    path: Path,
    *,
    args: Any,
    outer_fold: int,
    inner_fold: int,
    fit_subjects: Sequence[str],
    heldout_subjects: Sequence[str],
    best_epoch: int,
    history: list[dict[str, Any]],
    prototype_audit: dict[str, Any],
    records: list[dict[str, Any]],
) -> None:
    _atomic_torch_save({
        "version": _P23_RESUME_VERSION,
        "profile": str(args.p23_profile),
        "rtc_profile": str(getattr(args, "p2_rtc_profile", "")),
        "rtc_cohort": str(getattr(args, "cohort_mode", "")),
        "rtc_subject_set_hash": str(getattr(args, "rtc_cohort_audit", {}).get("subject_set_hash", "")),
        "rtc_fold_ledger_hash": str(getattr(args, "rtc_cohort_audit", {}).get("outer_fold_ledger_hash", "")),
        "model_seed": int(args.model_seed),
        "rtc_resume_contract": _rtc_resume_contract(args),
        "outer_fold": int(outer_fold),
        "inner_fold": int(inner_fold),
        "fit_subjects": sorted(map(str, fit_subjects)),
        "heldout_subjects": sorted(map(str, heldout_subjects)),
        "best_epoch": int(best_epoch),
        "history": history,
        "prototype_audit": prototype_audit,
        "records": records,
    }, path)


def _write_outer_resume_progress(output: Path, outer_fold: int, completed_inner_folds: Sequence[int]) -> None:
    path = output / "p23_resume" / f"outer_{outer_fold}" / "progress.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps({
        "outer_fold": int(outer_fold),
        "completed_inner_folds": sorted(map(int, completed_inner_folds)),
        "resume_version": _P23_RESUME_VERSION,
    }, indent=2), encoding="utf-8")
    temporary.replace(path)


def _direct_outer_resume_path(output: Path, outer_fold: int) -> Path:
    return output / "p23_resume" / f"outer_{outer_fold}" / "direct_outer_complete.pt"


def _direct_outer_signature(
    args: Any, outer_fold: int, outer_train: Sequence[str], outer_test: Sequence[str],
    fit_subjects: Sequence[str], validation_subjects: Sequence[str],
) -> dict[str, Any]:
    return {
        "version": _P23_RESUME_VERSION,
        "training_mode": "direct_outer_only",
        "profile": str(args.p23_profile),
        "rtc_profile": str(getattr(args, "p2_rtc_profile", "")),
        "rtc_cohort": str(getattr(args, "cohort_mode", "")),
        "rtc_subject_set_hash": str(getattr(args, "rtc_cohort_audit", {}).get("subject_set_hash", "")),
        "rtc_fold_ledger_hash": str(getattr(args, "rtc_cohort_audit", {}).get("outer_fold_ledger_hash", "")),
        "model_seed": int(args.model_seed),
        "rtc_resume_contract": _rtc_resume_contract(args),
        "outer_fold": int(outer_fold),
        "outer_train": sorted(map(str, outer_train)),
        "outer_test": sorted(map(str, outer_test)),
        "fit_subjects": sorted(map(str, fit_subjects)),
        "validation_subjects": sorted(map(str, validation_subjects)),
        "fixed_split_manifest": str(getattr(args, "fixed_split_manifest", "") or ""),
        "batch_size": int(args.batch_size),
        "patient_batch_size": int(args.patient_batch_size),
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "p23_regression_protocol": bool(getattr(args, "p23_regression_protocol", False)),
        "p23_use_p2_loss": bool(getattr(args, "p23_use_p2_loss", False)),
        "legacy_p2_causal": bool(getattr(args, "use_causal_propagation_residual", False)),
        "p23_max_total_correction": float(getattr(args, "p23_max_total_correction", 0.20)),
        "p23_gate_priors": (
            float(getattr(args, "p23_gate_prior_noop", 0.90)),
            float(getattr(args, "p23_gate_prior_anchor", 0.05)),
            float(getattr(args, "p23_gate_prior_seizure", 0.05)),
        ),
        "checkpoint_objective": str(getattr(args, "cane_selection_objective", "f1")),
        "early_stop_min_delta": float(getattr(args, "cane_early_stop_min_delta", 0.0)),
    }


def _load_direct_outer_resume(
    path: Path, *, args: Any, outer_fold: int,
    outer_train: Sequence[str], outer_test: Sequence[str], fit_subjects: Sequence[str],
    validation_subjects: Sequence[str],
) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as error:
        raise RuntimeError(f"P23 direct outer resume artifact is unreadable: {path}") from error
    expected = _direct_outer_signature(args, outer_fold, outer_train, outer_test, fit_subjects, validation_subjects)
    observed_signature = payload.get("signature")
    if observed_signature != expected:
        raise RuntimeError(
            f"P23 direct outer resume artifact does not match this experiment: {path}. "
            "Use a new output directory when changing batch size, profile, seed, or training settings."
        )
    required = ("records", "no_temporal_records", "summary", "history", "stage_rows")
    if any(name not in payload for name in required):
        raise RuntimeError(f"P23 direct outer resume artifact is incomplete: {path}")
    observed = sorted(str(row.get("subject_id")) for row in payload["records"])
    if observed != expected["outer_test"]:
        raise RuntimeError(f"P23 direct outer resume test ledger is invalid: {path}")
    return payload


def _save_direct_outer_resume(
    path: Path, *, args: Any, outer_fold: int,
    outer_train: Sequence[str], outer_test: Sequence[str], fit_subjects: Sequence[str],
    validation_subjects: Sequence[str],
    records: list[dict[str, Any]], no_temporal_records: list[dict[str, Any]],
    adapter_fit_records: list[dict[str, Any]], adapter_validation_records: list[dict[str, Any]],
    summary: dict[str, Any], history: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
) -> None:
    _atomic_torch_save({
        "signature": _direct_outer_signature(args, outer_fold, outer_train, outer_test, fit_subjects, validation_subjects),
        "records": records,
        "no_temporal_records": no_temporal_records,
        "adapter_fit_records": adapter_fit_records,
        "adapter_validation_records": adapter_validation_records,
        "summary": summary,
        "history": history,
        "stage_rows": stage_rows,
    }, path)


def _write_adapter_frozen_ledgers(
    output: Path,
    outer_fold: int,
    fit_records: Sequence[dict[str, Any]],
    validation_records: Sequence[dict[str, Any]],
    test_records: Sequence[dict[str, Any]],
    model: torch.nn.Module | None,
) -> None:
    """Export the exact frozen P2 fold artifacts needed by a post-hoc adapter.

    The three record sets are produced by the same selected P2 checkpoint and
    use the already-selected validation threshold. This does not fit anything
    on outer test patients.
    """
    fold_dir = output / f"fold_{outer_fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    for name, records in (("fit", fit_records), ("val", validation_records), ("test", test_records)):
        channels, _ = _flatten(records, outer_fold)
        if "final_nez_logit" not in channels or "contextual_channel_embedding" not in channels:
            raise RuntimeError("Frozen adapter ledger export lacks final logit or contextual embedding")
        channels["base_nez_logit"] = channels["final_nez_logit"]
        channels.to_csv(fold_dir / f"{name}_channel_predictions_neuroez_v2_fold_{outer_fold}.csv", index=False)
    checkpoint = fold_dir / "best_model.pt"
    if model is not None:
        _atomic_torch_save({"model_state_dict": model.state_dict(), "outer_fold": int(outer_fold)}, checkpoint)
    if not checkpoint.is_file():
        raise RuntimeError(f"Frozen P2 checkpoint missing for adapter export: {checkpoint}")


def _mrr(labels_nez: np.ndarray, scores_nez: np.ndarray, positive_nez: bool) -> float:
    target = labels_nez if positive_nez else 1 - labels_nez
    ranked = np.argsort(-(scores_nez if positive_nez else 1 - scores_nez))
    ranks = np.flatnonzero(target[ranked] > 0) + 1
    return float(1.0 / ranks[0]) if ranks.size else 0.0


def _ranking_summary(records: Sequence[dict[str, Any]]) -> dict[str, float]:
    nez_ap, ez_ap, ez_mrr = [], [], []
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels = np.asarray(record["labels_nez"])[valid].astype(int)
        scores = np.asarray(record["score_nez"])[valid]
        if np.unique(labels).size == 2:
            nez_ap.append(float(average_precision_score(labels, scores)))
            ez_ap.append(float(average_precision_score(1 - labels, 1 - scores)))
        ez_mrr.append(_mrr(labels, scores, positive_nez=False))
    nez = float(np.mean(nez_ap)) if nez_ap else 0.0
    ez = float(np.mean(ez_ap)) if ez_ap else 0.0
    return {
        "patient_macro_auprc_nez": nez, "patient_macro_auprc_ez": ez,
        "balanced_patient_auprc_hmean": 2 * nez * ez / max(nez + ez, 1e-12),
        "patient_macro_ez_mrr": float(np.mean(ez_mrr)) if ez_mrr else 0.0,
    }


def _select_global_threshold(
    records: Sequence[dict[str, Any]], *, audit_prefix: str = "inner_oof"
) -> tuple[float, dict[str, float]]:
    scores = np.concatenate([np.asarray(row["score_nez"])[np.asarray(row["channel_mask"], dtype=bool)] for row in records])
    candidates = set(np.linspace(0.01, 0.99, 99).tolist())
    if scores.size:
        candidates.update(np.quantile(scores, np.linspace(0.01, 0.99, 51)).tolist())
        unique = np.unique(scores)
        candidates.update(((unique[:-1] + unique[1:]) / 2.0).tolist())
    rows = []
    for threshold in sorted(value for value in candidates if 0.0 <= value <= 1.0):
        macro, nez, ez, balanced = [], [], [], []
        for row in records:
            valid = np.asarray(row["channel_mask"], dtype=bool)
            labels = np.asarray(row["labels_nez"])[valid].astype(int)
            pred = (np.asarray(row["score_nez"])[valid] >= threshold).astype(int)
            macro.append(f1_score(labels, pred, average="macro", zero_division=0))
            nez.append(f1_score(labels, pred, pos_label=1, zero_division=0))
            ez.append(f1_score(labels, pred, pos_label=0, zero_division=0))
            balanced.append(float(((pred[labels == 1] == 1).mean() + (pred[labels == 0] == 0).mean()) / 2.0) if np.any(labels == 1) and np.any(labels == 0) else 0.0)
        rows.append((float(np.mean(macro)), min(float(np.mean(nez)), float(np.mean(ez))), float(np.mean(balanced)), -abs(threshold - 0.5), threshold, float(np.mean(nez)), float(np.mean(ez))))
    if not rows:
        raise RuntimeError("Cannot select P23 threshold from an empty validation ledger")
    best = max(rows)
    return float(best[4]), {
        f"{audit_prefix}_macro_f1": best[0],
        f"{audit_prefix}_nez_f1": best[5],
        f"{audit_prefix}_ez_f1": best[6],
        f"{audit_prefix}_balanced_accuracy": best[2],
    }


def _patient_oracle_macro_f1(records: Sequence[dict[str, Any]]) -> float:
    """Diagnostic-only per-patient threshold upper bound; never used to predict."""
    values = []
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels = np.asarray(record["labels_nez"])[valid].astype(int)
        scores = np.asarray(record["score_nez"])[valid]
        if not scores.size:
            continue
        unique = np.unique(scores)
        candidates = np.r_[0.0, 1.0, (unique[:-1] + unique[1:]) / 2.0]
        values.append(max(
            f1_score(labels, scores >= threshold, average="macro", zero_division=0)
            for threshold in candidates
        ))
    return float(np.mean(values)) if values else 0.0


def _bootstrap_ci(records: Sequence[dict[str, Any]], *, seed: int, n_bootstrap: int = 500) -> pd.DataFrame:
    """Patient-resampled held-out CI. Predictions and thresholds stay fixed."""
    from exp_ez_hybrid import _summarize_prediction_records

    if not records:
        return pd.DataFrame(columns=["metric", "estimate", "ci_low", "ci_high", "n_bootstrap", "status"])
    rng = np.random.default_rng(seed)
    metric_names = ("patient_macro_f1", "patient_macro_nez_f1", "patient_macro_ez_f1", "patient_macro_auprc_nez", "patient_macro_auprc_ez", "patient_macro_ez_mrr")
    point, _ = _summarize_prediction_records(records)
    samples = {name: [] for name in metric_names}
    for _ in range(n_bootstrap):
        selected = [records[index] for index in rng.integers(0, len(records), len(records))]
        summary, _ = _summarize_prediction_records(selected)
        for name in metric_names:
            samples[name].append(float(summary.get(name, float("nan"))))
    return pd.DataFrame([
        {
            "metric": name, "estimate": float(point.get(name, float("nan"))),
            "ci_low": float(np.nanquantile(values, 0.025)), "ci_high": float(np.nanquantile(values, 0.975)),
            "n_bootstrap": n_bootstrap, "status": "patient_resampled_heldout_diagnostic_ci",
        }
        for name, values in samples.items()
    ])


def _update_ema(teacher: torch.nn.Module, student: torch.nn.Module, decay: float) -> None:
    with torch.no_grad():
        for teacher_value, student_value in zip(teacher.parameters(), student.parameters()):
            teacher_value.mul_(decay).add_(student_value, alpha=1.0 - decay)
        for teacher_value, student_value in zip(teacher.buffers(), student.buffers()):
            teacher_value.copy_(student_value)


def _set_stage(model: torch.nn.Module, epoch: int, args: Any) -> str:
    stage1, stage2 = int(args.p23_stage1_end), int(args.p23_stage2_end)
    for parameter in model.parameters():
        parameter.requires_grad = True
    if epoch <= stage1:
        frozen_prefixes = ("b0_encoder", "physics_encoder", "temporal_encoder", "seizure_aggregator", "channel_classifier", "physics_gate")
        for name, parameter in model.named_parameters():
            if name.startswith(frozen_prefixes):
                parameter.requires_grad = False
        return "stage1_new_heads"
    if epoch <= stage2:
        for name, parameter in model.named_parameters():
            if name.startswith("b0_encoder"):
                parameter.requires_grad = False
        return "stage2_partial_backbone"
    if not bool(args.p23_unfreeze_b0_stage3):
        for name, parameter in model.named_parameters():
            if name.startswith("b0_encoder"):
                parameter.requires_grad = False
    return "stage3_full_backbone"


def _optimizer(model: torch.nn.Module, args: Any) -> torch.optim.Optimizer:
    heads, backbone = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        (heads if name.startswith("p23_") else backbone).append(parameter)
    groups = []
    if heads:
        groups.append({"params": heads, "lr": float(args.p23_head_lr)})
    if backbone:
        groups.append({"params": backbone, "lr": float(args.p23_backbone_lr)})
    return torch.optim.AdamW(groups, weight_decay=float(args.weight_decay))


def _train_model(experiment: Any, fit_subjects: Sequence[str], val_subjects: Sequence[str], *, tag: str, max_epochs: int, select_checkpoint: bool) -> tuple[torch.nn.Module, int, list[dict[str, Any]], dict[str, Any]]:
    from neuroez_c.model import NeuroEZCModel

    fit_ds, val_ds, _, _ = experiment._build_datasets(fit_subjects, val_subjects, val_subjects)
    train_loader = _loader(experiment, fit_ds, train=True, seed=int(experiment.args.model_seed))
    val_loader = _loader(experiment, val_ds, train=False, seed=int(experiment.args.model_seed))
    model = NeuroEZCModel(experiment.args).to(experiment.device)
    use_p2_loss = bool(getattr(experiment.args, "p23_use_p2_loss", False))
    # The frozen P2 protocol initializes prototypes after warm-up at epoch 5.
    # RTC's A0 baseline must retain that timing rather than changing its start state.
    prototype_audit: dict[str, Any] = {}
    if not use_p2_loss:
        prototype_audit = initialize_clean_nez_prototypes(
            model, train_loader, experiment.device, int(experiment.args.model_seed),
        )
    teacher = copy.deepcopy(model).eval() if not use_p2_loss else None
    best_key, best_epoch, best_state, no_improve = None, 1, copy.deepcopy(model.state_dict()), 0
    history: list[dict[str, Any]] = []
    optimizer = None
    prior_stage = None
    for epoch in range(1, int(max_epochs) + 1):
        experiment.current_epoch = epoch
        stage = "rtc_p2_protocol" if use_p2_loss else _set_stage(model, epoch, experiment.args)
        if stage != prior_stage:
            optimizer = _p2_optimizer(model, experiment.args) if use_p2_loss else _optimizer(model, experiment.args)
            prior_stage = stage
        model.train(); losses = []
        for batch in train_loader:
            device_batch = _move(batch, experiment.device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(device_batch)
            if use_p2_loss:
                loss, parts = experiment._compute_loss(outputs, device_batch, torch.tensor(1.0, device=experiment.device), split_name="train")
            else:
                with torch.no_grad():
                    teacher_score = teacher(device_batch)["final_score_nez"]
                loss, parts = compute_p23_loss(outputs, device_batch, experiment.args, epoch=epoch, teacher_score_nez=teacher_score)
            loss.backward()
            bad_gradients = [
                name for name, parameter in model.named_parameters()
                if parameter.grad is not None and not bool(torch.isfinite(parameter.grad).all())
            ]
            if bad_gradients:
                raise FloatingPointError(
                    "P23 non-finite gradients before optimizer step: "
                    + ", ".join(bad_gradients[:12])
                )
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), float(experiment.args.grad_clip))
            if not bool(torch.isfinite(grad_norm)):
                raise FloatingPointError("P23 non-finite total gradient norm before optimizer step")
            optimizer.step()
            if teacher is not None:
                _update_ema(teacher, model, float(experiment.args.p23_ema_decay))
            losses.append(float(loss.detach().cpu()))
        if use_p2_loss and epoch == 5:
            prototype_audit = initialize_clean_nez_prototypes(
                model, train_loader, experiment.device, int(experiment.args.model_seed),
                max_samples=int(getattr(experiment.args, "cane_prototype_max_samples", 20_000)),
            )
        records = collect_ranking_records(experiment, model, val_loader, source=tag)
        metrics = ranking_validation_summary(records) if use_p2_loss else _ranking_summary(records)
        key = _p2_selection_key(metrics, epoch, str(getattr(experiment.args, "cane_selection_objective", "f1"))) if use_p2_loss else (metrics["balanced_patient_auprc_hmean"], metrics["patient_macro_auprc_ez"], metrics["patient_macro_ez_mrr"], metrics["patient_macro_auprc_nez"], -float(np.mean(losses) if losses else 0.0), -epoch)
        history.append({"epoch": epoch, "stage": stage, "train_loss": float(np.mean(losses) if losses else 0.0), **metrics})
        improved = best_key is None or key > best_key
        if improved:
            best_key, best_epoch, best_state, no_improve = key, epoch, copy.deepcopy(model.state_dict()), 0
        else:
            no_improve += 1
        if use_p2_loss:
            # _train_model receives the experiment wrapper, not a local args
            # variable.  Keep the log label accurate for the plain P2-Q10
            # profile as well as the optional RTC/ATC variants.
            run_args = experiment.args
            if bool(getattr(run_args, "use_p2_atc", False)):
                protocol_label = "P2-ATC"
            elif bool(getattr(run_args, "use_p2_rtc_shift", False)):
                protocol_label = "P2-RTC"
            else:
                protocol_label = "P2-Q10"
            experiment._log(f"[{protocol_label}] {tag} epoch {epoch}/{max_epochs} | train_loss={history[-1]['train_loss']:.4f} | val_macro_f1={metrics['validation_patient_macro_f1']:.4f} | val_ez_f1={metrics['validation_patient_ez_f1']:.4f}")
        else:
            experiment._log(f"[P23] {tag} epoch {epoch}/{max_epochs} | train_loss={history[-1]['train_loss']:.4f} | val_auprc_hmean={metrics['balanced_patient_auprc_hmean']:.4f} | val_ez_mrr={metrics['patient_macro_ez_mrr']:.4f}")
        if select_checkpoint and epoch >= int(experiment.args.min_epochs_before_early_stop) and no_improve >= int(experiment.args.patience):
            experiment._log(f"[P23] {tag} early stop at epoch {epoch}")
            break
    model.load_state_dict(best_state if select_checkpoint else model.state_dict())
    return model, best_epoch, history, prototype_audit


def _train_model_frozen_p2_protocol(
    experiment: Any, fit_subjects: Sequence[str], validation_subjects: Sequence[str], *,
    tag: str, max_epochs: int,
) -> tuple[torch.nn.Module, int, float, list[dict[str, Any]], dict[str, Any], torch.nn.Module | None]:
    """P2 direct-outer optimizer/checkpoint protocol, with optional P4/P5 EMA loss.

    This deliberately does not use P23's stage scheduler or learning rates. It
    is the comparability boundary for the regression audit: profiles differ
    only in their declared forward/loss component.
    """
    if experiment.device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    fit_ds, val_ds, _, _ = experiment._build_datasets(fit_subjects, validation_subjects, validation_subjects)
    seed = int(experiment.args.model_seed)
    train_loader = _loader(experiment, fit_ds, train=True, seed=seed)
    val_loader = _loader(experiment, val_ds, train=False, seed=seed)
    torch.manual_seed(seed)
    model = experiment.runtime["model_cls"](experiment.args).to(experiment.device)
    experiment._dry_initialize_lazy_layers(model, train_loader)
    optimizer = _p2_optimizer(model, experiment.args)
    noise_enabled = str(experiment.args.p23_profile).upper() in {"P4_NOISE_AWARE", "P5_FULL"}
    teacher = copy.deepcopy(model).eval() if noise_enabled else None
    unit_weight = torch.tensor(1.0, device=experiment.device)
    best_state, best_key, best_epoch = copy.deepcopy(model.state_dict()), None, 0
    best_teacher_state = copy.deepcopy(teacher.state_dict()) if teacher is not None else None
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
        model.train(); losses: list[float] = []; part_rows: dict[str, list[float]] = {}
        for batch in train_loader:
            device_batch = experiment._prepare_quality_batch(_move(batch, experiment.device), split_name="train")
            optimizer.zero_grad(set_to_none=True)
            outputs = model(device_batch)
            if bool(getattr(experiment.args, "p23_use_p2_loss", False)):
                loss, parts = experiment._compute_loss(outputs, device_batch, unit_weight, split_name="train")
            else:
                with torch.no_grad():
                    teacher_score = teacher(device_batch)["final_score_nez"] if teacher is not None else None
                loss, parts = compute_p23_loss(
                    outputs, device_batch, experiment.args, epoch=epoch, teacher_score_nez=teacher_score,
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(experiment.args.grad_clip))
            optimizer.step()
            if teacher is not None:
                _update_ema(teacher, model, float(experiment.args.p23_ema_decay))
            losses.append(float(loss.detach().cpu()))
            for name, value in parts.items():
                if isinstance(value, (int, float)):
                    part_rows.setdefault(name, []).append(float(value))
        if epoch == 5:
            prototype_audit = initialize_clean_nez_prototypes(
                model, train_loader, experiment.device, seed,
                max_samples=int(getattr(experiment.args, "cane_prototype_max_samples", 20_000)),
            )
        validation_records = collect_ranking_records(
            experiment, model, val_loader, source=tag, p23_teacher_model=teacher,
        )
        validation = ranking_validation_summary(validation_records)
        train_metrics = {"loss": float(np.mean(losses) if losses else 0.0)}
        train_metrics.update({name: float(np.mean(values)) for name, values in part_rows.items()})
        row = {"epoch": epoch, "training_protocol": "frozen_p2_regression", **train_metrics, **{f"val_{name}": value for name, value in validation.items()}}
        history.append(row)
        objective = str(getattr(experiment.args, "cane_selection_objective", "f1")).lower()
        key = _p2_selection_key(validation, epoch, objective)
        eligible = epoch >= 5
        minimum_delta = max(0.0, float(getattr(experiment.args, "cane_early_stop_min_delta", 0.0)))
        meaningful = best_key is None or key[0] > best_key[0] + minimum_delta
        if eligible and meaningful:
            best_key, best_epoch, best_threshold, best_state, no_improve = (
                key, epoch, float(validation["validation_selected_threshold"]), copy.deepcopy(model.state_dict()), 0,
            )
            if teacher is not None:
                best_teacher_state = copy.deepcopy(teacher.state_dict())
        else:
            no_improve += 1
        experiment._log(
            f"[P23][P2-regression] {tag} epoch {epoch}/{max_epochs} | "
            f"train_loss={train_metrics['loss']:.4f} | val_macro_f1={validation['validation_patient_macro_f1']:.4f} | "
            f"val_threshold={validation['validation_selected_threshold']:.3f}"
        )
        if epoch >= int(getattr(experiment.args, "min_epochs_before_early_stop", 18)) and no_improve >= int(getattr(experiment.args, "patience", 10)):
            experiment._log(f"[P23][P2-regression] {tag} early stop at epoch {epoch}")
            break
    if best_epoch < 5:
        raise RuntimeError("P23 regression ended before P2 prototype initialization")
    model.load_state_dict(best_state)
    if teacher is not None and best_teacher_state is not None:
        teacher.load_state_dict(best_teacher_state)
        teacher.eval()
    return model, best_epoch, best_threshold, history, prototype_audit, teacher


def _channel_scalar(value: Any, index: int) -> float:
    """Return a scalar channel diagnostic, averaging any seizure-level axis.

    P23 temporal diagnostics are emitted as either ``[channel]`` or
    ``[seizure, channel]``.  The channel ledger is deliberately one row per
    channel, so multi-seizure diagnostics must be reduced rather than cast
    directly to ``float``.
    """
    if value is None:
        return float("nan")
    array = np.asarray(value)
    if array.ndim == 0:
        return float(array)
    if array.ndim == 1:
        return float(array[index])
    # Channel is always the final dimension after record collection.  Mean is
    # also the desired validity fraction for bool temporal-bin diagnostics.
    channel_values = np.take(array, index, axis=-1)
    return float(np.nanmean(channel_values.astype(float)))


def _flatten(records: Sequence[dict[str, Any]], fold: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    channels, patients = [], []
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels = np.asarray(record["labels_nez"])
        for index, name in enumerate(record["canonical_channels"]):
            if not valid[index]:
                continue
            row = {"outer_fold": fold, "subject_id": record["subject_id"], "center": record["center"], "channel_name": name, "channel": name, "label_nez": int(labels[index]), "label_ez": int(1 - labels[index])}
            embedding = record.get("contextual_channel_embedding")
            if embedding is not None:
                vector = np.asarray(embedding)[index]
                row["contextual_channel_embedding"] = json.dumps(np.asarray(vector, dtype=float).tolist())
            for field in ("direct_nez_logit", "direct_score_nez", "score_nez", "score_ez", "final_nez_logit", "anchor_residual", "seizure_term", "r_direct", "u_anchor", "u_seizure", "u_causal", "q_cp", "w_noop", "w_anchor", "w_seizure", "w_causal", "delta", "correction_saturation", "temporal_gate", "temporal_delta_norm", "delta_onset_norm", "delta_spread_norm", "delta_late_norm", "slope_norm", "temporal_bin_valid_fraction", "valid_pre", "valid_onset", "valid_spread", "valid_late", "slope_valid", "seizure_nez_probability_mean", "seizure_nez_probability_std", "seizure_nez_probability_q10", "seizure_nez_probability_raw_q10", "seizure_nez_logit_mean", "seizure_nez_soft_tail_logit", "seizure_nez_robust_tail_logit", "seizure_nez_robust_tail_probability", "seizure_nez_tail_gap", "seizure_nez_bounded_tail_gap", "tail_shrinkage_alpha", "tail_reliability", "tail_residual_suppressed", "seizure_agreement", "valid_seizure_count", "tail_valid", "ema_teacher_score_nez", "observed_ez_reliability", "soft_target_nez", "patient_shift", "raw_final_nez_logit", "calibrated_nez_logit"):
                row[field] = _channel_scalar(record.get(field), index)
            row["predicted_threshold"] = float(record["predicted_patient_threshold"]); row["global_threshold"] = row["predicted_threshold"]; row["prediction_nez"] = int(record["predicted_nez_mask"][index]); row["predicted_nez"] = row["prediction_nez"]; row["predicted_ez"] = int(record["predicted_ez_mask"][index])
            channels.append(row)
        delta = np.asarray(record.get("delta", np.zeros_like(valid)), dtype=float)
        saturation = np.asarray(record.get("correction_saturation", np.zeros_like(valid)), dtype=bool)
        patients.append({"outer_fold": fold, "subject_id": record["subject_id"], "center": record["center"], "n_channels": int(valid.sum()), "predicted_threshold": float(record["predicted_patient_threshold"]), "mean_abs_temporal_delta": float(np.nanmean(np.asarray(record.get("temporal_delta_norm", np.zeros_like(valid)), dtype=float)[valid])), "mean_abs_final_delta": float(np.abs(delta[valid]).mean()), "delta_p95": float(np.quantile(np.abs(delta[valid]), 0.95)), "correction_saturation_rate": float(saturation[valid].mean()), "mean_w_noop": float(np.nanmean(np.asarray(record.get("w_noop", np.ones_like(valid)), dtype=float)[valid])), "mean_w_anchor": float(np.nanmean(np.asarray(record.get("w_anchor", np.zeros_like(valid)), dtype=float)[valid])), "mean_w_seizure": float(np.nanmean(np.asarray(record.get("w_seizure", np.zeros_like(valid)), dtype=float)[valid])), "mean_w_causal": float(np.nanmean(np.asarray(record.get("w_causal", np.zeros_like(valid)), dtype=float)[valid]))})
    return pd.DataFrame(channels), pd.DataFrame(patients)


def _counterfactual_records(records: Sequence[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    """Re-score a finished checkpoint without labels or retraining.

    Each record retains its already-selected inner-OOF threshold.  This is an
    inference ablation, not a replacement for profile-wise retraining.
    """
    result: list[dict[str, Any]] = []
    for original in records:
        record = dict(original)
        direct = np.asarray(record["direct_nez_logit"], dtype=np.float32)
        anchor = np.asarray(record.get("u_anchor", np.zeros_like(direct)), dtype=np.float32)
        seizure = np.asarray(record.get("u_seizure", np.zeros_like(direct)), dtype=np.float32)
        causal = np.asarray(record.get("u_causal", np.zeros_like(direct)), dtype=np.float32)
        weights = {
            "anchor": np.asarray(record.get("w_anchor", np.zeros_like(direct)), dtype=np.float32),
            "seizure": np.asarray(record.get("w_seizure", np.zeros_like(direct)), dtype=np.float32),
            "causal": np.asarray(record.get("w_causal", np.zeros_like(direct)), dtype=np.float32),
        }
        if variant == "C0_FULL":
            delta = np.asarray(record.get("delta", np.zeros_like(direct)), dtype=np.float32)
        elif variant == "C1_NO_TEMPORAL":
            raise RuntimeError("C1_NO_TEMPORAL must be collected through the checkpoint forward override")
        elif variant == "C2_NO_ANCHOR":
            delta = 0.20 * (weights["seizure"] * seizure + weights["causal"] * causal)
        elif variant == "C3_NO_SEIZURE":
            delta = 0.20 * (weights["anchor"] * anchor + weights["causal"] * causal)
        elif variant == "C4_NO_CORRECTION":
            delta = np.zeros_like(direct)
        elif variant == "C5_NO_CAUSAL":
            delta = 0.20 * (weights["anchor"] * anchor + weights["seizure"] * seizure)
        else:
            raise ValueError(f"Unknown P23 counterfactual variant: {variant}")
        mask = np.asarray(record["channel_mask"], dtype=bool)
        score_nez = 1.0 / (1.0 + np.exp(-(direct + delta)))
        threshold = float(record["predicted_patient_threshold"])
        record.update({
            "score_nez": score_nez,
            "score_ez": 1.0 - score_nez,
            "predicted_nez_mask": (score_nez >= threshold) & mask,
            "predicted_ez_mask": (score_nez < threshold) & mask,
            "decision_rule": "fixed_nez_probability_threshold",
            "classification_threshold": threshold,
        })
        result.append(record)
    return result


def run_p23_trn(experiment: Any) -> list[dict[str, Any]]:
    """Train P1-P6 without PATH heads, test labels, true counts, or center inputs."""
    from exp_ez_hybrid import _summarize_prediction_records

    args = experiment.args
    is_rtc = bool(getattr(args, "use_p2_rtc_shift", False))
    is_atc = bool(getattr(args, "use_p2_atc", False))
    is_formal_p2_variant = is_rtc or is_atc
    direct_outer_only = bool(getattr(args, "p23_direct_outer_only", False))
    regression_protocol = bool(getattr(args, "p23_regression_protocol", False))
    if str(args.p23_profile) == "P0_CURRENT_P2" and not regression_protocol:
        raise ValueError("P0 is the frozen P2 baseline. Run scripts/run_step4d_cane_path_cp_nez_80.ps1 for P0.")
    if str(args.p23_init_mode) != "from_scratch":
        raise ValueError("Leak-free P23 inner-OOF thresholding currently requires --p23-init-mode from_scratch; an outer P2 checkpoint would leak inner-heldout labels into threshold fitting.")
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    all_records, fold_rows, histories, stage_rows, channel_frames, patient_frames = [], [], [], [], [], []
    rtc_shift_rows: list[dict[str, Any]] = []
    rtc_inner_oof_frames: list[pd.DataFrame] = []
    no_temporal_records: list[dict[str, Any]] = []
    splits = list(experiment.outer_splits)
    selected_fold = int(getattr(args, "selected_outer_fold", 0) or 0)
    if selected_fold:
        splits = [split for split in splits if int(split["fold_idx"]) == selected_fold]
        if len(splits) != 1:
            raise ValueError(f"selected_outer_fold={selected_fold} is absent from the fixed split contract")
    maximum = int(getattr(args, "max_outer_folds", 0) or 0)
    if maximum:
        splits = splits[:maximum]
    experiment._log(
        f"[P23] start | training_mode={'direct_outer_only' if direct_outer_only else 'nested_inner_oof'} | "
        f"outer_folds={len(splits)} | inner_crossfit={'disabled' if direct_outer_only else 'enabled'} | "
        f"protocol={'frozen_p2_regression' if regression_protocol else 'p23_native'}"
    )
    for split in splits:
        fold, outer_train, outer_test = int(split["fold_idx"]), list(split["train_subjects"]), list(split["test_subjects"])
        if direct_outer_only and "fit_subjects" in split and "validation_subjects" in split:
            fit_subjects, validation_subjects = list(split["fit_subjects"]), list(split["validation_subjects"])
        elif direct_outer_only:
            fit_subjects, validation_subjects = split_train_val_subjects(
                outer_train,
                val_ratio=float(getattr(args, "val_ratio", 0.2)),
                random_seed=int(getattr(args, "outer_split_seed", 42)),
                fold_idx=fold,
            )
        else:
            fit_subjects, validation_subjects = [], []
        experiment._log(f"[P23] outer {fold}/{len(splits)} start | train_patients={len(outer_train)} | test_patients={len(outer_test)}")
        direct_resume_path = _direct_outer_resume_path(output, fold)
        direct_resumed = _load_direct_outer_resume(
            direct_resume_path, args=args, outer_fold=fold,
            outer_train=outer_train, outer_test=outer_test, fit_subjects=fit_subjects,
            validation_subjects=validation_subjects,
        ) if direct_outer_only else None
        if direct_resumed is not None:
            enriched = list(direct_resumed["records"])
            resumed_no_temporal = list(direct_resumed["no_temporal_records"])
            fold_rows.append(dict(direct_resumed["summary"]))
            all_records.extend(enriched)
            no_temporal_records.extend(resumed_no_temporal)
            histories.extend(list(direct_resumed["history"]))
            stage_rows.extend(list(direct_resumed["stage_rows"]))
            if "adapter_fit_records" in direct_resumed and "adapter_validation_records" in direct_resumed:
                _write_adapter_frozen_ledgers(
                    output, fold,
                    direct_resumed["adapter_fit_records"],
                    direct_resumed["adapter_validation_records"],
                    enriched, None,
                )
            else:
                experiment._log(
                    f"[P23][Resume][OuterOnly] fold {fold}: legacy resume has no fit/validation adapter exports"
                )
            cframe, pframe = _flatten(enriched, fold)
            channel_frames.append(cframe); patient_frames.append(pframe)
            experiment._log(
                f"[P23][Resume][OuterOnly] fold {fold}/{len(splits)} restored | patients={len(enriched)}"
            )
            continue
        if direct_outer_only:
            fit_set, validation_set, test_set = set(fit_subjects), set(validation_subjects), set(outer_test)
            if fit_set & validation_set or (fit_set | validation_set) & test_set:
                raise RuntimeError(f"P23 outer-only patient leakage detected in fold {fold}")
            if fit_set | validation_set != set(outer_train):
                raise RuntimeError(f"P23 outer-only fit/validation split does not cover outer train in fold {fold}")
            experiment._log(
                f"[P23][OuterOnly] fold {fold}/{len(splits)} | fit_patients={len(fit_subjects)} | "
                f"val_patients={len(validation_subjects)} | test_patients={len(outer_test)}"
            )
            if regression_protocol:
                model, best_epoch, selected_threshold, history, proto, teacher = _train_model_frozen_p2_protocol(
                    experiment, fit_subjects, validation_subjects,
                    tag=f"outer{fold}_direct", max_epochs=int(args.epochs),
                )
            else:
                model, best_epoch, history, proto = _train_model(
                    experiment, fit_subjects, validation_subjects,
                    tag=f"outer{fold}_direct", max_epochs=int(args.epochs), select_checkpoint=True,
                )
                teacher = None
            validation_ds = experiment._build_datasets(
                fit_subjects, validation_subjects, outer_test
            )[1]
            validation_records = collect_ranking_records(
                experiment, model,
                _loader(experiment, validation_ds, train=False, seed=int(args.model_seed)),
                source=f"outer{fold}_validation", p23_teacher_model=teacher,
            )
            if {str(row["subject_id"]) for row in validation_records} != set(map(str, validation_subjects)):
                raise RuntimeError(f"P23 outer-only validation ledger is incomplete in fold {fold}")
            if regression_protocol:
                validation_summary = ranking_validation_summary(validation_records)
                threshold = float(selected_threshold)
                threshold_audit = {
                    "validation_macro_f1": float(validation_summary["validation_patient_macro_f1"]),
                    "validation_nez_f1": float(validation_summary["validation_patient_nez_f1"]),
                    "validation_ez_f1": float("nan"),
                    "validation_balanced_accuracy": float("nan"),
                    "validation_threshold_grid": "0.05_to_0.95_step_0.025",
                }
            else:
                threshold, threshold_audit = _select_global_threshold(
                    validation_records, audit_prefix="validation"
                )
            histories.extend({"outer_fold": fold, "inner_fold": 0, **row} for row in history)
            stage_rows.append({
                "outer_fold": fold, "inner_fold": 0, "best_epoch": best_epoch,
                "training_mode": "direct_outer_only", **proto,
            })
            test_ds = experiment._build_datasets(
                fit_subjects, validation_subjects, outer_test
            )[2]
            fit_ds = experiment._build_datasets(
                fit_subjects, validation_subjects, outer_test
            )[0]
            threshold_source = "fold_validation_macro_f1_p2_grid" if regression_protocol else "fold_validation_macro_f1"
            checkpoint_selection = "validation_patient_macro_f1_p2_protocol" if regression_protocol else "validation_balanced_patient_auprc_hmean"
        else:
            oof, best_epochs, completed_inner_folds = [], [], []
            for inner in build_inner_crossfit_splits(outer_train, int(args.inner_splits), int(args.inner_split_seed)):
                inner_fold = int(inner["inner_fold"])
                tag = f"outer{fold}_inner{inner_fold}"
                resume_path = _inner_resume_path(output, fold, inner_fold)
                resumed = _load_inner_resume(
                    resume_path, args=args, outer_fold=fold, inner_fold=inner_fold,
                    fit_subjects=inner["fit_subjects"], heldout_subjects=inner["heldout_subjects"],
                )
                if resumed is not None:
                    best_epoch = int(resumed["best_epoch"])
                    history = list(resumed["history"])
                    proto = dict(resumed["prototype_audit"])
                    records = list(resumed["records"])
                    experiment._log(f"[P23][Resume] {tag} restored | best_epoch={best_epoch} | patients={len(records)}")
                else:
                    model, best_epoch, history, proto = _train_model(experiment, inner["fit_subjects"], inner["heldout_subjects"], tag=tag, max_epochs=int(args.epochs), select_checkpoint=True)
                    ds = experiment._build_datasets(inner["fit_subjects"], inner["heldout_subjects"], inner["heldout_subjects"])[1]
                    records = collect_ranking_records(experiment, model, _loader(experiment, ds, train=False, seed=int(args.model_seed)), source=tag)
                    _save_inner_resume(
                        resume_path, args=args, outer_fold=fold, inner_fold=inner_fold,
                        fit_subjects=inner["fit_subjects"], heldout_subjects=inner["heldout_subjects"],
                        best_epoch=best_epoch, history=history, prototype_audit=proto, records=records,
                    )
                    experiment._log(f"[P23][Resume] {tag} checkpointed | path={resume_path}")
                    del model; gc.collect()
                    if experiment.device.type == "cuda":
                        # Checkpoints are already written before cleanup. A full CUDA
                        # synchronize is unnecessary here and can terminate the Windows
                        # Python process with a driver-level access violation.
                        torch.cuda.empty_cache()
                oof.extend(records); best_epochs.append(best_epoch); histories.extend({"outer_fold": fold, "inner_fold": inner["inner_fold"], **row} for row in history); stage_rows.append({"outer_fold": fold, "inner_fold": inner["inner_fold"], "best_epoch": best_epoch, **proto})
                completed_inner_folds.append(inner_fold)
                _write_outer_resume_progress(output, fold, completed_inner_folds)
            if {row["subject_id"] for row in oof} != set(outer_train):
                raise RuntimeError("P23 inner OOF threshold ledger does not cover outer train exactly once")
            shift_calibrator = None
            shift_audit: dict[str, Any] = {}
            if bool(getattr(args, "use_p2_rtc_shift", False)) and str(getattr(args, "p2_rtc_profile", "A0")).upper() == "A5":
                from .p2_monotonic_shift_calibrator import apply_patient_shift, fit_patient_shift_calibrator

                values = [float(value) for value in str(getattr(args, "rtc_shift_lambda_grid", "0.1,1,10,100")).split(",") if value.strip()]
                shift_calibrator, shift_audit = fit_patient_shift_calibrator(
                    oof, seed=int(args.inner_split_seed), b_max=float(args.rtc_shift_b_max), lambda_grid=values,
                )
                oof, inner_shift_summary = apply_patient_shift(oof, shift_calibrator)
                shift_audit.update({"outer_fold": fold, "scope": "inner_oof", **inner_shift_summary})
                rtc_shift_rows.append(dict(shift_audit))
            threshold, threshold_audit = _select_global_threshold(oof)
            if shift_audit:
                threshold_audit.update({"shift_calibration": True, "shift_selected_lambda": shift_audit["selected_lambda"]})
            final_epochs = max(1, int(round(median(best_epochs))))
            model, _, history, proto = _train_model(experiment, outer_train, outer_train, tag=f"outer{fold}_final", max_epochs=final_epochs, select_checkpoint=False)
            histories.extend({"outer_fold": fold, "inner_fold": 0, **row} for row in history); stage_rows.append({"outer_fold": fold, "inner_fold": 0, "best_epoch": final_epochs, **proto})
            test_ds = experiment._build_datasets(outer_train, outer_train, outer_test)[2]
            threshold_source = "outer_train_inner_oof_global_macro_f1"
            checkpoint_selection = "validation_balanced_patient_auprc_hmean"
        test = collect_ranking_records(
            experiment, model, _loader(experiment, test_ds, train=False, seed=int(args.model_seed)),
            source=f"outer{fold}_test", p23_teacher_model=teacher if direct_outer_only else None,
        )
        if not direct_outer_only and bool(getattr(args, "use_p2_rtc_shift", False)) and str(getattr(args, "p2_rtc_profile", "A0")).upper() == "A5":
            from .p2_monotonic_shift_calibrator import apply_patient_shift

            test, test_shift_summary = apply_patient_shift(test, shift_calibrator)
            rtc_shift_rows.append({"outer_fold": fold, "scope": "outer_test", **test_shift_summary})
        predicted = apply_fixed_nez_probability_threshold(test, threshold, threshold_source=threshold_source)
        if direct_outer_only:
            fit_records = apply_fixed_nez_probability_threshold(
                collect_ranking_records(
                    experiment, model, _loader(experiment, fit_ds, train=False, seed=int(args.model_seed)),
                    source=f"outer{fold}_fit", p23_teacher_model=teacher,
                ), threshold, threshold_source=threshold_source,
            )
            validation_records = apply_fixed_nez_probability_threshold(
                validation_records, threshold, threshold_source=threshold_source,
            )
            _write_adapter_frozen_ledgers(
                output, fold, fit_records, validation_records, predicted, model,
            )
        no_temporal = collect_ranking_records(
            experiment,
            model,
            _loader(experiment, test_ds, train=False, seed=int(args.model_seed)),
            source=f"outer{fold}_test_no_temporal",
            p23_disable_temporal=True,
            p23_teacher_model=teacher if direct_outer_only else None,
        )
        fold_no_temporal = apply_fixed_nez_probability_threshold(
            no_temporal, threshold, threshold_source=threshold_source
        )
        no_temporal_records.extend(fold_no_temporal)
        summary, enriched = _summarize_prediction_records(predicted); summary.update({
            "outer_fold": fold, "classification_threshold": threshold,
            "threshold_source": threshold_source,
            "checkpoint_selection": checkpoint_selection,
            "training_mode": "direct_outer_only" if direct_outer_only else "nested_inner_oof",
            "inner_crossfit_used": not direct_outer_only,
            "training_protocol": "frozen_p2_regression" if regression_protocol else "p23_native",
            "patient_oracle_macro_f1": _patient_oracle_macro_f1(enriched),
            "patient_oracle_status": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE",
            **threshold_audit,
        })
        fold_rows.append(summary); all_records.extend(enriched)
        cframe, pframe = _flatten(enriched, fold); channel_frames.append(cframe); patient_frames.append(pframe)
        if not direct_outer_only:
            inner_channel, _ = _flatten(oof, fold)
            rtc_inner_oof_frames.append(inner_channel)
        if direct_outer_only:
            fold_history = [row for row in histories if int(row.get("outer_fold", -1)) == fold]
            fold_stages = [row for row in stage_rows if int(row.get("outer_fold", -1)) == fold]
            _save_direct_outer_resume(
                direct_resume_path, args=args, outer_fold=fold,
                outer_train=outer_train, outer_test=outer_test, fit_subjects=fit_subjects,
                validation_subjects=validation_subjects,
                records=enriched, no_temporal_records=fold_no_temporal,
                adapter_fit_records=fit_records,
                adapter_validation_records=validation_records,
                summary=summary, history=fold_history, stage_rows=fold_stages,
            )
            experiment._log(f"[P23][Resume][OuterOnly] fold {fold} checkpointed | path={direct_resume_path}")
        del model; gc.collect()
        if experiment.device.type == "cuda": torch.cuda.empty_cache()
    overall, _ = _summarize_prediction_records(all_records)
    cohort_name = str(getattr(args, "cohort_mode", "sensitivity80")).lower()
    overall.update({
        "analysis_status": "posthoc_sensitivity_not_primary" if cohort_name == "sensitivity80" else "primary_frozen_all90",
        "training_mode": "direct_outer_only" if direct_outer_only else "nested_inner_oof",
        "inner_crossfit_used": not direct_outer_only,
        "training_protocol": "frozen_p2_regression" if regression_protocol else "p23_native",
        "decision_rule": "fold_validation_global_nez_probability_threshold" if direct_outer_only else "outer_train_inner_oof_global_nez_probability_threshold",
        "true_count_used_for_prediction": False, "oracle_threshold_used_for_prediction": False,
        "center_used_as_model_input": False,
        "patient_oracle_macro_f1": _patient_oracle_macro_f1(all_records),
        "patient_oracle_status": "DIAGNOSTIC_ONLY_NOT_DEPLOYABLE_NOT_USED_FOR_PREDICTION",
    })
    channels = pd.concat(channel_frames, ignore_index=True); patients = pd.concat(patient_frames, ignore_index=True)
    pd.DataFrame(histories).to_csv(output / "p23_train_history.csv", index=False); pd.DataFrame(histories).to_csv(output / "p23_validation_history.csv", index=False)
    pd.DataFrame(stage_rows).to_csv(output / "p23_training_stage_audit.csv", index=False); channels.to_csv(output / "p23_channel_predictions.csv", index=False); patients.to_csv(output / "p23_patient_predictions.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "p23_fold_summary.csv", index=False); pd.DataFrame([overall]).to_csv(output / "p23_overall_summary.csv", index=False)
    if is_formal_p2_variant:
        # Generic artifacts coexist with P23 compatibility outputs and are
        # consumed by the formal RTC/ATC comparison audits.
        channels.to_csv(output / "outer_test_channel_predictions.csv", index=False)
        patients.to_csv(output / "patient_metrics.csv", index=False)
        pd.DataFrame(fold_rows).to_csv(output / "fold_summary.csv", index=False)
        pd.DataFrame([overall]).to_csv(output / "overall_summary.csv", index=False)
        pd.DataFrame(histories).to_csv(output / "train_history.csv", index=False)
        if rtc_inner_oof_frames:
            pd.concat(rtc_inner_oof_frames, ignore_index=True).to_csv(output / "inner_oof_channel_predictions.csv", index=False)
        if is_rtc:
            pd.DataFrame(rtc_shift_rows).to_csv(output / "shift_calibrator_audit.csv", index=False)
    center_rows = []
    for center in sorted({str(record["center"]) for record in all_records}):
        center_records = [record for record in all_records if str(record["center"]) == center]
        center_summary, _ = _summarize_prediction_records(center_records)
        center_rows.append({"center": center, "n_patients": len(center_records), **center_summary})
    pd.DataFrame(center_rows).to_csv(output / "p23_center_summary.csv", index=False)
    if is_formal_p2_variant:
        pd.DataFrame(center_rows).to_csv(output / "center_summary.csv", index=False)
        tail_columns = [
            name for name in (
                "seizure_nez_probability_raw_q10", "seizure_nez_robust_tail_logit",
                "seizure_nez_robust_tail_probability", "seizure_nez_tail_gap",
                "seizure_nez_soft_tail_logit", "seizure_nez_bounded_tail_gap",
                "tail_shrinkage_alpha", "tail_reliability", "valid_seizure_count", "seizure_agreement",
            ) if name in channels.columns
        ]
        if tail_columns:
            channels.groupby(["outer_fold", "center"], as_index=False)[tail_columns].mean().to_csv(output / "tail_diagnostics.csv", index=False)
        pd.DataFrame(stage_rows).to_csv(output / "checkpoint_selection_audit.csv", index=False)
    channels.groupby(["outer_fold", "center"], as_index=False)[["temporal_delta_norm", "delta_onset_norm", "delta_spread_norm", "delta_late_norm", "slope_norm"]].mean().to_csv(output / "p23_temporal_diagnostics.csv", index=False)
    channels.groupby(["outer_fold", "center"], as_index=False)[["seizure_nez_probability_q10", "seizure_agreement", "u_seizure"]].mean().to_csv(output / "p23_seizure_tail_diagnostics.csv", index=False)
    channels.groupby(["outer_fold", "center"], as_index=False)[["w_noop", "w_anchor", "w_seizure", "w_causal", "delta"]].mean().to_csv(output / "p23_gate_diagnostics.csv", index=False)
    channels.groupby(["outer_fold", "center"], as_index=False)[["observed_ez_reliability", "soft_target_nez"]].mean().to_csv(output / "p23_noise_diagnostics.csv", index=False)
    variants = ["C0_FULL", "C1_NO_TEMPORAL", "C2_NO_ANCHOR", "C3_NO_SEIZURE", "C4_NO_CORRECTION"]
    if bool(args.p23_use_causal):
        variants.append("C5_NO_CAUSAL")
    ablation_rows = []
    for name in variants:
        ablated = no_temporal_records if name == "C1_NO_TEMPORAL" else _counterfactual_records(all_records, name)
        ablation_summary, _ = _summarize_prediction_records(ablated)
        ablation_rows.append({
            "variant": name,
            "analysis_status": "COUNTERFACTUAL_INFERENCE_ABLATION_NOT_RETRAINED",
            "mean_abs_delta": float(np.mean([np.abs(np.asarray(row.get("delta", 0.0))).mean() for row in ablated])),
            **ablation_summary,
        })
    pd.DataFrame(ablation_rows).to_csv(output / "p23_counterfactual_ablation.csv", index=False)
    _bootstrap_ci(all_records, seed=int(args.model_seed)).to_csv(output / "p23_bootstrap_ci.csv", index=False)
    formal_threshold_source = "fold_validation_only" if direct_outer_only else "outer_train_inner_oof_only"
    method = "P2_ATC" if is_atc else "P2_RTC_SHIFT" if is_rtc else "P23_TRN_NEZ_80"
    profile_name = str(getattr(args, "p2_atc_profile", "")) if is_atc else str(getattr(args, "p2_rtc_profile", args.p23_profile)) if is_rtc else args.p23_profile
    protocol = {"method": method, "profile": profile_name, "cohort_name": cohort_name, "positive_label": "nez", "score_semantics": "P(NEZ)", "v3_used": False, "raw_used": False, "true_k_used": False, "test_labels_used_for_prediction": False, "training_mode": "direct_outer_only" if direct_outer_only else "nested_inner_oof", "inner_crossfit_used": not direct_outer_only, "training_protocol": "frozen_p2_regression" if regression_protocol else "p23_native", "p23_use_p2_loss": bool(getattr(args, "p23_use_p2_loss", False)), "threshold_source": formal_threshold_source, "causal_used": bool(args.p23_use_causal), "legacy_p2_causal_retained": bool(getattr(args, "use_causal_propagation_residual", False))}
    if is_rtc:
        protocol.update({"robust_tail_tau": float(args.rtc_tail_tau), "tail_shrinkage_formula_version": "softmin_logit_v1", "reliability_formula_version": "count_agreement_v1", "checkpoint_objective": str(args.cane_selection_objective), "shift_b_max": float(args.rtc_shift_b_max), "shift_lambda_grid": str(args.rtc_shift_lambda_grid), "cohort_subject_set_hash": getattr(args, "rtc_cohort_audit", {}).get("subject_set_hash", ""), "outer_fold_ledger_hash": getattr(args, "rtc_cohort_audit", {}).get("outer_fold_ledger_hash", "")})
    if is_atc:
        protocol.update({"robust_tail_tau": float(args.p2_atc_robust_tail_tau), "tail_feature_version": "atc7_v1", "checkpoint_objective": str(args.cane_selection_objective), "cohort_subject_set_hash": getattr(args, "rtc_cohort_audit", {}).get("subject_set_hash", ""), "outer_fold_ledger_hash": getattr(args, "rtc_cohort_audit", {}).get("outer_fold_ledger_hash", "")})
    (output / "p23_protocol_audit.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    if is_formal_p2_variant:
        (output / "protocol_audit.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
        (output / "leakage_audit.json").write_text(json.dumps({"outer_test_labels_used_for_prediction": False, "true_count_used_for_prediction": False, "center_as_model_input": False, "inner_oof_used": True, "status": "passed"}, indent=2), encoding="utf-8")
    (output / "p23_training_stage_audit.json").write_text(json.dumps(stage_rows, indent=2), encoding="utf-8")
    (output / "p23_feature_contract_audit.json").write_text(json.dumps({
        "window_centers_used": True, "center_as_input": False, "causal_required": bool(args.p23_use_causal),
        "raw_waveform_used": False, "v3_used": False, "true_k_used": False,
        "formal_threshold_source": formal_threshold_source,
    }, indent=2), encoding="utf-8")
    (output / "P23_TRN_SENSITIVITY80_REPORT.md").write_text(
        "# P23-TRN Sensitivity80 Result\n\n"
        f"Profile: `{args.p23_profile}`\n\n"
        f"Formal patient macro-F1: `{overall.get('patient_macro_f1', float('nan')):.6f}`\n\n"
        f"Patient-oracle macro-F1 (diagnostic only, not deployable): `{overall['patient_oracle_macro_f1']:.6f}`\n\n"
        f"Formal predictions use only {'fold validation' if direct_outer_only else 'outer-train inner-OOF'} global thresholds. This run does not use V3, raw waveforms, true-K, center-specific models, center-specific thresholds, or test labels for prediction.\n",
        encoding="utf-8",
    )
    return all_records
