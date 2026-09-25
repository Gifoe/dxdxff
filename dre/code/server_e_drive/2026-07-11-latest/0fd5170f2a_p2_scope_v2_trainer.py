"""Fixed-validation, outer-fold-only trainer for P2-SCOPE-v2."""
from __future__ import annotations

import csv
import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score

from .cane_path_cp_heads import initialize_clean_nez_prototypes
from .cane_path_cp_trainer import _loader, _move, _optimizer, collect_ranking_records
from .p2_scope_v2_decoder import (
    FORMAL_PREDICTION_SOURCE,
    PREDICTED_K_DECISION_RULE,
    decode_predicted_k_topk,
)


def set_scope_fold_seed(base_seed: int, outer_fold: int) -> int:
    """Reset all stochastic state so isolated and full runs match per fold."""
    seed = int(base_seed) + 1009 * int(outer_fold)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _stable_json_hash(value: Any) -> str:
    return _sha256_bytes(json.dumps(value, sort_keys=True, default=str, separators=(",", ":")).encode("utf-8"))


def _read_ledger(path: Path, outer_splits: Iterable[dict[str, Any]]) -> tuple[list[dict[str, str]], str]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    if not rows:
        raise RuntimeError("SCOPE fixed validation ledger is empty")
    by_fold = {int(split["fold_idx"]): split for split in outer_splits}
    for fold, split in by_fold.items():
        partition = {role: {str(row["subject_id"]) for row in rows if int(row["outer_fold"]) == fold and row["role"] == role} for role in ("fit", "validation", "test")}
        expected_train = set(map(str, split["train_subjects"]))
        expected_test = set(map(str, split["test_subjects"]))
        if partition["test"] != expected_test:
            raise RuntimeError(f"SCOPE ledger outer test mismatch in fold {fold}")
        if partition["fit"] & partition["validation"] or partition["fit"] & partition["test"] or partition["validation"] & partition["test"]:
            raise RuntimeError(f"SCOPE ledger patient leakage in fold {fold}")
        if partition["fit"] | partition["validation"] != expected_train:
            raise RuntimeError(f"SCOPE fit/validation does not cover outer train in fold {fold}")
    return rows, _sha256_bytes(path.read_bytes())


def _global_decoder_bounds(patient_index: dict[str, Any]) -> tuple[int, bool, dict[str, Any]]:
    """Set decoder bounds once for the whole cohort; never per test patient."""
    all_mixed = True
    for meta in patient_index.values():
        labels = np.asarray(meta.get("labels", []))
        valid = labels[labels >= 0]
        all_mixed = all_mixed and bool(np.any(valid > .5)) and bool(np.any(valid <= .5))
    return (1 if all_mixed else 0, all_mixed, {"min_k": 1 if all_mixed else 0, "max_k": "n_minus_1" if all_mixed else "n", "all_patients_have_ez_and_nez": all_mixed})


def _decode_records(records: list[dict[str, Any]], *, min_k: int, require_mixed_channels: bool, outer_fold: int) -> list[dict[str, Any]]:
    """Formal decode uses only scores, mask, alpha, and beta."""
    decoded: list[dict[str, Any]] = []
    for record in records:
        row = dict(record)
        valid_channels = int(np.asarray(row["channel_mask"], dtype=bool).sum())
        result = decode_predicted_k_topk(
            scope_ez_score=torch.as_tensor(row["scope_ez_score"], dtype=torch.float32).unsqueeze(0),
            channel_mask=torch.as_tensor(row["channel_mask"], dtype=torch.bool).unsqueeze(0),
            alpha=torch.tensor([float(row["scope_count_alpha"])], dtype=torch.float32),
            beta=torch.tensor([float(row["scope_count_beta"])], dtype=torch.float32),
            min_k=min_k,
            max_k=valid_channels - 1 if require_mixed_channels else valid_channels,
        )
        row.update({
            "score_nez": np.asarray(row["scope_score_nez"]),
            "score_ez": np.asarray(row["scope_score_ez"]),
            "predicted_ez_mask": result["predicted_ez_mask"][0].cpu().numpy(),
            "predicted_nez_mask": result["predicted_nez_mask"][0].cpu().numpy(),
            "scope_predicted_k": int(result["scope_predicted_k"][0]),
            "scope_predicted_ez_fraction": float(result["scope_predicted_ez_fraction"][0]),
            "scope_count_prior_mode": int(result["scope_count_prior_mode"][0]),
            "scope_count_prior_mean": float(result["scope_count_prior_mean"][0]),
            "decision_rule": PREDICTED_K_DECISION_RULE,
            "formal_prediction_source": FORMAL_PREDICTION_SOURCE,
            "true_count_used_for_prediction": False,
            "classification_threshold": float("nan"),
            "threshold_source": "not_used",
            "outer_fold": int(outer_fold),
        })
        decoded.append(row)
    return decoded


def _true_k_diagnostic_macro_f1(records: list[dict[str, Any]]) -> float:
    """Label-assisted diagnostic only. It never writes formal prediction masks."""
    values: list[float] = []
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        labels_ez = np.asarray(record["labels_ez"])[valid].astype(int)
        score = np.asarray(record["scope_ez_score"])[valid]
        prediction = np.zeros_like(labels_ez)
        prediction[np.argsort(score)[::-1][:int(labels_ez.sum())]] = 1
        values.append(float(f1_score(labels_ez, prediction, average="macro", labels=[0, 1], zero_division=0)))
    return float(np.mean(values)) if values else 0.0


def _rolling(history: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([row[key] for row in history[-3:]]))


def _apply_p2_lr_schedule(optimizer: torch.optim.Optimizer, args: Any, epoch: int) -> tuple[float, float]:
    backbone_lr = float(getattr(args, "cane_backbone_lr_stage1", 1e-4)) if epoch <= 5 else float(getattr(args, "cane_backbone_lr_after_warmup", 2e-5))
    head_lr = float(getattr(args, "cane_head_lr", 1e-4))
    for group in optimizer.param_groups:
        group["lr"] = head_lr if group.get("group_name") == "cane_heads" else backbone_lr
    return backbone_lr, head_lr


def _prior(patient_index: dict[str, Any], subjects: list[str]) -> float:
    values: list[float] = []
    for subject in subjects:
        labels = np.asarray(patient_index[subject]["labels"])
        values.extend(labels[labels >= 0].tolist())
    if not values:
        raise RuntimeError("Cannot fit SCOPE cardinality prior from empty fit labels")
    return float(np.mean(np.asarray(values) > .5))


def _completion_matches(path: Path, *, config_hash: str, ledger_hash: str, outer_hash: str) -> bool:
    if not path.is_file():
        return False
    saved = json.loads(path.read_text(encoding="utf-8"))
    mismatched = [name for name, expected in (("config_hash", config_hash), ("validation_ledger_hash", ledger_hash), ("outer_fold_hash", outer_hash)) if saved.get(name) != expected]
    if mismatched:
        raise RuntimeError(f"SCOPE resume hash mismatch in {path}: {mismatched}")
    return saved.get("status") == "complete"


def _patient_rows(records: list[dict[str, Any]], best_epochs: dict[int, int], fold_seeds: dict[int, int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool)
        y_ez = np.asarray(record["labels_ez"])[valid].astype(int)
        pred_ez = np.asarray(record["predicted_ez_mask"])[valid].astype(int)
        y_nez, pred_nez = 1 - y_ez, 1 - pred_ez
        fold = int(record["outer_fold"])
        rows.append({
            "subject_id": record["subject_id"], "center": record["center"], "outer_fold": fold,
            "n_channels": int(valid.sum()), "n_seizures": int(record.get("n_seizures", 0)),
            "true_ez_count": int(y_ez.sum()), "predicted_ez_count": int(pred_ez.sum()),
            "count_absolute_error": abs(int(y_ez.sum()) - int(pred_ez.sum())),
            "true_ez_fraction": float(y_ez.mean()), "predicted_ez_fraction": float(pred_ez.mean()),
            "fraction_absolute_error": float(abs(y_ez.mean() - pred_ez.mean())),
            "patient_macro_f1": float(f1_score(y_nez, pred_nez, average="macro", labels=[0, 1], zero_division=0)),
            "patient_ez_f1": float(f1_score(y_ez, pred_ez, pos_label=1, zero_division=0)),
            "patient_nez_f1": float(f1_score(y_nez, pred_nez, pos_label=1, zero_division=0)),
            "patient_balanced_accuracy": float(np.mean([(pred_ez[y_ez == 1] == 1).mean() if np.any(y_ez == 1) else 0.0, (pred_nez[y_nez == 1] == 1).mean() if np.any(y_nez == 1) else 0.0])),
            "ez_auprc": float(average_precision_score(y_ez, np.asarray(record["scope_score_ez"])[valid])) if np.unique(y_ez).size > 1 else 0.0, "ez_mrr": float(record.get("ez_mrr", 0.0)),
            "diagnostic_true_k_topk_macro_f1": _true_k_diagnostic_macro_f1([record]),
            "diagnostic_true_k_topk_ez_f1": float(f1_score(y_ez, np.isin(np.arange(len(y_ez)), np.argsort(np.asarray(record["scope_ez_score"])[valid])[::-1][:int(y_ez.sum())]).astype(int), pos_label=1, zero_division=0)),
            "decision_rule": PREDICTED_K_DECISION_RULE, "formal_prediction_source": FORMAL_PREDICTION_SOURCE,
            "true_count_used_for_prediction": False, "best_epoch": best_epochs[fold], "fold_seed": fold_seeds[fold],
        })
    return rows


def _channel_rows(records: list[dict[str, Any]], best_epochs: dict[int, int], fold_seeds: dict[int, int], ledger_hash: str, config_hash: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    fields = ("scope_count_prior_mu0", "scope_count_mu_delta", "scope_count_predicted_mu", "scope_count_predicted_kappa", "scope_count_alpha", "scope_count_beta", "scope_predicted_k", "scope_predicted_ez_fraction")
    for record in records:
        valid = np.asarray(record["channel_mask"], dtype=bool); fold = int(record["outer_fold"])
        for idx, name in enumerate(record["canonical_channels"]):
            if not valid[idx]:
                continue
            row = {"subject_id": record["subject_id"], "center": record["center"], "outer_fold": fold, "channel_name": name,
                   "label_nez": int(record["labels_nez"][idx]), "label_ez": int(record["labels_ez"][idx]),
                   "direct_nez_logit": float(record["direct_nez_logit"][idx]), "scope_boundary_delta": float(record["scope_boundary_delta"][idx]),
                   "scope_nez_logit": float(record["scope_nez_logit"][idx]), "scope_score_nez": float(record["scope_score_nez"][idx]),
                   "scope_score_ez": float(record["scope_score_ez"][idx]), "scope_ez_score": float(record["scope_ez_score"][idx]),
                   "scope_ez_rank": int(np.sum(np.asarray(record["scope_ez_score"])[valid] > record["scope_ez_score"][idx]) + 1),
                   "patient_relative_direct_rank": float(record["patient_relative_direct_rank"][idx]),
                   "predicted_nez": int(record["predicted_nez_mask"][idx]), "predicted_ez": int(record["predicted_ez_mask"][idx]),
                   "decision_rule": PREDICTED_K_DECISION_RULE, "formal_prediction_source": FORMAL_PREDICTION_SOURCE,
                   "true_count_used_for_prediction": False, "best_epoch": best_epochs[fold], "fold_seed": fold_seeds[fold],
                   "validation_ledger_hash": ledger_hash, "config_hash": config_hash}
            row.update({field: record.get(field) for field in fields})
            rows.append(row)
    return rows


def run_p2_scope_v2(exp: Any):
    from exp_ez_hybrid import _summarize_prediction_records

    args = exp.args
    ledger_path = Path(args.scope_validation_ledger_path)
    if not ledger_path.is_file():
        raise FileNotFoundError(f"SCOPE fixed validation ledger missing: {ledger_path}")
    ledger_rows, ledger_hash = _read_ledger(ledger_path, exp.outer_splits)
    output = Path(args.output_dir); output.mkdir(parents=True, exist_ok=True)
    config_hash = _stable_json_hash(vars(args)); min_k, require_mixed_channels, decoder_contract = _global_decoder_bounds(exp.patient_index)
    all_records: list[dict[str, Any]] = []; fold_rows: list[dict[str, Any]] = []; history_rows: list[dict[str, Any]] = []
    best_epochs: dict[int, int] = {}; fold_seeds: dict[int, int] = {}
    splits = list(exp.outer_splits)[:int(args.max_outer_folds) or None]
    for split in splits:
        fold = int(split["fold_idx"]); fold_dir = output / f"fold_{fold}"; fold_dir.mkdir(parents=True, exist_ok=True)
        outer_hash = _stable_json_hash({"fold": fold, "train": sorted(split["train_subjects"]), "test": sorted(split["test_subjects"])})
        completion = fold_dir / "completion.json"; records_path = fold_dir / "test_records.pt"
        if bool(getattr(args, "skip_existing", False)) and _completion_matches(completion, config_hash=config_hash, ledger_hash=ledger_hash, outer_hash=outer_hash):
            saved = torch.load(records_path, map_location="cpu", weights_only=False)
            all_records.extend(saved["records"]); best_epochs[fold] = int(saved["best_epoch"]); fold_seeds[fold] = int(saved["fold_seed"])
            fold_rows.append(saved["summary"]); continue
        partition = {role: sorted(row["subject_id"] for row in ledger_rows if int(row["outer_fold"]) == fold and row["role"] == role) for role in ("fit", "validation", "test")}
        fold_seed = set_scope_fold_seed(int(args.model_seed), fold); fold_seeds[fold] = fold_seed
        fit, validation, test, normalizer = exp._build_datasets(partition["fit"], partition["validation"], partition["test"])
        train_loader = _loader(exp, fit, train=True, seed=fold_seed); validation_loader = _loader(exp, validation, train=False, seed=fold_seed)
        model = exp.runtime["model_cls"](args).to(exp.device)
        model.scope_fit_global_ez_fraction_prior.fill_(_prior(exp.patient_index, partition["fit"]))
        exp._dry_initialize_lazy_layers(model, train_loader)
        optimizer = _optimizer(model, args); best_key = None; best_epoch = 0; stale = 0; history: list[dict[str, Any]] = []
        best_path = fold_dir / "best_model.pt"
        exp._log(f"[P2-SCOPE-v2] fold {fold}/5 start | fit={len(partition['fit'])} validation={len(partition['validation'])} test={len(partition['test'])} | inner=disabled | seed={fold_seed}")
        for epoch in range(1, int(args.epochs) + 1):
            exp.current_epoch = epoch; backbone_lr, head_lr = _apply_p2_lr_schedule(optimizer, args, epoch)
            model.train(); losses: list[float] = []
            for batch in train_loader:
                device_batch = exp._prepare_quality_batch(_move(batch, exp.device), split_name="train")
                optimizer.zero_grad(set_to_none=True); loss, _ = exp._compute_loss(model(device_batch), device_batch, torch.tensor(1.0, device=exp.device), split_name="train")
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip)); optimizer.step(); losses.append(float(loss.detach().cpu()))
            if epoch == 5:
                initialize_clean_nez_prototypes(model, train_loader, exp.device, fold_seed)
            validation_raw = collect_ranking_records(exp, model, validation_loader, source=f"scope_fold{fold}_validation")
            validation_records = _decode_records(validation_raw, min_k=min_k, require_mixed_channels=require_mixed_channels, outer_fold=fold)
            validation_summary, _ = _summarize_prediction_records(validation_records)
            row = {"outer_fold": fold, "epoch": epoch, "training_protocol": "single_fixed_validation", "train_loss": float(np.mean(losses) if losses else 0.0),
                   "backbone_lr": backbone_lr, "head_lr": head_lr,
                   "validation_legal_predicted_k_macro_f1": float(validation_summary["patient_macro_f1"]),
                   "validation_legal_predicted_k_ez_f1": float(validation_summary["patient_macro_ez_f1"]),
                   "validation_legal_predicted_k_nez_f1": float(validation_summary["patient_macro_nez_f1"]),
                   "validation_legal_predicted_k_balanced_accuracy": float(validation_summary["patient_macro_balanced_accuracy"]),
                   "validation_ez_auprc": float(validation_summary["patient_macro_auprc_ez"]),
                   "validation_ez_mrr": float(validation_summary["patient_macro_ez_mrr"]),
                   "validation_count_mae": float(np.mean([abs(int((np.asarray(r['labels_ez'])[np.asarray(r['channel_mask'], bool)] > .5).sum()) - int(r['scope_predicted_k'])) for r in validation_records])),
                   "validation_fraction_mae": float(np.mean([abs(float((np.asarray(r['labels_ez'])[np.asarray(r['channel_mask'], bool)] > .5).mean()) - float(r['scope_predicted_ez_fraction'])) for r in validation_records])),
                   "validation_true_k_topk_diagnostic_macro_f1": _true_k_diagnostic_macro_f1(validation_records)}
            history.append(row); history_rows.append(row)
            if len(history) >= 3:
                row.update({f"rolling_{key}": _rolling(history, key) for key in ("validation_legal_predicted_k_macro_f1", "validation_true_k_topk_diagnostic_macro_f1", "validation_legal_predicted_k_ez_f1", "validation_ez_auprc", "validation_ez_mrr", "validation_fraction_mae")})
            eligible = epoch >= int(args.scope_checkpoint_min_epoch) and len(history) >= 3
            row["checkpoint_eligible"] = eligible
            if eligible:
                key = (row["rolling_validation_legal_predicted_k_macro_f1"], row["rolling_validation_true_k_topk_diagnostic_macro_f1"], row["rolling_validation_legal_predicted_k_ez_f1"], row["rolling_validation_ez_auprc"], row["rolling_validation_ez_mrr"], -row["rolling_validation_fraction_mae"], -epoch)
                if best_key is None or key > best_key:
                    best_key, best_epoch, stale = key, epoch, 0
                    torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "best_epoch": best_epoch, "best_selection_key": key, "validation_summary": dict(row), "scope_fit_global_ez_fraction_prior": float(model.scope_fit_global_ez_fraction_prior.item()), "normalizer": getattr(normalizer, "__dict__", normalizer), "fold_seed": fold_seed, "dataloader_seed": fold_seed, "validation_ledger_hash": ledger_hash, "outer_fold_hash": outer_hash, "config_hash": config_hash, "label_semantics": "1=NEZ,0=EZ", "score_semantics": "P(NEZ)", "decision_rule": PREDICTED_K_DECISION_RULE, "formal_prediction_source": FORMAL_PREDICTION_SOURCE, "true_count_used_for_prediction": False}, best_path)
                else:
                    stale += 1
            exp._log(f"[P2-SCOPE-v2] fold {fold} epoch {epoch}/{args.epochs} | loss={row['train_loss']:.4f} | val_predicted_k_f1={row['validation_legal_predicted_k_macro_f1']:.4f} | backbone_lr={backbone_lr:.2e} | head_lr={head_lr:.2e}")
            if eligible and stale >= int(args.patience):
                exp._log(f"[P2-SCOPE-v2] fold {fold} early stop at epoch {epoch}"); break
        if best_epoch < int(args.scope_checkpoint_min_epoch) or not best_path.is_file():
            raise RuntimeError("SCOPE failed to save an eligible best checkpoint")
        saved_checkpoint = torch.load(best_path, map_location=exp.device, weights_only=False)
        model.load_state_dict(saved_checkpoint["model_state_dict"])
        test_raw = collect_ranking_records(exp, model, _loader(exp, test, train=False, seed=fold_seed), source=f"scope_fold{fold}_test")
        test_records = _decode_records(test_raw, min_k=min_k, require_mixed_channels=require_mixed_channels, outer_fold=fold)
        summary, enriched = _summarize_prediction_records(test_records)
        summary.update({"outer_fold": fold, "n_fit_patients": len(partition["fit"]), "n_validation_patients": len(partition["validation"]), "n_test_patients": len(partition["test"]), "best_epoch": best_epoch, "fold_seed": fold_seed, "training_mode": "single_fixed_validation", "inner_crossfit_used": False, "decision_rule": PREDICTED_K_DECISION_RULE, "formal_prediction_source": FORMAL_PREDICTION_SOURCE, "true_count_used_for_prediction": False, "validation_ledger_hash": ledger_hash})
        all_records.extend(enriched); fold_rows.append(summary); best_epochs[fold] = best_epoch
        torch.save({"records": enriched, "summary": summary, "best_epoch": best_epoch, "fold_seed": fold_seed}, records_path)
        pd.DataFrame([summary]).to_csv(fold_dir / "fold_summary.csv", index=False); pd.DataFrame(history).to_csv(fold_dir / "train_history.csv", index=False)
        completion.write_text(json.dumps({"status": "complete", "outer_fold": fold, "config_hash": config_hash, "validation_ledger_hash": ledger_hash, "outer_fold_hash": outer_hash, "best_checkpoint_path": str(best_path), "n_test_patients": len(partition["test"]), "decision_rule": PREDICTED_K_DECISION_RULE, "formal_prediction_source": FORMAL_PREDICTION_SOURCE, "true_count_used_for_prediction": False}, indent=2), encoding="utf-8")
    if not all_records:
        raise RuntimeError("SCOPE produced no completed test records")
    overall, _ = _summarize_prediction_records(all_records)
    overall.update({"training_mode": "single_fixed_validation", "inner_crossfit_used": False, "decision_rule": PREDICTED_K_DECISION_RULE, "formal_prediction_source": FORMAL_PREDICTION_SOURCE, "true_count_used_for_prediction": False, "validation_ledger_hash": ledger_hash, "config_hash": config_hash})
    patient_rows = _patient_rows(all_records, best_epochs, fold_seeds); channel_rows = _channel_rows(all_records, best_epochs, fold_seeds, ledger_hash, config_hash)
    patient_frame = pd.DataFrame(patient_rows)
    center_rows = []
    for center, group in patient_frame.groupby("center"):
        center_rows.append({"center": center, "n_patients": len(group), "patient_macro_f1": group.patient_macro_f1.mean(), "patient_ez_f1": group.patient_ez_f1.mean(), "patient_nez_f1": group.patient_nez_f1.mean(), "patient_balanced_accuracy": group.patient_balanced_accuracy.mean(), "patient_macro_ez_auprc": group.ez_auprc.mean(), "patient_macro_ez_mrr": group.ez_mrr.mean(), "count_mae": group.count_absolute_error.mean(), "fraction_mae": group.fraction_absolute_error.mean()})
    pd.DataFrame([overall]).to_csv(output / "p2_scope_v2_overall.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(output / "p2_scope_v2_by_fold.csv", index=False)
    pd.DataFrame(history_rows).to_csv(output / "p2_scope_v2_train_history.csv", index=False)
    pd.DataFrame(channel_rows).to_csv(output / "p2_scope_v2_channel_predictions.csv", index=False)
    patient_frame.to_csv(output / "p2_scope_v2_by_patient.csv", index=False)
    pd.DataFrame(center_rows).to_csv(output / "p2_scope_v2_by_center.csv", index=False)
    (output / "protocol_audit.json").write_text(json.dumps({"method": "P2_SCOPE_V2_FIXED", "inner_crossfit_used": False, "formal_prediction_source": FORMAL_PREDICTION_SOURCE, "outer_test_true_k_used_for_prediction": False, "validation_ledger": str(ledger_path), "validation_ledger_hash": ledger_hash, "decoder_contract": decoder_contract}, indent=2), encoding="utf-8")
    return all_records
