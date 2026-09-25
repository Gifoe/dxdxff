from __future__ import annotations

import csv
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .config import BNPDGSConfig
from .data_interface import load_patient_records
from .dynamic_dataset import (
    DynamicPatientSample,
    build_dynamic_patient_samples,
    collate_dynamic_patient_batch,
    estimate_positive_class_weight,
)
from .evaluate import (
    DEFAULT_SELECTION_MODE,
    apply_selection_mode,
    calibrate_selection,
    move_batch_to_device,
    predict_batches,
    site_stratified_summaries,
    summarize_records,
)
from .losses import bn_pdgs_loss
from .logging_utils import log
from .model import BNPDGSModel
from .report import write_patient_reports, write_summary


class DynamicPatientDataset(Dataset):
    def __init__(self, samples: Sequence[DynamicPatientSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> DynamicPatientSample:
        return self.samples[index]


def run_training(args: Any | None = None) -> dict[str, Any]:
    cfg = BNPDGSConfig.from_args(args)
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Run started. output_dir={cfg.output_dir}")
    patient_records = load_patient_records(args)
    if not patient_records:
        raise ValueError("No PatientRecord objects were loaded. Check dataset paths and annotations.")
    dynamic_samples = build_dynamic_patient_samples(patient_records, cfg)
    if not dynamic_samples:
        raise ValueError("No dynamic patient samples were built. Check peri-onset coverage and channel labels.")
    return run_training_from_samples(
        run_name="single_run",
        samples=dynamic_samples,
        cfg=cfg,
        output_dir=cfg.output_dir,
        n_patients_loaded=len(patient_records),
    )


def run_training_from_samples(
    run_name: str,
    samples: Sequence[DynamicPatientSample],
    cfg: BNPDGSConfig,
    output_dir: str | Path,
    n_patients_loaded: int | None = None,
    splits: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    output_dir = Path(output_dir)
    for subdir in ["logs", "checkpoints", "predictions", "reports", "reports/patient_reports", "folds"]:
        (output_dir / subdir).mkdir(parents=True, exist_ok=True)

    samples = list(samples)
    if not samples:
        summary = {
            "run_name": run_name,
            "status": "insufficient_patients",
            "reason": "no_cached_samples",
            "n_patients_loaded": int(n_patients_loaded or 0),
            "n_dynamic_patients": 0,
            "n_folds": 0,
            "folds": [],
        }
        write_summary(summary, output_dir / "reports" / "summary.json")
        return summary

    if splits is None:
        splits = build_sample_splits(samples, seed=cfg.seed)
    if not splits:
        summary = {
            "run_name": run_name,
            "status": "insufficient_patients",
            "reason": "fewer_than_10_patients",
            "n_patients_loaded": int(n_patients_loaded or len(samples)),
            "n_dynamic_patients": len(samples),
            "n_folds": 0,
            "folds": [],
        }
        write_summary(summary, output_dir / "reports" / "summary.json")
        return summary

    sample_by_subject = {sample.subject_id: sample for sample in samples}
    fold_summaries = []
    for fold in splits:
        fold_idx = int(fold.get("fold_idx", 0))
        write_summary(fold, output_dir / "folds" / f"fold{fold_idx}_split.json")
        fold_summary = fit_fold_cached(run_name, fold, sample_by_subject, cfg, output_dir)
        fold_summaries.append(fold_summary)

    summary = summarize_fold_summaries(run_name, fold_summaries, len(samples), int(n_patients_loaded or len(samples)))
    summary["status"] = "completed"
    summary["config"] = cfg.to_dict()
    write_summary(summary, output_dir / "reports" / "summary.json")
    _write_fold_summary_csv(fold_summaries, output_dir / "reports" / "fold_summary.csv")
    return summary


def build_sample_splits(samples: Sequence[DynamicPatientSample], seed: int = 42) -> list[dict[str, Any]]:
    subjects = [sample.subject_id for sample in samples]
    n = len(subjects)
    if n < 10:
        return []
    n_splits = 5 if n >= 20 else 3
    rng = np.random.default_rng(int(seed))
    subjects = [str(subject) for subject in rng.permutation(np.asarray(subjects, dtype=object)).tolist()]
    chunks = np.array_split(np.asarray(subjects, dtype=object), n_splits)
    folds = []
    for fold_idx in range(n_splits):
        test_subjects = [str(item) for item in chunks[fold_idx].tolist()]
        val_subjects = [str(item) for item in chunks[(fold_idx + 1) % n_splits].tolist()]
        blocked = set(test_subjects) | set(val_subjects)
        train_subjects = [subject for subject in subjects if subject not in blocked]
        folds.append(
            {
                "fold_idx": fold_idx,
                "train_subjects": train_subjects,
                "val_subjects": val_subjects,
                "test_subjects": test_subjects,
            }
        )
    return folds


def fit_fold_cached(
    run_name: str,
    fold: dict[str, Any],
    sample_by_subject: dict[str, DynamicPatientSample],
    cfg: BNPDGSConfig,
    output_dir: Path,
) -> dict[str, Any]:
    fold_idx = int(fold.get("fold_idx", 0))
    train_samples = _select_samples(sample_by_subject, fold.get("train_subjects", []))
    val_samples = _select_samples(sample_by_subject, fold.get("val_subjects", []))
    test_samples = _select_samples(sample_by_subject, fold.get("test_subjects", []))
    if not train_samples or not val_samples or not test_samples:
        raise ValueError(f"Fold {fold_idx} has empty train/val/test samples.")

    train_loader = _make_loader(train_samples, cfg, shuffle=True)
    val_loader = _make_loader(val_samples, cfg, shuffle=False)
    test_loader = _make_loader(test_samples, cfg, shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BNPDGSModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = estimate_positive_class_weight(train_samples, cap=cfg.pos_weight_max) if cfg.use_positive_weight else 1.0

    best_score = -float("inf")
    best_state = None
    best_val_summary: dict[str, Any] | None = None
    best_calibration: dict[str, float] | None = None
    bad_epochs = 0
    metrics_path = output_dir / "logs" / "metrics.jsonl"

    for epoch in range(1, int(cfg.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, pos_weight, device)
        val_raw = predict_batches(model, val_loader, device)
        calibration = calibrate_selection(val_raw)
        val_records = apply_selection_mode(val_raw, DEFAULT_SELECTION_MODE, calibration)
        val_summary = summarize_records(val_records)
        val_summary.update({"epoch": epoch, "train_loss": train_loss, "fold_idx": fold_idx})
        _append_jsonl(metrics_path, {"run_name": run_name, **val_summary})
        score = float(val_summary["selection_score"])
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_val_summary = val_summary
            best_calibration = calibration
            bad_epochs = 0
        else:
            bad_epochs += 1
        if epoch >= int(cfg.min_epochs) and bad_epochs >= int(cfg.patience):
            log(f"{run_name} fold={fold_idx}: early stopping at epoch={epoch}, bad_epochs={bad_epochs}")
            break
        if epoch == 1 or epoch % 5 == 0:
            log(
                f"{run_name} fold={fold_idx} epoch={epoch}: loss={train_loss:.4f}, "
                f"val_selection={score:.4f}, val_AUC_PR={val_summary.get('macro_AUC_PR', float('nan')):.4f}, "
                f"best={best_score:.4f}"
            )

    torch.save(
        {"model_state": model.state_dict(), "config": cfg.to_dict(), "fold": fold, "epoch": epoch},
        output_dir / "checkpoints" / f"last_fold{fold_idx}.pt",
    )
    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "fold": fold,
            "pos_weight": pos_weight,
            "best_val_summary": best_val_summary,
            "selection_calibration": best_calibration,
        },
        output_dir / "checkpoints" / f"best_fold{fold_idx}.pt",
    )

    test_raw = predict_batches(model, test_loader, device)
    calibration = best_calibration or calibrate_selection(predict_batches(model, val_loader, device))
    test_records = apply_selection_mode(test_raw, DEFAULT_SELECTION_MODE, calibration)
    test_summary = summarize_records(test_records)
    test_summary.update(site_stratified_summaries(test_records))
    write_summary(best_val_summary or {}, output_dir / "reports" / f"fold{fold_idx}_val_summary.json")
    write_summary(test_summary, output_dir / "reports" / f"fold{fold_idx}_test_summary.json")
    write_patient_reports(test_records, output_dir / "reports")
    _write_predictions(test_records, output_dir / "predictions" / f"fold{fold_idx}_channel_scores.csv")
    _write_patient_predictions(test_records, output_dir / "predictions" / f"fold{fold_idx}_patient_predictions.csv")

    return {
        "fold_idx": fold_idx,
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "pos_weight": float(pos_weight),
        "selection_calibration": calibration,
        "best_val_summary": best_val_summary or {},
        "test_summary": test_summary,
    }


def train_one_epoch(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    cfg: BNPDGSConfig,
    pos_weight: float,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    total_batches = 0
    for batch in dataloader:
        batch = move_batch_to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(batch)
        loss, _ = bn_pdgs_loss(outputs, batch, cfg, pos_weight=pos_weight)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(cfg.grad_clip_norm))
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
    return total_loss / max(total_batches, 1)


def summarize_fold_summaries(
    run_name: str,
    fold_summaries: Sequence[dict[str, Any]],
    n_dynamic_patients: int,
    n_patients_loaded: int,
) -> dict[str, Any]:
    test_summaries = [fold.get("test_summary", {}) for fold in fold_summaries]
    fields = [
        "macro_AUC",
        "macro_AUC_PR",
        "macro_F1",
        "macro_PREC",
        "macro_REC",
        "macro_TOPK_RECALL",
        "macro_TOP1_HIT",
        "macro_TOP3_HIT",
        "macro_COUNT_BIAS",
        "macro_ABS_COUNT_BIAS_RATIO",
        "selection_score",
    ]
    result = {
        "run_name": run_name,
        "n_patients_loaded": int(n_patients_loaded),
        "n_dynamic_patients": int(n_dynamic_patients),
        "n_folds": len(fold_summaries),
        "folds": list(fold_summaries),
    }
    for field in fields:
        values = np.asarray([summary.get(field, np.nan) for summary in test_summaries], dtype=np.float32)
        result[f"mean_{field}"] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    site_keys = sorted({key for summary in test_summaries for key in summary if key.startswith("site_")})
    for site_key in site_keys:
        site_items = [summary.get(site_key, {}) for summary in test_summaries if isinstance(summary.get(site_key), dict)]
        result[site_key] = {}
        for field in fields:
            values = np.asarray([item.get(field, np.nan) for item in site_items], dtype=np.float32)
            result[site_key][f"mean_{field}"] = float(np.nanmean(values)) if np.any(np.isfinite(values)) else float("nan")
    return result


def _make_loader(samples: Sequence[DynamicPatientSample], cfg: BNPDGSConfig, shuffle: bool) -> DataLoader:
    return DataLoader(
        DynamicPatientDataset(samples),
        batch_size=int(cfg.batch_size),
        shuffle=bool(shuffle),
        num_workers=0,
        collate_fn=collate_dynamic_patient_batch,
    )


def _select_samples(
    sample_by_subject: dict[str, DynamicPatientSample],
    subjects: Sequence[str],
) -> list[DynamicPatientSample]:
    return [sample_by_subject[subject] for subject in subjects if subject in sample_by_subject]


def _write_predictions(records: Sequence[dict[str, Any]], path: Path) -> None:
    rows = [row for record in records for row in record.get("channel_scores", [])]
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_patient_predictions(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "subject_id",
        "center_id",
        "selection_mode",
        "n_channels",
        "n_true_ez",
        "n_pred_ez",
        "AUC_PR",
        "TOPK_RECALL",
        "F1",
        "COUNT_BIAS",
    ]
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for record in records:
            metrics = record.get("metrics", {})
            writer.writerow(
                {
                    "subject_id": record.get("subject_id"),
                    "center_id": record.get("center_id"),
                    "selection_mode": record.get("selection_mode"),
                    "n_channels": record.get("n_channels"),
                    "n_true_ez": record.get("n_true_ez"),
                    "n_pred_ez": record.get("n_pred_ez"),
                    "AUC_PR": metrics.get("AUC_PR"),
                    "TOPK_RECALL": metrics.get("TOPK_RECALL"),
                    "F1": metrics.get("F1"),
                    "COUNT_BIAS": metrics.get("COUNT_BIAS"),
                }
            )


def _write_fold_summary_csv(fold_summaries: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["fold_idx", "n_train", "n_val", "n_test", "selection_score", "macro_AUC_PR", "macro_TOPK_RECALL", "macro_F1"]
    with open(path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=fields)
        writer.writeheader()
        for fold in fold_summaries:
            summary = fold.get("test_summary", {})
            writer.writerow(
                {
                    "fold_idx": fold.get("fold_idx"),
                    "n_train": fold.get("n_train"),
                    "n_val": fold.get("n_val"),
                    "n_test": fold.get("n_test"),
                    "selection_score": summary.get("selection_score"),
                    "macro_AUC_PR": summary.get("macro_AUC_PR"),
                    "macro_TOPK_RECALL": summary.get("macro_TOPK_RECALL"),
                    "macro_F1": summary.get("macro_F1"),
                }
            )


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fout:
        fout.write(json.dumps(payload, ensure_ascii=False, default=_json_default) + "\n")


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


__all__ = [
    "DynamicPatientDataset",
    "build_sample_splits",
    "fit_fold_cached",
    "run_training",
    "run_training_from_samples",
    "summarize_fold_summaries",
    "train_one_epoch",
]
