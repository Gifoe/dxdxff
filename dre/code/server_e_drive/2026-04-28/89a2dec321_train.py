from __future__ import annotations

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
    build_patient_splits,
    collate_dynamic_patient_batch,
    estimate_positive_class_weight,
)
from .evaluate import move_batch_to_device, predict_batches, summarize_records
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
    log(
        f"Config summary: datasets={cfg.datasets}, success_only={cfg.success_only}, "
        f"epochs={cfg.epochs}, batch_size={cfg.batch_size}, model_dim={cfg.model_dim}, "
        f"feature_num_workers={cfg.feature_num_workers}"
    )

    log("Stage 1/5: loading EDF signals and labels.")
    patient_records = load_patient_records(args)
    if not patient_records:
        raise ValueError("No PatientRecord objects were loaded. Check dataset paths and annotations.")
    log(
        f"Stage 1/5 done: loaded patients={len(patient_records)}, "
        f"seizures={sum(len(patient.seizures) for patient in patient_records)}"
    )

    log("Stage 2/5: building peri-onset dynamic graph-spectral samples.")
    dynamic_samples = build_dynamic_patient_samples(patient_records, cfg)
    if not dynamic_samples:
        raise ValueError("No dynamic patient samples were built. Check peri-onset coverage and channel labels.")
    log(f"Stage 2/5 done: dynamic patients={len(dynamic_samples)}")

    log("Stage 3/5: building patient-wise train/val/test splits.")
    splits = build_patient_splits(patient_records, n_splits=cfg.n_splits, seed=cfg.seed)
    log(f"Built patient-wise splits. folds={len(splits)}")
    sample_by_subject = {sample.subject_id: sample for sample in dynamic_samples}
    fold_summaries = []
    for fold in splits:
        log(
            f"Starting fold {fold.get('fold_idx')}: "
            f"train={len(fold.get('train_subjects', []))}, "
            f"val={len(fold.get('val_subjects', []))}, test={len(fold.get('test_subjects', []))}"
        )
        fold_summary = fit_fold(fold, sample_by_subject, cfg)
        fold_summaries.append(fold_summary)

    global_summary = {
        "config": cfg.to_dict(),
        "n_patients_loaded": len(patient_records),
        "n_dynamic_patients": len(dynamic_samples),
        "folds": fold_summaries,
        "mean_selection_score": float(np.mean([fold["best_val_summary"]["selection_score"] for fold in fold_summaries])),
    }
    write_summary(global_summary, cfg.output_dir / "summary.json")
    log(f"Run finished. summary={cfg.output_dir / 'summary.json'}")
    return global_summary


def fit_fold(
    fold: dict[str, Any],
    sample_by_subject: dict[str, DynamicPatientSample],
    cfg: BNPDGSConfig,
) -> dict[str, Any]:
    train_samples = _select_samples(sample_by_subject, fold.get("train_subjects", []))
    val_samples = _select_samples(sample_by_subject, fold.get("val_subjects", [])) or train_samples
    test_samples = _select_samples(sample_by_subject, fold.get("test_subjects", [])) or val_samples
    if not train_samples:
        raise ValueError(f"Fold {fold.get('fold_idx')} has no train samples.")

    train_loader = _make_loader(train_samples, cfg, shuffle=True)
    val_loader = _make_loader(val_samples, cfg, shuffle=False)
    test_loader = _make_loader(test_samples, cfg, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(
        f"Stage 4/5 fold {fold.get('fold_idx')}: training. "
        f"device={device}, train={len(train_samples)}, val={len(val_samples)}, test={len(test_samples)}"
    )
    model = BNPDGSModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    pos_weight = estimate_positive_class_weight(train_samples, cap=cfg.pos_weight_max) if cfg.use_positive_weight else 1.0
    log(f"Fold {fold.get('fold_idx')}: pos_weight={pos_weight:.4f}")

    best_score = -float("inf")
    best_state = None
    best_val_summary: dict[str, Any] | None = None
    bad_epochs = 0
    for epoch in range(1, int(cfg.epochs) + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, cfg, pos_weight, device)
        val_records = predict_batches(model, val_loader, device)
        val_summary = summarize_records(val_records)
        val_summary["epoch"] = epoch
        val_summary["train_loss"] = train_loss
        score = float(val_summary["selection_score"])
        if score > best_score:
            best_score = score
            best_state = deepcopy(model.state_dict())
            best_val_summary = val_summary
            bad_epochs = 0
        else:
            bad_epochs += 1
        if bad_epochs >= int(cfg.patience):
            log(f"Fold {fold.get('fold_idx')}: early stopping at epoch={epoch}, bad_epochs={bad_epochs}")
            break
        if epoch == 1 or epoch % 5 == 0:
            log(
                f"Fold {fold.get('fold_idx')} epoch={epoch}: train_loss={train_loss:.4f}, "
                f"val_selection={score:.4f}, val_AUC_PR={val_summary.get('macro_AUC_PR', float('nan')):.4f}, "
                f"best={best_score:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
    test_records = predict_batches(model, test_loader, device)
    test_summary = summarize_records(test_records)
    log(f"Stage 5/5 fold {fold.get('fold_idx')}: writing test reports.")

    fold_dir = cfg.output_dir / f"fold_{int(fold.get('fold_idx', 0))}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": cfg.to_dict(),
            "fold": fold,
            "pos_weight": pos_weight,
            "best_val_summary": best_val_summary,
            "test_summary": test_summary,
        },
        fold_dir / "model.pt",
    )
    write_summary(best_val_summary or {}, fold_dir / "val_summary.json")
    write_summary(test_summary, fold_dir / "test_summary.json")
    write_patient_reports(test_records, fold_dir)
    log(
        f"Fold {fold.get('fold_idx')}: test_selection={test_summary.get('selection_score', float('nan')):.4f}, "
        f"test_AUC_PR={test_summary.get('macro_AUC_PR', float('nan')):.4f}, output={fold_dir}"
    )
    return {
        "fold_idx": int(fold.get("fold_idx", 0)),
        "n_train": len(train_samples),
        "n_val": len(val_samples),
        "n_test": len(test_samples),
        "pos_weight": float(pos_weight),
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_batches += 1
        if total_batches == 1:
            log(
                f"First train batch: features={tuple(batch['features'].shape)}, "
                f"channels={tuple(batch['channel_mask'].shape)}, seizures={tuple(batch['seizure_mask'].shape)}"
            )
    return total_loss / max(total_batches, 1)


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


__all__ = [
    "DynamicPatientDataset",
    "fit_fold",
    "run_training",
    "train_one_epoch",
]
