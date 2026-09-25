from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, Subset

from task1_baselines.prediction_aggregation import aggregate_task1_window_probabilities
from task1_baselines.thresholds import select_patient_macro_threshold
from .model import SEEGformerTask1
from .multichannel_data import Task1MultichannelDataset, collate_multichannel_windows


@dataclass
class SEEGformerOOFResult:
    oof: pd.DataFrame
    training_audit: pd.DataFrame
    fit_subjects: dict[int, tuple[str, ...]]
    checkpoint_paths: dict[int, str]


def _loader(dataset: Task1MultichannelDataset, indices: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(Subset(dataset, indices.tolist()), batch_size=min(batch_size, len(indices)), shuffle=shuffle, generator=generator, num_workers=0, collate_fn=collate_multichannel_windows)


def _loss(logits: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor, pos_weight: torch.Tensor) -> torch.Tensor:
    values = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, pos_weight=pos_weight, reduction="none")
    return (values * mask.float()).sum() / mask.float().sum().clamp_min(1.0)


def _evaluate(model: SEEGformerTask1, loader: DataLoader, device: torch.device, pos_weight: torch.Tensor, collect: bool = False) -> tuple[float, pd.DataFrame]:
    model.eval(); losses: list[float] = []; rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for batch in loader:
            mask = batch["channel_mask"].to(device)
            labels = batch["label_nez"].to(device)
            logits = model(batch["waveform"].to(device), mask)["channel_logits"]
            losses.append(float(_loss(logits, labels, mask, pos_weight).item()))
            if collect:
                probability = torch.sigmoid(logits).cpu().numpy()
                valid = batch["channel_mask"].numpy()
                target = batch["label_nez"].numpy()
                for item_index, names in enumerate(batch["channel_names"]):
                    for channel_index, channel in enumerate(names):
                        if valid[item_index, channel_index]:
                            rows.append({"subject_id": batch["subject_ids"][item_index], "center": batch["centers"][item_index], "modality": batch["modalities"][item_index], "seizure_id": batch["seizure_ids"][item_index], "window_id": batch["window_ids"][item_index], "channel_name": channel, "label_nez": int(target[item_index, channel_index]), "score_nez_probability": float(probability[item_index, channel_index])})
    return float(np.mean(losses)) if losses else float("nan"), pd.DataFrame(rows)


def _fit(dataset: Task1MultichannelDataset, fit_indices: np.ndarray, valid_indices: np.ndarray | None, *, seed: int, max_epochs: int, batch_size: int, grad_accum_steps: int, learning_rate: float, patience: int, device: torch.device, amp: bool, model_config: dict[str, Any]) -> tuple[SEEGformerTask1, int, float, list[dict[str, Any]]]:
    torch.manual_seed(seed); np.random.seed(seed)
    model = SEEGformerTask1(**model_config).to(device)
    labels = np.concatenate([dataset.items[index].label_nez for index in fit_indices])
    pos_weight = torch.tensor(float((labels == 0).sum()) / max(float((labels == 1).sum()), 1.0), device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    train_loader = _loader(dataset, fit_indices, batch_size, True, seed)
    valid_loader = _loader(dataset, valid_indices, batch_size, False, seed) if valid_indices is not None and len(valid_indices) else None
    scaler = torch.amp.GradScaler("cuda", enabled=amp and device.type == "cuda")
    best_state = None; best_loss = float("inf"); best_epoch = 1; stale = 0; history: list[dict[str, Any]] = []
    for epoch in range(1, max_epochs + 1):
        model.train(); optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            mask = batch["channel_mask"].to(device); target = batch["label_nez"].to(device)
            with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
                logits = model(batch["waveform"].to(device), mask)["channel_logits"]
                loss = _loss(logits, target, mask, pos_weight) / grad_accum_steps
            scaler.scale(loss).backward()
            if step % grad_accum_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True)
        if valid_loader is None:
            history.append({"epoch": epoch, "validation_loss": float("nan")})
            continue
        validation_loss, _ = _evaluate(model, valid_loader, device, pos_weight)
        history.append({"epoch": epoch, "validation_loss": validation_loss})
        if validation_loss < best_loss:
            best_loss, best_epoch, stale = validation_loss, epoch, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale += 1
            if stale >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch, best_loss, history


def run_task1_seegformer_oof(dataset: Task1MultichannelDataset, fold_manifest: pd.DataFrame, *, model_name: str, seed: int, inner_folds: int = 4, max_epochs: int = 30, batch_size: int = 8, device: str = "cpu", amp: bool = False, grad_accum_steps: int = 2, max_outer_folds: int = 0, model_config: dict[str, Any] | None = None, training_config: dict[str, Any] | None = None, checkpoint_root: str | Path | None = None, save_attention: bool = False, progress: Any | None = None) -> SEEGformerOOFResult:
    metadata = dataset.rows.merge(fold_manifest[["subject_id", "outer_fold"]], on="subject_id", how="inner", validate="many_to_one")
    if len(metadata) != len(dataset) or set(metadata["subject_id"]) != set(fold_manifest["subject_id"]):
        raise ValueError("Multichannel dataset does not exactly match the frozen fold cohort.")
    active = torch.device(device)
    config = {"sfreq": 200.0, "n_times": 800, "n_fft": 1600, "freq_min": 0.5, "freq_max": 80.0, "embed_dim": 64, "num_heads": 4, "num_blocks": 2, "mlp_ratio": 2.0, "dropout": 0.2, **dict(model_config or {})}
    training = {"learning_rate": 5e-4, "patience": 6, **dict(training_config or {})}
    folds = sorted(metadata["outer_fold"].unique())[: max_outer_folds or None]
    oof: list[pd.DataFrame] = []; audits: list[dict[str, Any]] = []; subjects: dict[int, tuple[str, ...]] = {}; checkpoints: dict[int, str] = {}
    for fold in folds:
        if progress:
            progress(f"START outer_fold={fold}/{len(folds)} model={model_name} seed={seed}")
        train_indices = np.flatnonzero(metadata["outer_fold"].to_numpy() != fold); test_indices = np.flatnonzero(metadata["outer_fold"].to_numpy() == fold)
        train_subjects = metadata.iloc[train_indices]["subject_id"].to_numpy(); subjects[int(fold)] = tuple(sorted(set(train_subjects)))
        splitter = GroupKFold(n_splits=min(inner_folds, len(set(train_subjects))))
        inner_rows: list[pd.DataFrame] = []; selected_epochs: list[int] = []
        for inner_fold, (fit_local, valid_local) in enumerate(splitter.split(train_indices, groups=train_subjects), start=1):
            model, epoch, _, history = _fit(dataset, train_indices[fit_local], train_indices[valid_local], seed=seed + inner_fold, max_epochs=max_epochs, batch_size=batch_size, grad_accum_steps=grad_accum_steps, learning_rate=float(training["learning_rate"]), patience=int(training["patience"]), device=active, amp=amp, model_config=config)
            for row in history:
                audits.append({"record_type": "inner_epoch", "outer_fold": fold, "inner_fold": inner_fold, "seed": seed, **row})
            _, windows = _evaluate(model, _loader(dataset, train_indices[valid_local], batch_size, False, seed), active, torch.tensor(1.0, device=active), collect=True)
            inner_rows.append(windows); selected_epochs.append(epoch)
        threshold = select_patient_macro_threshold(aggregate_task1_window_probabilities(pd.concat(inner_rows, ignore_index=True)))
        final_epochs = max(1, int(round(median(selected_epochs))))
        model, _, _, history = _fit(dataset, train_indices, None, seed=seed, max_epochs=final_epochs, batch_size=batch_size, grad_accum_steps=grad_accum_steps, learning_rate=float(training["learning_rate"]), patience=int(training["patience"]), device=active, amp=amp, model_config=config)
        for row in history:
            audits.append({"record_type": "final_epoch", "outer_fold": fold, "inner_fold": pd.NA, "seed": seed, **row})
        checkpoint_path = ""
        if checkpoint_root is not None:
            location = Path(checkpoint_root) / f"outer_fold_{fold}" / f"seed_{seed}"; location.mkdir(parents=True, exist_ok=True)
            checkpoint_path = str(location / "best_model.pt")
            torch.save({"state_dict": model.state_dict(), "config": config, "seed": seed, "outer_fold": fold}, checkpoint_path)
            checkpoints[int(fold)] = checkpoint_path
        if save_attention and checkpoint_root is not None:
            diagnostic_loader = _loader(dataset, test_indices, batch_size, False, seed)
            diagnostic_batch = next(iter(diagnostic_loader))
            with torch.no_grad():
                diagnostic = model(diagnostic_batch["waveform"].to(active), diagnostic_batch["channel_mask"].to(active))
            attention_path = Path(checkpoint_root) / f"outer_fold_{fold}" / f"seed_{seed}" / "attention_diagnostic.npz"
            np.savez_compressed(
                attention_path,
                channel_mask=diagnostic_batch["channel_mask"].numpy(),
                real=np.asarray(diagnostic["real_attention"][-1].detach().cpu()),
                imag=np.asarray(diagnostic["imag_attention"][-1].detach().cpu()),
                amplitude=np.asarray(diagnostic["amplitude_attention"][-1].detach().cpu()),
            )
        _, windows = _evaluate(model, _loader(dataset, test_indices, batch_size, False, seed), active, torch.tensor(1.0, device=active), collect=True)
        channel = aggregate_task1_window_probabilities(windows)
        modality = windows.groupby(["subject_id", "channel_name"], as_index=False)["modality"].first()
        channel = channel.merge(modality, on=["subject_id", "channel_name"], validate="one_to_one")
        channel["outer_fold"] = fold; channel["model"] = model_name; channel["seed"] = seed; channel["selected_threshold"] = threshold.threshold; channel["threshold_source"] = threshold.source; channel["checkpoint_path"] = checkpoint_path; channel["predicted_nez"] = (channel["score_nez_probability"] >= threshold.threshold).astype(int); channel["predicted_ez"] = 1 - channel["predicted_nez"]; channel["clinical_true_nez"] = channel["label_nez"]; channel["clinical_true_ez"] = 1 - channel["label_nez"]
        oof.append(channel); audits.append({"record_type": "outer_fold", "outer_fold": fold, "seed": seed, "selected_epochs": final_epochs, "selected_threshold": threshold.threshold, "inner_folds": len(selected_epochs), "train_subject_count": len(set(train_subjects)), "test_subject_count": len(set(metadata.iloc[test_indices]["subject_id"])), "checkpoint_path": checkpoint_path})
        if progress:
            progress(f"DONE outer_fold={fold}/{len(folds)} model={model_name} seed={seed} oof_rows={len(channel)}")
    return SEEGformerOOFResult(pd.concat(oof, ignore_index=True).sort_values(["subject_id", "channel_name"]).reset_index(drop=True), pd.DataFrame(audits), subjects, checkpoints)
