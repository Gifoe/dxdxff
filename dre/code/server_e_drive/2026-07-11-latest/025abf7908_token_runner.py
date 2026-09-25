from __future__ import annotations

from dataclasses import dataclass
from statistics import median
import time
from typing import Callable

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader, TensorDataset

from task1_baselines.prediction_aggregation import aggregate_task1_window_probabilities
from task1_baselines.thresholds import Task1Threshold, select_patient_macro_threshold
from task1_baselines.token_data import Task1TokenDataset


class TokenClassifier(torch.nn.Module):
    def __init__(self, encoder: torch.nn.Module, *, frozen_backbone: bool) -> None:
        super().__init__()
        if not hasattr(encoder, "output_dim"):
            raise ValueError("Token encoder must expose output_dim.")
        self.encoder = encoder
        self.frozen_backbone = bool(frozen_backbone)
        self.head = torch.nn.Sequential(
            torch.nn.LayerNorm(int(encoder.output_dim)),
            torch.nn.Linear(int(encoder.output_dim), 128),
            torch.nn.GELU(), torch.nn.Dropout(0.2), torch.nn.Linear(128, 1),
        )
        if self.frozen_backbone:
            self.encoder.eval()
            for parameter in self.encoder.parameters():
                parameter.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.frozen_backbone:
            self.encoder.eval()
        return self

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        if self.frozen_backbone:
            with torch.no_grad():
                embedding = self.encoder(values)
        else:
            embedding = self.encoder(values)
        return self.head(embedding).squeeze(-1)


@dataclass
class TokenOOFResult:
    oof: pd.DataFrame
    fit_subjects: dict[int, tuple[str, ...]]
    training_audit: pd.DataFrame


def _fit(
    values: np.ndarray,
    labels: np.ndarray,
    *,
    encoder_factory: Callable[[], torch.nn.Module],
    frozen_backbone: bool,
    seed: int,
    max_epochs: int,
    batch_size: int,
    device: torch.device,
    validation: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[TokenClassifier, int]:
    torch.manual_seed(int(seed))
    model = TokenClassifier(encoder_factory(), frozen_backbone=frozen_backbone).to(device)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=1e-3, weight_decay=1e-4)
    positive = max(int((labels == 1).sum()), 1)
    negative = max(int((labels == 0).sum()), 1)
    loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(negative / positive, device=device))
    dataset = TensorDataset(torch.as_tensor(values, dtype=torch.float32), torch.as_tensor(labels, dtype=torch.float32))
    generator = torch.Generator().manual_seed(int(seed))
    loader = DataLoader(dataset, batch_size=min(int(batch_size), len(dataset)), shuffle=True, generator=generator, num_workers=0)
    best_state = None
    best_loss = float("inf")
    best_epoch = max(1, int(max_epochs))
    for epoch in range(1, max(1, int(max_epochs)) + 1):
        model.train()
        for batch_values, batch_labels in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_values.to(device).unsqueeze(1))
            loss = loss_fn(logits, batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        if validation is not None:
            model.eval()
            with torch.no_grad():
                validation_logits = model(torch.as_tensor(validation[0], dtype=torch.float32, device=device).unsqueeze(1))
                validation_loss = float(loss_fn(validation_logits, torch.as_tensor(validation[1], dtype=torch.float32, device=device)).item())
            if validation_loss < best_loss:
                best_loss = validation_loss
                best_epoch = epoch
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_epoch


def _predict(model: TokenClassifier, values: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(values), int(batch_size)):
            tensor = torch.as_tensor(values[start : start + int(batch_size)], dtype=torch.float32, device=device).unsqueeze(1)
            probabilities.append(torch.sigmoid(model(tensor)).cpu().numpy())
    return np.concatenate(probabilities)


def run_task1_token_oof(
    tokens: Task1TokenDataset,
    fold_manifest: pd.DataFrame,
    *,
    model_name: str,
    seed: int,
    encoder_factory: Callable[[], torch.nn.Module],
    frozen_backbone: bool,
    inner_folds: int = 4,
    max_epochs: int = 30,
    batch_size: int = 128,
    device: str = "cpu",
    fixed_baseline: bool = False,
) -> TokenOOFResult:
    started_at = time.perf_counter()

    def progress(message: str) -> None:
        print(f"[Task1 token +{time.perf_counter() - started_at:8.1f}s] {message}", flush=True)

    metadata = tokens.rows.merge(fold_manifest[["subject_id", "outer_fold"]], on="subject_id", how="inner", validate="many_to_one")
    if len(metadata) != len(tokens.rows):
        raise ValueError("Task 1 token subjects do not exactly match frozen fold manifest.")
    active_device = torch.device(device)
    oof_rows = []
    audits = []
    fit_subjects: dict[int, tuple[str, ...]] = {}
    folds = sorted(metadata["outer_fold"].unique())
    for fold in folds:
        progress(f"START outer_fold={fold}/{len(folds)} model={model_name} seed={seed}")
        train_indices = np.flatnonzero(metadata["outer_fold"].to_numpy() != fold)
        test_indices = np.flatnonzero(metadata["outer_fold"].to_numpy() == fold)
        train_subject_values = metadata.iloc[train_indices]["subject_id"].to_numpy()
        fit_subjects[int(fold)] = tuple(sorted(set(train_subject_values)))
        if fixed_baseline:
            threshold = Task1Threshold(0.5, float("nan"), float("nan"), float("nan"), source="fixed_0.5")
            final_epochs = max(1, int(max_epochs))
        else:
            splitter = GroupKFold(n_splits=min(int(inner_folds), len(set(train_subject_values))))
            inner_window_rows = []
            best_epochs = []
            for inner_index, (fit_local, valid_local) in enumerate(splitter.split(train_indices, groups=train_subject_values), start=1):
                fit_indices = train_indices[fit_local]
                valid_indices = train_indices[valid_local]
                model, best_epoch = _fit(
                    tokens.values[fit_indices], metadata.iloc[fit_indices]["label_nez"].to_numpy(dtype=int),
                    encoder_factory=encoder_factory, frozen_backbone=frozen_backbone, seed=seed + inner_index,
                    max_epochs=max_epochs, batch_size=batch_size, device=active_device,
                    validation=(tokens.values[valid_indices], metadata.iloc[valid_indices]["label_nez"].to_numpy(dtype=int)),
                )
                current = metadata.iloc[valid_indices][["subject_id", "center", "seizure_id", "channel_name", "window_id", "label_nez"]].copy()
                current["score_nez_probability"] = _predict(model, tokens.values[valid_indices], active_device, batch_size)
                inner_window_rows.append(current)
                best_epochs.append(best_epoch)
            inner_channel = aggregate_task1_window_probabilities(pd.concat(inner_window_rows, ignore_index=True))
            threshold = select_patient_macro_threshold(inner_channel)
            final_epochs = max(1, int(round(median(best_epochs))))
        final_model, _ = _fit(
            tokens.values[train_indices], metadata.iloc[train_indices]["label_nez"].to_numpy(dtype=int),
            encoder_factory=encoder_factory, frozen_backbone=frozen_backbone, seed=seed,
            max_epochs=final_epochs, batch_size=batch_size, device=active_device,
        )
        outer_windows = metadata.iloc[test_indices][["subject_id", "center", "seizure_id", "channel_name", "window_id", "label_nez"]].copy()
        outer_windows["score_nez_probability"] = _predict(final_model, tokens.values[test_indices], active_device, batch_size)
        outer_channel = aggregate_task1_window_probabilities(outer_windows)
        outer_channel["outer_fold"] = int(fold)
        outer_channel["model"] = model_name
        outer_channel["seed"] = int(seed)
        outer_channel["selected_threshold"] = threshold.threshold
        outer_channel["threshold_source"] = threshold.source
        outer_channel["predicted_nez"] = (outer_channel["score_nez_probability"] >= threshold.threshold).astype(int)
        outer_channel["predicted_ez"] = 1 - outer_channel["predicted_nez"]
        outer_channel["clinical_true_nez"] = outer_channel["label_nez"]
        outer_channel["clinical_true_ez"] = 1 - outer_channel["label_nez"]
        oof_rows.append(outer_channel)
        audits.append({"outer_fold": int(fold), "seed": int(seed), "model": model_name, "selected_epochs": final_epochs, "frozen_backbone": frozen_backbone})
        progress(f"DONE outer_fold={fold}/{len(folds)} model={model_name} seed={seed} oof_rows={len(outer_channel)}")
    return TokenOOFResult(pd.concat(oof_rows, ignore_index=True).sort_values(["subject_id", "channel_name"]).reset_index(drop=True), fit_subjects, pd.DataFrame(audits))


__all__ = ["TokenClassifier", "TokenOOFResult", "run_task1_token_oof"]
