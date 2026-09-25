from __future__ import annotations

import copy
import json
import random
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from outcome_hifos.checkpoint import load_checkpoint, save_checkpoint_atomic
from outcome_hifos.collate import collate_outcome_patients
from outcome_hifos.dataset import OutcomePatientExample
from outcome_hifos.training.evaluator import evaluate_model
from outcome_hifos.training.losses import compute_fold_class_weight, compute_outcome_loss
from outcome_hifos.metrics import MetricBundle


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_metric: float
    checkpoint_path: Path
    history: pd.DataFrame
    validation_predictions: pd.DataFrame


def select_early_stop_score(
    metrics: MetricBundle,
    predictions: pd.DataFrame,
    requested_metric: str,
) -> tuple[float, str, str | None]:
    requested = str(requested_metric).lower()
    allowed = {"validation_loss", "auroc", "auprc", "macro_f1_at_0_5"}
    if requested not in allowed:
        raise ValueError(f"early_stop_metric must be one of {sorted(allowed)}, got {requested!r}.")
    logits = predictions["logit"].to_numpy(dtype=np.float64)
    targets = predictions["outcome"].to_numpy(dtype=np.float64)
    validation_loss = float(np.mean(np.logaddexp(0.0, logits) - targets * logits))
    if requested == "validation_loss":
        return -validation_loss, "validation_loss", None
    metric_key = "macro_f1" if requested == "macro_f1_at_0_5" else requested
    value = float(metrics.values.get(metric_key, float("nan")))
    if np.isfinite(value):
        return value, requested, None
    reason = metrics.undefined_reasons.get(metric_key, "non_finite_validation_metric")
    return -validation_loss, "validation_loss", reason


class OutcomeTrainer:
    def __init__(self, config: dict[str, Any], device: str | torch.device = "cpu") -> None:
        self.config = dict(config)
        self.device = torch.device(device)

    def _loader(self, examples: Sequence[OutcomePatientExample], shuffle: bool) -> DataLoader:
        return DataLoader(
            list(examples),
            batch_size=int(self.config.get("batch_size", 1)),
            shuffle=bool(shuffle),
            num_workers=int(self.config.get("num_workers", 0)),
            collate_fn=partial(collate_outcome_patients, padding_value=float(self.config.get("padding_value", 0.0))),
        )

    def fit(
        self,
        model: torch.nn.Module,
        train_examples: Sequence[OutcomePatientExample],
        validation_examples: Sequence[OutcomePatientExample],
        run_dir: str | Path,
        *,
        resume: bool = False,
    ) -> TrainingResult:
        if not train_examples or not validation_examples:
            raise ValueError("OutcomeTrainer requires non-empty train and validation examples.")
        seed = int(self.config.get("random_seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model = model.to(self.device)
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(self.config.get("learning_rate", 1e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-3)),
        )
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        class_weight = compute_fold_class_weight(
            [example.target for example in train_examples],
            [example.subject_id for example in train_examples],
        )
        class_weight.save(output_dir / "class_weight_audit.json")
        last_path = output_dir / "checkpoint_last.pt"
        best_path = output_dir / "checkpoint_best.pt"
        start_epoch = 1
        best_metric = -float("inf")
        best_epoch = 0
        amp_enabled = bool(self.config.get("amp", True)) and self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        if resume and last_path.exists():
            state = load_checkpoint(last_path, model=model, optimizer=optimizer, scaler=scaler, map_location=self.device)
            start_epoch = state.epoch + 1
            best_metric = state.best_metric
            best_epoch = int(state.extra.get("best_epoch", state.epoch))
        train_loader = self._loader(train_examples, True)
        validation_loader = self._loader(validation_examples, False)
        history_rows: list[dict[str, float]] = []
        requested_early_stop = str(self.config.get("early_stop_metric", "auroc"))
        patience = int(self.config.get("patience", 12))
        epochs_without_improvement = 0
        accumulation = max(1, int(self.config.get("gradient_accumulation", 1)))
        for epoch in range(start_epoch, int(self.config.get("epochs", 100)) + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for batch_index, batch in enumerate(train_loader, start=1):
                model_input = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch["model_input"].items()}
                targets = batch["outcome"].to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    outputs = model(model_input, persist_diagnostics=False)
                    loss, _ = compute_outcome_loss(outputs, targets, self.config, pos_weight=class_weight.pos_weight)
                scaler.scale(loss / accumulation).backward()
                if batch_index % accumulation == 0 or batch_index == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config.get("grad_clip", 1.0)))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                losses.append(float(loss.detach().cpu()))
            validation_metrics, validation_predictions = evaluate_model(model, validation_loader, self.device)
            score, effective_metric, fallback_reason = select_early_stop_score(
                validation_metrics,
                validation_predictions,
                requested_early_stop,
            )
            history_rows.append(
                {
                    "epoch": float(epoch),
                    "train_loss": float(np.mean(losses)),
                    "val_macro_f1_at_0_5": float(validation_metrics.values["macro_f1"]),
                    "val_auroc": float(validation_metrics.values["auroc"]),
                    "val_auprc": float(validation_metrics.values["auprc"]),
                    "early_stop_score": score,
                    "early_stop_metric_effective": effective_metric,
                    "fallback_reason": fallback_reason,
                }
            )
            (output_dir / "early_stopping_audit.json").write_text(
                json.dumps(
                    {
                        "early_stop_metric_requested": requested_early_stop,
                        "early_stop_metric_effective": effective_metric,
                        "fallback_reason": fallback_reason,
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            improved = score > best_metric + 1e-8
            if improved:
                best_metric = score
                best_epoch = epoch
                epochs_without_improvement = 0
                save_checkpoint_atomic(best_path, model=model, optimizer=optimizer, epoch=epoch, best_metric=best_metric, extra={"best_epoch": best_epoch}, scaler=scaler)
            else:
                epochs_without_improvement += 1
            save_checkpoint_atomic(last_path, model=model, optimizer=optimizer, epoch=epoch, best_metric=best_metric, extra={"best_epoch": best_epoch}, scaler=scaler)
            pd.DataFrame(history_rows).to_csv(output_dir / "training_curve.csv", index=False)
            if patience > 0 and epochs_without_improvement >= patience:
                break
        load_checkpoint(best_path, model=model, map_location=self.device)
        _, final_predictions = evaluate_model(model, validation_loader, self.device)
        return TrainingResult(best_epoch, best_metric, best_path, pd.DataFrame(history_rows), final_predictions)

    def fit_fixed_epochs(
        self,
        model: torch.nn.Module,
        train_examples: Sequence[OutcomePatientExample],
        run_dir: str | Path,
        *,
        epochs: int,
    ) -> Path:
        """Fit on the complete outer-train set for an epoch count selected by inner CV."""

        if not train_examples or int(epochs) < 1:
            raise ValueError("fit_fixed_epochs requires training examples and epochs >= 1.")
        seed = int(self.config.get("random_seed", 42))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        model.to(self.device)
        output_dir = Path(run_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        class_weight = compute_fold_class_weight(
            [example.target for example in train_examples],
            [example.subject_id for example in train_examples],
        )
        class_weight.save(output_dir / "class_weight_audit.json")
        optimizer = torch.optim.AdamW(
            (parameter for parameter in model.parameters() if parameter.requires_grad),
            lr=float(self.config.get("learning_rate", 1e-4)),
            weight_decay=float(self.config.get("weight_decay", 1e-3)),
        )
        loader = self._loader(train_examples, True)
        amp_enabled = bool(self.config.get("amp", True)) and self.device.type == "cuda"
        try:
            scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
        except (AttributeError, TypeError):
            scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
        history = []
        for epoch in range(1, int(epochs) + 1):
            model.train()
            losses = []
            for batch in loader:
                optimizer.zero_grad(set_to_none=True)
                model_input = {key: value.to(self.device) if torch.is_tensor(value) else value for key, value in batch["model_input"].items()}
                targets = batch["outcome"].to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=amp_enabled):
                    outputs = model(model_input, persist_diagnostics=False)
                    loss, _ = compute_outcome_loss(outputs, targets, self.config, pos_weight=class_weight.pos_weight)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(self.config.get("grad_clip", 1.0)))
                scaler.step(optimizer)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            history.append({"epoch": epoch, "train_loss": float(np.mean(losses))})
        checkpoint = output_dir / "checkpoint_outer_train_final.pt"
        save_checkpoint_atomic(checkpoint, model=model, optimizer=optimizer, epoch=int(epochs), best_metric=float("nan"), extra={"selection_source": "inner_oof_epoch_selection"}, scaler=scaler)
        pd.DataFrame(history).to_csv(output_dir / "outer_train_curve.csv", index=False)
        return checkpoint


__all__ = ["OutcomeTrainer", "TrainingResult", "select_early_stop_score"]
