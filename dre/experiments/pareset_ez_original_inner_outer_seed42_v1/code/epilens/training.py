"""Outer-fold training with validation-only checkpoint and threshold selection."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import copy
import json
import random

import numpy as np
import pandas as pd
import torch

from .data import PatientRecord, select_records
from .evaluation import select_threshold
from .features import Standardizer, fit_standardizer, four_view_expansion
from .models import BCRNet, PRQNet
from .objectives import bcr_loss, prq_loss


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 6
    minimum_epochs: int = 6
    patient_batch_size: int = 4
    gradient_clip: float = 1.0
    device: str = "cuda"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _tensorize(
    record: PatientRecord, standardizer: Standardizer, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    expanded, valid = four_view_expansion(
        record.descriptors, record.window_times, record.valid
    )
    expanded = standardizer.apply(expanded, valid)
    return (
        torch.as_tensor(expanded, device=device),
        torch.as_tensor(valid, device=device),
        torch.as_tensor(record.label_nez, device=device, dtype=torch.long),
    )


def predict_records(
    model: torch.nn.Module,
    records: list[PatientRecord],
    standardizer: Standardizer,
    branch: str,
    device: torch.device,
) -> pd.DataFrame:
    model.eval()
    rows = []
    with torch.no_grad():
        for record in records:
            features, valid, labels = _tensorize(record, standardizer, device)
            output = model(features, valid)
            probability_nez = output["probability_nez"].detach().cpu().numpy()
            logit_ez = (
                output["logit_ez"].detach().cpu().numpy()
                if branch == "bcr"
                else np.full(len(labels), np.nan)
            )
            for channel, name in enumerate(record.channel_names):
                rows.append(
                    {
                        "patient_id": record.patient_id,
                        "center": record.center,
                        "channel_name": name,
                        "label_nez": int(record.label_nez[channel]),
                        "probability_nez": float(probability_nez[channel]),
                        "logit_ez": float(logit_ez[channel]),
                    }
                )
    return pd.DataFrame(rows)


def load_trained_branch(
    checkpoint_path: str | Path, device: torch.device
) -> tuple[torch.nn.Module, Standardizer, str]:
    """Restore a primary PRQ-Net or BCR-Net checkpoint without repository state."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    branch = str(checkpoint["branch"])
    model = (
        PRQNet(
            quantile=float(checkpoint.get("quantile", 0.10)),
            use_temporal=bool(checkpoint.get("use_temporal", True)),
            use_patient_relative=bool(checkpoint.get("use_patient_relative", True)),
            temporal_aggregation=str(checkpoint.get("temporal_aggregation", "quantile")),
        )
        if branch == "prq"
        else BCRNet(quantile_control=bool(checkpoint.get("quantile_control", False)))
    )
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    standardizer = Standardizer(
        np.asarray(checkpoint["standardizer_mean"], dtype=np.float32),
        np.asarray(checkpoint["standardizer_scale"], dtype=np.float32),
    )
    return model, standardizer, branch


def train_outer_fold(
    branch: str,
    records: list[PatientRecord],
    manifest: pd.DataFrame,
    outer_fold: int,
    seed: int,
    output_directory: str | Path,
    config: TrainingConfig,
    *,
    quantile: float = 0.10,
    use_temporal: bool = True,
    use_patient_relative: bool = True,
    temporal_aggregation: str = "quantile",
    quantile_control: bool = False,
    boundary_weight: float = 0.05,
    coverage_weight: float = 0.08,
) -> dict[str, object]:
    """Train one independently parameterized branch and evaluate its test fold."""
    if branch not in {"prq", "bcr"}:
        raise ValueError("branch must be prq or bcr")
    seed_everything(seed)
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    fit = select_records(records, manifest, outer_fold, "fit")
    validation = select_records(records, manifest, outer_fold, "validation")
    test = select_records(records, manifest, outer_fold, "test")
    standardizer = fit_standardizer(fit)
    model: torch.nn.Module
    if branch == "prq":
        model = PRQNet(
            quantile=quantile,
            use_temporal=use_temporal,
            use_patient_relative=use_patient_relative,
            temporal_aggregation=temporal_aggregation,
        )
    else:
        model = BCRNet(quantile_control=quantile_control)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    best_state = None
    best_selection = None
    stale = 0
    history = []
    shuffled = list(fit)
    for epoch in range(1, config.max_epochs + 1):
        random.Random(seed + epoch).shuffle(shuffled)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for index, record in enumerate(shuffled, start=1):
            features, valid, labels = _tensorize(record, standardizer, device)
            output = model(features, valid)
            if branch == "prq":
                loss = prq_loss(output["logit_nez"], labels, output["channel_valid"])
            else:
                loss, _ = bcr_loss(
                    output["logit_ez"],
                    labels,
                    output["channel_valid"],
                    boundary_weight,
                    coverage_weight,
                )
            (loss / config.patient_batch_size).backward()
            losses.append(float(loss.detach().cpu()))
            if index % config.patient_batch_size == 0 or index == len(shuffled):
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        validation_ledger = predict_records(
            model, validation, standardizer, branch, device
        )
        selection = select_threshold(validation_ledger)
        history.append(
            {
                "epoch": epoch,
                "training_loss": float(np.mean(losses)),
                **asdict(selection),
            }
        )
        signal = (
            selection.patient_macro_f1,
            selection.patient_ez_f1,
            selection.patient_balanced_accuracy,
        )
        best_signal = (
            (
                best_selection.patient_macro_f1,
                best_selection.patient_ez_f1,
                best_selection.patient_balanced_accuracy,
            )
            if best_selection is not None
            else (-np.inf, -np.inf, -np.inf)
        )
        if signal > best_signal:
            best_state = copy.deepcopy(model.state_dict())
            best_selection = selection
            stale = 0
        else:
            stale += 1
        if epoch >= config.minimum_epochs and stale >= config.patience:
            break
    if best_state is None or best_selection is None:
        raise RuntimeError("No checkpoint was selected")
    model.load_state_dict(best_state)
    validation_ledger = predict_records(model, validation, standardizer, branch, device)
    test_ledger = predict_records(model, test, standardizer, branch, device)
    for frame, partition in ((validation_ledger, "validation"), (test_ledger, "test")):
        frame["outer_fold"] = outer_fold
        frame["seed"] = seed
        frame["partition"] = partition
        frame["branch"] = "PRQ-Net" if branch == "prq" else "BCR-Net"
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": best_state,
            "standardizer_mean": standardizer.mean,
            "standardizer_scale": standardizer.scale,
            "branch": branch,
            "seed": seed,
            "outer_fold": outer_fold,
            "quantile": quantile,
            "use_temporal": use_temporal,
            "use_patient_relative": use_patient_relative,
            "temporal_aggregation": temporal_aggregation,
            "quantile_control": quantile_control,
        },
        output / f"{branch}_seed_{seed}_fold_{outer_fold}.pt",
    )
    pd.DataFrame(history).to_csv(
        output / f"{branch}_seed_{seed}_fold_{outer_fold}_history.csv", index=False
    )
    validation_ledger.to_csv(
        output / f"{branch}_seed_{seed}_fold_{outer_fold}_validation.csv", index=False
    )
    test_ledger.to_csv(
        output / f"{branch}_seed_{seed}_fold_{outer_fold}_test.csv", index=False
    )
    (output / f"{branch}_seed_{seed}_fold_{outer_fold}_threshold.json").write_text(
        json.dumps(asdict(best_selection), indent=2), encoding="utf-8"
    )
    return {
        "checkpoint": str(output / f"{branch}_seed_{seed}_fold_{outer_fold}.pt"),
        "threshold": best_selection.threshold,
        "validation_ledger": validation_ledger,
        "test_ledger": test_ledger,
    }
