from __future__ import annotations

import random
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from .nvr_loss import nvr_loss

try:
    from tqdm.auto import trange
except ImportError:  # pragma: no cover
    trange = None


class NVRDataset(Dataset):
    def __init__(self, records: Sequence[dict[str, Any]]) -> None: self.records = list(records)
    def __len__(self) -> int: return len(self.records)
    def __getitem__(self, index: int) -> dict[str, Any]: return self.records[index]


def _pad_variant(records: Sequence[dict[str, Any]], key: str) -> dict[str, torch.Tensor]:
    values = [record[key] for record in records]
    maximum = max(value["patient_channel_embedding"].shape[0] for value in values)
    embedding_dim = values[0]["patient_channel_embedding"].shape[-1]
    scalar_dim = values[0]["channel_scalars"].shape[-1]
    batch = len(values)
    output = {
        "scalar_values": torch.stack([value["scalar_values"] for value in values]),
        "patient_channel_embedding": torch.zeros(batch, maximum, embedding_dim),
        "channel_scalars": torch.zeros(batch, maximum, scalar_dim),
        "target": torch.zeros(batch, maximum, dtype=torch.bool),
        "channel_mask": torch.zeros(batch, maximum, dtype=torch.bool),
        "reliable_abnormality": torch.zeros(batch, maximum),
    }
    for index, value in enumerate(values):
        n = value["patient_channel_embedding"].shape[0]
        for name in ("patient_channel_embedding", "channel_scalars", "target", "channel_mask", "reliable_abnormality"):
            output[name][index, :n] = value[name]
    return output


def collate_nvr(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"main": _pad_variant(records, "main")}
    for key in ("drop", "plus", "minus"):
        if all(key in record for record in records): output[key] = _pad_variant(records, key)
    output.update({"patient_key": [str(record["patient_key"]) for record in records], "center": [str(record["center"]) for record in records],
                   "outcome_target": torch.tensor([int(record["outcome_target"]) for record in records], dtype=torch.float32),
                   "counterfactual_valid": torch.tensor([bool(record.get("counterfactual_valid", False)) for record in records])})
    return output


def make_nvr_loader(records: Sequence[dict[str, Any]], *, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    return DataLoader(NVRDataset(records), batch_size=int(batch_size), shuffle=shuffle, generator=torch.Generator().manual_seed(seed), collate_fn=collate_nvr, num_workers=0)


def _to_device(value: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]: return {key: item.to(device) for key, item in value.items()}


def fit_nvr_normalizer(model: torch.nn.Module, records: Sequence[dict[str, Any]]) -> None:
    model.fit_normalizer(torch.stack([record["main"]["scalar_values"] for record in records]).to(next(model.parameters()).device))


def train_nvr_fixed_epochs(model: torch.nn.Module, loader: DataLoader, *, device: str | torch.device, epochs: int = 25,
                           seed: int = 42, robust: bool = False, progress_prefix: str = "NVR",
                           checkpoint_path: str | Path | None = None, resume: bool = True, resume_signature: str = "") -> pd.DataFrame:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    target_device = torch.device(device); model.to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
    path = Path(checkpoint_path) if checkpoint_path else None; rows: list[dict[str, Any]] = []; start = 1; total = int(epochs)
    if resume and path is not None and path.exists():
        payload = torch.load(path, map_location=target_device, weights_only=False)
        if payload.get("resume_signature") == resume_signature and int(payload.get("total_epochs", -1)) == total:
            model.load_state_dict(payload["model_state_dict"]); optimizer.load_state_dict(payload["optimizer_state_dict"])
            rows = list(payload.get("history", [])); start = int(payload.get("completed_epoch", 0)) + 1
            print(f"[{progress_prefix}] resume from epoch {start}/{total}", flush=True)
    iterator = trange(start, total+1, initial=start-1, total=total, desc=progress_prefix, ascii=True, dynamic_ncols=True) if trange is not None else range(start, total+1)
    for epoch in iterator:
        model.train(); losses = []
        for batch in loader:
            optimizer.zero_grad(set_to_none=True); main = model(_to_device(batch["main"], target_device)); dropped = plus = minus = None
            if robust:
                dropped = model(_to_device(batch["drop"], target_device))["outcome_logit_success"]
                plus = model(_to_device(batch["plus"], target_device))["outcome_logit_success"]
                minus = model(_to_device(batch["minus"], target_device))["outcome_logit_success"]
            parts = nvr_loss(main["outcome_logit_success"], batch["outcome_target"].to(target_device), batch["center"], model.parameters(),
                             dropped_logit=dropped, plus_logit=plus, minus_logit=minus,
                             counterfactual_valid=batch["counterfactual_valid"].to(target_device), robust=robust)
            if not torch.isfinite(parts["loss"]): raise FloatingPointError(f"Non-finite NVR loss at epoch {epoch}")
            parts["loss"].backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step(); losses.append(float(parts["loss"].detach().cpu()))
        mean = float(np.mean(losses)); rows.append({"epoch": epoch, "train_loss": mean, "role": "outer_train_fixed_epoch"})
        if trange is not None: iterator.set_postfix(loss=f"{mean:.6f}")
        else: print(f"[{progress_prefix}] epoch {epoch}/{total} loss={mean:.6f}", flush=True)
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True); temporary = path.with_suffix(path.suffix+".tmp")
            torch.save({"model_state_dict": model.state_dict(), "optimizer_state_dict": optimizer.state_dict(), "completed_epoch": epoch,
                        "total_epochs": total, "history": rows, "resume_signature": resume_signature}, temporary); temporary.replace(path)
    return pd.DataFrame(rows)


def predict_nvr(model: torch.nn.Module, loader: DataLoader, device: str | torch.device) -> pd.DataFrame:
    target_device = torch.device(device); model.eval(); rows = []
    with torch.no_grad():
        for batch in loader:
            output = model(_to_device(batch["main"], target_device)); probabilities = output["outcome_probability_success"].cpu().numpy()
            for index, patient in enumerate(batch["patient_key"]):
                p = float(probabilities[index]); y = int(batch["outcome_target"][index])
                row = {"patient_key": patient, "patient_id": patient, "center": batch["center"][index], "outcome_true": y,
                             "outcome_group": "success" if y else "failure", "outcome_probability_success": p,
                             "outcome_probability_failure": 1-p, "outcome_pred_05": int(p >= .5)}
                for key in ("residual_risk_score", "network_risk_score", "diffuse_risk_score", "support_score", "structured_success_logit", "set_delta"):
                    row[key] = float(output[key][index].cpu()) if key in output else 0.0
                rows.append(row)
    return pd.DataFrame(rows)


__all__ = ["NVRDataset", "collate_nvr", "fit_nvr_normalizer", "make_nvr_loader", "predict_nvr", "train_nvr_fixed_epochs"]
