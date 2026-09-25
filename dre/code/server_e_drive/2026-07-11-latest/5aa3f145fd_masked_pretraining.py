from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import torch
import torch.nn.functional as F
from torch import nn


def _move_tensors_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def _mask_like_features(features: torch.Tensor, batch: dict[str, Any], args: Any) -> tuple[torch.Tensor, torch.Tensor]:
    window_mask = batch.get("window_mask")
    seizure_channel_mask = batch.get("seizure_channel_mask")
    if window_mask is None:
        window_mask = torch.ones(features.shape[:3], dtype=torch.bool, device=features.device)
    if seizure_channel_mask is None:
        seizure_channel_mask = torch.ones((*features.shape[:2], features.shape[3]), dtype=torch.bool, device=features.device)
    valid = window_mask[:, :, :, None, None] & seizure_channel_mask[:, :, None, :, None]
    mask = torch.zeros_like(features, dtype=torch.bool)
    window_prob = float(getattr(args, "pretrain_mask_window_prob", 0.15))
    channel_prob = float(getattr(args, "pretrain_mask_channel_prob", 0.15))
    feature_prob = float(getattr(args, "pretrain_mask_feature_prob", 0.10))
    if window_prob > 0.0:
        mask |= torch.rand((*features.shape[:3], 1, 1), device=features.device) < window_prob
    if channel_prob > 0.0:
        mask |= torch.rand((features.shape[0], features.shape[1], 1, features.shape[3], 1), device=features.device) < channel_prob
    if feature_prob > 0.0:
        mask |= torch.rand((1, 1, 1, 1, features.shape[-1]), device=features.device) < feature_prob
    mask = mask & valid
    if not torch.any(mask & valid):
        flat_valid = torch.nonzero(valid.expand_as(features), as_tuple=False)
        if flat_valid.numel() > 0:
            index = flat_valid[0]
            mask[tuple(index.tolist())] = True
    masked_features = features.masked_fill(mask, 0.0)
    return masked_features, mask


def _mse_on_mask(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if not torch.any(mask):
        return pred.sum() * 0.0
    return F.mse_loss(pred[mask], target[mask])


def _assert_fold_safe(batch_subjects: Iterable[str], train_subjects: set[str], forbidden_subjects: set[str]) -> None:
    batch_set = {str(subject_id) for subject_id in batch_subjects}
    leaked = batch_set & forbidden_subjects
    if leaked:
        raise AssertionError(f"Masked pretraining received non-training subject(s): {sorted(leaked)}")
    outside = batch_set - train_subjects
    if outside:
        raise AssertionError(f"Masked pretraining received subject(s) outside fit fold: {sorted(outside)}")


def pretrain_masked_windows(
    model: nn.Module,
    loader: Any,
    args: Any,
    device: torch.device,
    *,
    train_subjects: Iterable[str],
    forbidden_subjects: Iterable[str],
    output_path: Path | None = None,
) -> dict[str, Any]:
    epochs = int(getattr(args, "pretrain_epochs", 0))
    if epochs <= 0:
        return {"enabled": bool(getattr(args, "pretrain_masked_windows", False)), "epochs": 0, "losses": []}

    train_subject_set = {str(subject_id) for subject_id in train_subjects}
    forbidden_subject_set = {str(subject_id) for subject_id in forbidden_subjects}
    reconstruct_parts = str(getattr(args, "pretrain_reconstruct_parts", "all")).lower()
    train_b0 = reconstruct_parts in {"b0", "all"}
    train_physics = reconstruct_parts in {"physics", "all"} and hasattr(model, "physics_encoder")

    b0_decoder: nn.Module | None = None
    physics_decoder: nn.Module | None = None
    params: list[nn.Parameter] = []
    if train_b0:
        b0_decoder = nn.LazyLinear(loader.dataset.patient_examples[0]["b0_features"][0].shape[-1]).to(device)
        _ = b0_decoder(torch.zeros((1, int(model.b0_encoder.model_dim)), dtype=torch.float32, device=device))
        params.extend(list(model.b0_encoder.parameters()))
        params.extend(list(b0_decoder.parameters()))
    if train_physics:
        physics_decoder = nn.LazyLinear(loader.dataset.patient_examples[0]["physics_features"][0].shape[-1]).to(device)
        _ = physics_decoder(torch.zeros((1, int(model.physics_encoder.model_dim)), dtype=torch.float32, device=device))
        params.extend(list(model.physics_encoder.state_proj.parameters()))
        params.extend(list(physics_decoder.parameters()))
    if not params:
        summary = {
            "enabled": True,
            "epochs": 0,
            "losses": [],
            "warning": f"No reconstructable parts available for pretrain_reconstruct_parts={reconstruct_parts!r}.",
        }
        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    optimizer = torch.optim.AdamW(
        params,
        lr=float(getattr(args, "pretrain_learning_rate", 0.0003)),
        weight_decay=float(getattr(args, "pretrain_weight_decay", 0.0001)),
    )
    losses: list[float] = []
    model.train()
    for _epoch in range(1, epochs + 1):
        epoch_losses: list[float] = []
        for raw_batch in loader:
            _assert_fold_safe(raw_batch.get("subject_id", []), train_subject_set, forbidden_subject_set)
            batch = _move_tensors_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            part_losses: list[torch.Tensor] = []
            if train_b0 and b0_decoder is not None:
                masked, mask = _mask_like_features(batch["b0_features"], batch, args)
                embedding = model.b0_encoder(masked, None, batch.get("seizure_channel_mask"))
                recon = b0_decoder(embedding)
                part_losses.append(_mse_on_mask(recon, batch["b0_features"], mask))
            if train_physics and physics_decoder is not None:
                masked_physics, physics_mask = _mask_like_features(batch["physics_features"], batch, args)
                embedding = model.physics_encoder.state_proj(masked_physics)
                recon = physics_decoder(embedding)
                part_losses.append(_mse_on_mask(recon, batch["physics_features"], physics_mask))
            loss = torch.stack(part_losses).mean()
            loss.backward()
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))
        losses.append(float(sum(epoch_losses) / max(len(epoch_losses), 1)))

    if bool(getattr(args, "pretrain_freeze_after", False)):
        for parameter in model.b0_encoder.parameters():
            parameter.requires_grad = False
        if train_physics and hasattr(model, "physics_encoder"):
            for parameter in model.physics_encoder.state_proj.parameters():
                parameter.requires_grad = False

    summary = {
        "enabled": True,
        "epochs": epochs,
        "losses": losses,
        "final_loss": losses[-1] if losses else 0.0,
        "train_patient_count": len(train_subject_set),
        "train_subjects_only": True,
        "forbidden_patient_count": len(forbidden_subject_set),
        "reconstruct_parts": reconstruct_parts,
        "trained_parts": ",".join(part for part, enabled in (("b0", train_b0), ("physics", train_physics)) if enabled),
    }
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


__all__ = ["pretrain_masked_windows"]
