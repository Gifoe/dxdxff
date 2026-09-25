from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class ClassWeightAudit:
    positive_count: int
    negative_count: int
    pos_weight: float
    fit_subject_ids: tuple[str, ...]

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["fit_subject_ids"] = list(self.fit_subject_ids)
        output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def compute_fold_class_weight(targets: Sequence[float], subject_ids: Sequence[str]) -> ClassWeightAudit:
    if len(targets) != len(subject_ids) or not targets:
        raise ValueError("Class-weight targets and subject IDs must be non-empty and have equal length.")
    values = [float(value) for value in targets]
    if any(value not in (0.0, 1.0) for value in values):
        raise ValueError("Class-weight targets must be binary patient outcomes.")
    positive_count = sum(value == 1.0 for value in values)
    negative_count = sum(value == 0.0 for value in values)
    if positive_count == 0 or negative_count == 0:
        raise ValueError("Training split must contain both outcome classes to compute a fixed class weight.")
    return ClassWeightAudit(
        positive_count=positive_count,
        negative_count=negative_count,
        pos_weight=float(negative_count / positive_count),
        fit_subject_ids=tuple(str(value) for value in subject_ids),
    )


def compute_outcome_loss(
    outputs: dict[str, torch.Tensor],
    targets: torch.Tensor,
    config: dict[str, Any],
    *,
    pos_weight: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    positive_weight = torch.as_tensor(float(pos_weight), device=targets.device, dtype=targets.dtype)
    bce = functional.binary_cross_entropy_with_logits(outputs["logits"], targets, pos_weight=positive_weight)
    zero = bce * 0.0
    anchor_loss = zero
    if "core_states" in outputs and "core_anchors" in outputs and "core_masses" in outputs:
        drift = (outputs["core_states"] - outputs["core_anchors"][:, None, None]).square().mean(dim=-1)
        anchor_loss = (drift * outputs["core_masses"]).sum() / outputs["core_masses"].sum().clamp_min(1e-8)
    usage_loss = zero
    if "core_masses" in outputs:
        epoch_usage = outputs["core_masses"].mean(dim=(0, 1, 2))
        usage_loss = torch.relu(float(config.get("minimum_core_usage", 0.01)) - epoch_usage).mean()
    total = bce + float(config.get("anchor_loss_weight", 0.0)) * anchor_loss + float(config.get("usage_loss_weight", 0.0)) * usage_loss
    return total, {"bce": bce, "anchor": anchor_loss, "usage": usage_loss}


__all__ = ["ClassWeightAudit", "compute_fold_class_weight", "compute_outcome_loss"]
