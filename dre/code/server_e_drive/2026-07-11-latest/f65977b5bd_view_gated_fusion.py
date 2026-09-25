from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass(frozen=True)
class ViewSlice:
    name: str
    start: int
    end: int


def parse_view_slices(raw: str, feature_dim: int | None = None) -> list[ViewSlice]:
    text = str(raw or "").strip()
    if not text:
        raise ValueError("physics_view_slices is required when use_view_gated_fusion=True.")
    views: list[ViewSlice] = []
    seen: set[str] = set()
    for part in text.split(","):
        tokens = [token.strip() for token in part.split(":")]
        if len(tokens) != 3 or not tokens[0]:
            raise ValueError(f"Invalid physics_view_slices entry {part!r}; expected name:start:end.")
        name = tokens[0].lower()
        if name in seen:
            raise ValueError(f"Duplicate physics_view_slices view name {name!r}.")
        try:
            start = int(tokens[1])
            end = int(tokens[2])
        except ValueError as exc:
            raise ValueError(f"Invalid physics_view_slices entry {part!r}; start/end must be integers.") from exc
        if start < 0 or end <= start:
            raise ValueError(f"Invalid physics_view_slices entry {part!r}; expected 0 <= start < end.")
        if feature_dim is not None and end > int(feature_dim):
            raise ValueError(
                f"physics_view_slices entry {part!r} is outside available physics feature dim={int(feature_dim)}."
            )
        seen.add(name)
        views.append(ViewSlice(name=name, start=start, end=end))
    return views


class ViewGatedFusion(nn.Module):
    """Static-safe view fusion over selected physics feature slices."""

    def __init__(
        self,
        args: Any | None = None,
        *,
        model_dim: int = 32,
    ) -> None:
        super().__init__()
        self.model_dim = int(model_dim)
        self.raw_slices = str(getattr(args, "physics_view_slices", "") if args is not None else "")
        self.gate_init = float(getattr(args, "view_gate_init", -3.0) if args is not None else -3.0)
        self.static_fallback = bool(getattr(args, "static_fallback", True) if args is not None else True)
        self.dropout = nn.Dropout(float(getattr(args, "view_dropout", 0.15) if args is not None else 0.15))
        self.view_specs: list[ViewSlice] = []
        self.view_mlps = nn.ModuleDict()
        self.gates = nn.ParameterDict()
        self._built_feature_dim: int | None = None

    def _build(self, feature_dim: int, device: torch.device) -> None:
        if self._built_feature_dim == int(feature_dim):
            return
        specs = parse_view_slices(self.raw_slices, feature_dim=feature_dim)
        if self.static_fallback and not any(spec.name == "static" for spec in specs):
            raise ValueError("physics_view_slices must include a 'static' view when static_fallback=True.")
        self.view_specs = specs
        self.view_mlps = nn.ModuleDict()
        self.gates = nn.ParameterDict()
        for spec in specs:
            self.view_mlps[spec.name] = nn.Sequential(
                nn.LazyLinear(self.model_dim),
                nn.GELU(),
                nn.LayerNorm(self.model_dim),
            )
            if spec.name != "static":
                self.gates[spec.name] = nn.Parameter(torch.tensor(self.gate_init, dtype=torch.float32))
        self._built_feature_dim = int(feature_dim)
        self.to(device)

    def forward(
        self,
        physics_features: torch.Tensor,
        seizure_channel_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        feature_dim = int(physics_features.shape[-1])
        self._build(feature_dim, physics_features.device)
        outputs: dict[str, torch.Tensor] = {}
        fused: torch.Tensor | None = None
        gate_values: list[torch.Tensor] = []
        for spec in self.view_specs:
            view_tensor = physics_features[..., spec.start : spec.end]
            projected = self.view_mlps[spec.name](view_tensor)
            if spec.name == "static" and fused is None:
                fused = projected
                outputs["view_gate_static"] = projected.new_tensor(1.0)
                continue
            gate = torch.sigmoid(self.gates[spec.name])
            gate_values.append(gate)
            outputs[f"view_gate_{spec.name}"] = gate
            residual = self.dropout(projected)
            fused = residual * gate if fused is None else fused + gate * residual
        if fused is None:
            raise ValueError("No usable view was produced by ViewGatedFusion.")
        if seizure_channel_mask is not None:
            fused = fused * seizure_channel_mask[:, :, None, :, None].float()
        if gate_values:
            outputs["view_gate_l1_loss"] = torch.stack([gate.abs() for gate in gate_values]).mean()
        else:
            outputs["view_gate_l1_loss"] = fused.sum() * 0.0
        return fused, outputs


__all__ = ["ViewGatedFusion", "ViewSlice", "parse_view_slices"]
