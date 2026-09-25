from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn


def _softplus_inverse(value: float) -> float:
    value = float(max(value, 1e-6))
    return float(torch.log(torch.expm1(torch.tensor(value))).item())


class NegativeAnchorHead(nn.Module):
    """One-class NEZ anchor branch whose distance residual boosts EZ scores.

    A9v2: projection is Linear->GELU->Dropout->Linear (no terminal LayerNorm by
    default).  Supports per-patient z-score normalisation of anchor-space
    distances before feeding the anchor logit.

    A9v4: optional center-specific residual gate so each clinical center can
    learn its own anchor reliance, plus per-center gate diagnostics.
    """

    def __init__(
        self,
        input_dim: int,
        anchor_dim: int = 16,
        margin: float = 1.0,
        gamma: float = 0.5,
        gate_init: float = -2.0,
        center_prototypes: bool = False,
        num_centers: int = 5,
        norm: str = "none",
        distance_mode: str = "patient_zscore",
        dropout: float = 0.1,
        center_gate: bool = False,
        center_gate_init: float = 0.0,
        pediatric_gate_delta_init: float = -2.0,
    ) -> None:
        super().__init__()
        self.anchor_dim = int(anchor_dim)
        self.margin = float(margin)
        self.gamma = float(gamma)
        self.center_prototypes = bool(center_prototypes)
        self.norm = str(norm).lower()
        self.distance_mode = str(distance_mode).lower()
        self.center_gate = bool(center_gate)
        self.num_centers = int(num_centers)

        layers: list[nn.Module] = [
            nn.Linear(int(input_dim), self.anchor_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.anchor_dim, self.anchor_dim),
        ]
        if self.norm == "layernorm":
            layers.append(nn.LayerNorm(self.anchor_dim))
        self.proj = nn.Sequential(*layers)

        self.prototype = nn.Parameter(torch.zeros((self.anchor_dim,), dtype=torch.float32))
        if self.center_prototypes:
            self.center_prototype = nn.Parameter(torch.zeros((self.num_centers, self.anchor_dim), dtype=torch.float32))
        self.alpha_raw = nn.Parameter(torch.tensor(_softplus_inverse(1.0), dtype=torch.float32))
        self.beta = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))
        self.gate_raw = nn.Parameter(torch.tensor(float(gate_init), dtype=torch.float32))

        if self.center_gate:
            self.center_gate_delta = nn.Parameter(torch.zeros((self.num_centers,), dtype=torch.float32))
            nn.init.constant_(self.center_gate_delta, float(center_gate_init))
            if 3 < self.num_centers:
                with torch.no_grad():
                    self.center_gate_delta[3] = float(pediatric_gate_delta_init)
        else:
            self.register_buffer("center_gate_delta", torch.zeros((self.num_centers,), dtype=torch.float32), persistent=False)

    @classmethod
    def from_args(cls, args: Any, *, input_dim: int) -> "NegativeAnchorHead":
        num_centers_val = int(getattr(args, "negative_anchor_num_centers", 5))
        return cls(
            input_dim=input_dim,
            anchor_dim=int(getattr(args, "negative_anchor_dim", 16)),
            margin=float(getattr(args, "negative_anchor_margin", 1.0)),
            gamma=float(getattr(args, "negative_anchor_gamma", 0.5)),
            gate_init=float(getattr(args, "negative_anchor_gate_init", -2.0)),
            center_prototypes=bool(getattr(args, "negative_anchor_center_prototypes", False)),
            num_centers=num_centers_val,
            norm=str(getattr(args, "negative_anchor_norm", "none")),
            distance_mode=str(getattr(args, "negative_anchor_distance_mode", "patient_zscore")),
            dropout=float(getattr(args, "negative_anchor_dropout", 0.1)),
            center_gate=bool(getattr(args, "negative_anchor_center_gate", False)),
            center_gate_init=float(getattr(args, "negative_anchor_center_gate_init", 0.0)),
            pediatric_gate_delta_init=float(getattr(args, "negative_anchor_pediatric_gate_delta_init", -2.0)),
        )

    def forward(
        self,
        patient_channel_embedding: torch.Tensor,
        labels_ez: torch.Tensor,
        channel_mask: torch.Tensor,
        supervised_logits: torch.Tensor,
        *,
        positive_label: str,
        center_id: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = self.proj(patient_channel_embedding)
        prototype = self.prototype.view(1, 1, -1)
        if self.center_prototypes and center_id is not None:
            center_id_safe = center_id.to(z.device).long().clamp(0, self.center_prototype.shape[0] - 1)
            prototype = prototype + self.center_prototype[center_id_safe].unsqueeze(1)
        distance = (z - prototype).square().sum(dim=-1).masked_fill(~channel_mask, 0.0)

        alpha = F.softplus(self.alpha_raw)

        if self.distance_mode == "patient_zscore":
            distance_transformed = distance.clone()
            for b in range(distance.shape[0]):
                valid_b = channel_mask[b]
                if valid_b.any():
                    d_valid = distance[b][valid_b]
                    d_mean = d_valid.mean()
                    d_std = d_valid.std(unbiased=False).clamp_min(1e-6)
                    distance_transformed[b][valid_b] = (d_valid - d_mean) / d_std
            anchor_logit_ez = alpha * (distance_transformed - self.beta)
        else:
            anchor_logit_ez = alpha * (distance - self.beta)

        # ---- gate: global or center-specific ----
        if self.center_gate and center_id is not None:
            cid = center_id.to(z.device).long().clamp(0, self.num_centers - 1)
            center_delta = self.center_gate_delta[cid].unsqueeze(1)  # [B, 1]
            gate = torch.sigmoid(self.gate_raw + center_delta)        # [B, 1]
        else:
            gate = torch.sigmoid(self.gate_raw)                       # scalar

        positive_label = str(positive_label).strip().lower()
        if positive_label == "ez":
            fused_logits = supervised_logits + gate * anchor_logit_ez
            score_ez = torch.sigmoid(fused_logits)
        elif positive_label == "nez":
            fused_logits = supervised_logits - gate * anchor_logit_ez
            score_ez = 1.0 - torch.sigmoid(fused_logits)
        else:
            raise ValueError(f"Unsupported positive_label={positive_label!r}; expected 'ez' or 'nez'.")

        valid = channel_mask & (labels_ez >= 0.0)
        ez_mask = valid & (labels_ez > 0.5)
        nez_mask = valid & (labels_ez <= 0.5)
        zero = supervised_logits.sum() * 0.0
        nez_loss = distance[nez_mask].mean() if torch.any(nez_mask) else zero
        ez_loss = F.relu(float(self.margin) - distance[ez_mask]).square().mean() if torch.any(ez_mask) else zero
        mean_d_ez = distance[ez_mask].mean() if torch.any(ez_mask) else zero
        mean_d_nez = distance[nez_mask].mean() if torch.any(nez_mask) else zero
        loss = nez_loss + float(self.gamma) * ez_loss

        # ---- per-center gate diagnostics ----
        if self.center_gate and center_id is not None:
            gate_per_patient = gate.squeeze(-1)  # [B]
            _gate_center = lambda mask: gate_per_patient[mask].mean() if mask.any() else gate_per_patient.mean() * 0.0
            gate_hup = _gate_center(cid == 0)
            gate_lzu = _gate_center(cid == 1)
            gate_multicenter = _gate_center(cid == 2)
            gate_pediatric = _gate_center(cid == 3)
            gate_mean = gate_per_patient.mean()
        else:
            gate_scalar = gate.detach()
            gate_per_patient = gate_scalar.expand(patient_channel_embedding.shape[0])
            gate_hup = gate_scalar
            gate_lzu = gate_scalar
            gate_multicenter = gate_scalar
            gate_pediatric = gate_scalar
            gate_mean = gate_scalar

        return {
            "logits": fused_logits,
            "score_ez_final": score_ez,
            "score_ez_anchor": torch.sigmoid(anchor_logit_ez).masked_fill(~channel_mask, 0.0),
            "score_ez_base": score_ez.new_tensor(0.0) + (torch.sigmoid(supervised_logits) if positive_label == "ez" else 1.0 - torch.sigmoid(supervised_logits)),
            "negative_anchor_distance": distance,
            "negative_anchor_loss": loss,
            "negative_anchor_gate": gate_mean,
            "negative_anchor_gate_per_patient": gate_per_patient,
            "negative_anchor_gate_hup": gate_hup,
            "negative_anchor_gate_lzu": gate_lzu,
            "negative_anchor_gate_multicenter": gate_multicenter,
            "negative_anchor_gate_pediatric": gate_pediatric,
            "negative_anchor_mean_d_ez": mean_d_ez,
            "negative_anchor_mean_d_nez": mean_d_nez,
            "negative_anchor_separation": mean_d_ez - mean_d_nez,
            "negative_anchor_alpha": alpha,
            "negative_anchor_beta": self.beta,
            "negative_anchor_logit_mean": anchor_logit_ez[valid].mean() if torch.any(valid) else zero,
        }


__all__ = ["NegativeAnchorHead"]
