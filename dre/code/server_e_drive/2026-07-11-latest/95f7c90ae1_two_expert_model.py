"""A9v5 Fold-Safe Two-Expert NAR-EZ model.

Expert-S (static physics, no anchor) and Expert-A (S5 physics + negative
anchor) run as independent NeuroEZCModel instances.  Their logits are fused
via center-specific lambdas (learned or fixed)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .model import NeuroEZCModel

_STATIC_FEATURES = (
    "early_high_gamma_slope,early_line_length_slope,"
    "onset_latency_high_gamma,onset_latency_line_length,"
    "onset_rank_high_gamma,onset_rank_line_length"
)
_STATIC_FEATURE_COUNT = 6


def _expert_args(base_args: Any, **overrides: Any) -> SimpleNamespace:
    """Clone *base_args* into a SimpleNamespace and apply overrides."""
    expert = SimpleNamespace()
    for key in vars(base_args):
        setattr(expert, key, getattr(base_args, key))
    for key, value in overrides.items():
        setattr(expert, key, value)
    return expert


class TwoExpertNeuroEZCModel(nn.Module):
    """Two-expert model with center-specific score-level fusion.

    Expert-S  – static physics features, no anchor branch.
    Expert-A  – S5 physics features + negative-anchor head (A9v3 config).
    """

    def __init__(self, args: Any) -> None:
        super().__init__()
        self.router_mode = str(getattr(args, "two_expert_router_mode", "center_learned")).lower()
        self.pediatric_preserve = bool(getattr(args, "two_expert_static_preserve_pediatric", False))
        self.pediatric_preserve_weight = float(getattr(args, "two_expert_pediatric_preserve_weight", 0.05))
        self.gate_l2 = float(getattr(args, "two_expert_gate_l2", 0.001))
        self.entropy_reg = float(getattr(args, "two_expert_entropy_reg", 0.0))
        self.positive_label = str(getattr(args, "positive_label", "ez")).lower()
        self.num_centers = 5

        # ---- Expert-S: static physics, no anchor ----
        args_s = _expert_args(
            args,
            physics_state_features=_STATIC_FEATURES,
            use_negative_anchor_head=False,
            use_diffusion_residual=False,
            use_view_gated_fusion=False,
            use_ez_ranking_loss=False,
            use_hard_topk_loss=False,
        )
        # force physics_dynamics on for Expert-S (it was on in base args)
        args_s.use_physics_dynamics = True
        self.expert_s = NeuroEZCModel(args_s)

        # ---- Expert-A: S5 physics + anchor (from base args) ----
        # base args already have S5 features and anchor config
        args_a = _expert_args(
            args,
            use_diffusion_residual=False,
            use_view_gated_fusion=False,
        )
        # Ensure anchor is enabled
        args_a.use_negative_anchor_head = True
        args_a.use_physics_dynamics = True
        self.expert_a = NeuroEZCModel(args_a)

        # ---- fusion: center-specific lambda ----
        if self.router_mode == "center_learned":
            gate_init_map = {
                0: float(getattr(args, "two_expert_gate_init_hup", 1.0)),
                1: float(getattr(args, "two_expert_gate_init_lzu", 0.5)),
                2: float(getattr(args, "two_expert_gate_init_multicenter", 1.0)),
                3: float(getattr(args, "two_expert_gate_init_pediatric", -3.0)),
                4: -6.0,
            }
            init_vals = [gate_init_map.get(i, 0.0) for i in range(self.num_centers)]
            self.center_lambda_raw = nn.Parameter(torch.tensor(init_vals, dtype=torch.float32))
        else:
            # center_fixed: lambdas provided via CLI
            fixed_map = {
                0: float(getattr(args, "two_expert_lambda_hup", 1.0)),
                1: float(getattr(args, "two_expert_lambda_lzu", 0.5)),
                2: float(getattr(args, "two_expert_lambda_multicenter", 1.0)),
                3: float(getattr(args, "two_expert_lambda_pediatric", 0.0)),
                4: 0.5,
            }
            fixed_vals = [fixed_map.get(i, 0.5) for i in range(self.num_centers)]
            self.register_buffer("center_lambda_fixed", torch.tensor(fixed_vals, dtype=torch.float32))

    def _get_lambda(self, center_id: torch.Tensor, device: torch.device) -> torch.Tensor:
        """Return per-patient lambda [B, 1]."""
        cid = center_id.to(device).long().clamp(0, self.num_centers - 1)
        if self.router_mode == "center_learned":
            lambda_raw = self.center_lambda_raw[cid]  # [B]
            return torch.sigmoid(lambda_raw).unsqueeze(1)  # [B, 1]
        else:
            return self.center_lambda_fixed[cid].unsqueeze(1)  # [B, 1]

    def _scores_from_logits(self, logits: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = torch.sigmoid(logits).masked_fill(~channel_mask, 0.0)
        if self.positive_label == "ez":
            score_ez = scores
            score_nez = 1.0 - scores
        else:
            score_nez = scores
            score_ez = 1.0 - scores
        return {
            "logits": logits,
            "scores": scores,
            "score_nez": score_nez.masked_fill(~channel_mask, 0.0),
            "score_ez": score_ez.masked_fill(~channel_mask, 0.0),
        }

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        channel_mask = batch["channel_mask"]
        center_id = batch.get("center_id")

        # ---- Expert-S: static physics only ----
        batch_s = dict(batch)
        phys_full = batch["physics_features"]
        batch_s["physics_features"] = phys_full[..., :_STATIC_FEATURE_COUNT]
        out_s = self.expert_s(batch_s)
        logit_s = out_s["logits"]

        # ---- Expert-A: full S5 physics + anchor ----
        out_a = self.expert_a(batch)
        logit_a = out_a["logits"]

        # ---- Fusion ----
        if center_id is not None:
            lambda_center = self._get_lambda(center_id, logit_s.device)  # [B, 1]
        else:
            lambda_center = torch.full((logit_s.shape[0], 1), 0.5, device=logit_s.device)

        logit_fused = (1.0 - lambda_center) * logit_s + lambda_center * logit_a

        # ---- build outputs ----
        fused = self._scores_from_logits(logit_fused, channel_mask)
        output: dict[str, torch.Tensor] = {
            "logits": fused["logits"],
            "scores": fused["scores"],
            "score_nez": fused["score_nez"],
            "score_ez": fused["score_ez"],
            "score_ez_static": out_s["score_ez"],
            "score_ez_anchor": out_a.get("score_ez_final", out_a["score_ez"]),
            "score_ez_anchor_final": out_a.get("score_ez_final", out_a["score_ez"]),
            "score_ez_final": fused["score_ez"],
            "patient_channel_embedding": out_a.get("patient_channel_embedding", out_s.get("patient_channel_embedding", logit_s)),
            "task_embedding": out_a.get("task_embedding", out_s.get("task_embedding", logit_s)),
        }

        # ---- per-center lambda diagnostics ----
        if center_id is not None:
            cid = center_id.to(logit_s.device).long().clamp(0, self.num_centers - 1)
            lam_per_patient = lambda_center.squeeze(-1)  # [B]
            _lam_center = lambda m: lam_per_patient[m].mean() if m.any() else lam_per_patient.mean() * 0.0
            output["two_expert_lambda_mean"] = lam_per_patient.mean()
            output["two_expert_lambda_hup"] = _lam_center(cid == 0)
            output["two_expert_lambda_lzu"] = _lam_center(cid == 1)
            output["two_expert_lambda_multicenter"] = _lam_center(cid == 2)
            output["two_expert_lambda_pediatric"] = _lam_center(cid == 3)
            output["two_expert_lambda_patient"] = lam_per_patient
            output["two_expert_lambda_per_patient"] = lam_per_patient
        else:
            lam_mean = lambda_center.mean().detach()
            output["two_expert_lambda_mean"] = lam_mean
            output["two_expert_lambda_hup"] = lam_mean
            output["two_expert_lambda_lzu"] = lam_mean
            output["two_expert_lambda_multicenter"] = lam_mean
            output["two_expert_lambda_pediatric"] = lam_mean
            output["two_expert_lambda_patient"] = lambda_center.squeeze(-1)
            output["two_expert_lambda_per_patient"] = lambda_center.squeeze(-1)

        # ---- forward anchor diagnostics from Expert-A ----
        for key, value in out_a.items():
            if key.startswith("negative_anchor_"):
                output[key] = value

        # ---- forward physics branch losses from both experts ----
        for key, value in out_s.items():
            if key.endswith("_loss") or key.endswith("_mean") or key == "physics_gate_mean":
                output[f"expert_s_{key}"] = value
        for key, value in out_a.items():
            if (key.endswith("_loss") or key.endswith("_mean") or key == "physics_gate_mean") and not key.startswith("negative_anchor"):
                output[f"expert_a_{key}"] = value

        # ---- pediatric preserve loss ----
        if self.pediatric_preserve and center_id is not None:
            ped_mask = (center_id.to(logit_s.device) == 3)
            if ped_mask.any():
                preserve_loss = F.mse_loss(
                    fused["score_ez"][ped_mask],
                    out_s["score_ez"][ped_mask],
                )
                output["two_expert_pediatric_preserve_loss"] = preserve_loss

        # ---- gate L2 regularisation (learned mode) ----
        if self.router_mode == "center_learned":
            gate_l2_loss = self.center_lambda_raw.square().mean()
            output["two_expert_gate_l2_loss"] = gate_l2_loss

        # ---- entropy regularisation (learned mode) ----
        if self.router_mode == "center_learned":
            lam = torch.sigmoid(self.center_lambda_raw)
            entropy = -(lam * torch.log(lam.clamp_min(1e-7)) + (1 - lam) * torch.log((1 - lam).clamp_min(1e-7))).mean()
            output["two_expert_entropy_loss"] = -entropy

        return output


__all__ = ["TwoExpertNeuroEZCModel"]
