"""NeuroEZ-C patient-level B0-Pruned model for EZ/NEZ localization.

Supports:
- single-expert (default)
- A9v5 same-embedding two-head router (use_two_expert_router)
- A9v6 feature-separated two-expert (use_feature_separated_two_expert)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from graph_spectral_encoder import WindowGraphSpectralEncoder
from patient_channel_ranker import PatientChannelClassifier
from seizure_aggregator import CrossSeizureMILAggregator
from temporal_encoder import ChannelTemporalEncoder
from .diffusion_residual import DiffusionSourceResidualEncoder
from .negative_anchor import NegativeAnchorHead
from .physics_dynamics import NeuralDynamicsResidualEncoder
from .raw_window_encoder import RawWindowEncoder
from .view_gated_fusion import ViewGatedFusion
from .v3_qbc_profiles import get_v3_qbc_profile
from .v3_rcc_profiles import get_v3_rcc_profile


def _parse_feature_slice(spec: str, total_dim: int) -> slice:
    """Parse a feature-slice string like ``"0:6"`` or ``"all"``.

    Returns a :class:`slice` that can index the last dimension of a tensor.
    Raises :exc:`ValueError` if the slice is out of range.
    """
    spec = str(spec).strip().lower()
    if spec == "all":
        return slice(0, total_dim)
    parts = spec.split(":")
    if len(parts) != 2:
        raise ValueError(f"Invalid feature slice spec {spec!r}; expected 'start:end' or 'all'.")
    start = int(parts[0])
    end = int(parts[1])
    if start < 0 or end > total_dim or start >= end:
        raise ValueError(
            f"Feature slice {spec!r} out of range for total_dim={total_dim}."
        )
    return slice(start, end)


class NeuroEZCModel(nn.Module):
    """B0-Pruned patient-level spectral/classical model for EZ/NEZ localization."""

    def __init__(self, args: Any | None = None) -> None:
        super().__init__()
        self.args = args
        self.model_dim = int(getattr(args, "model_dim", 32) if args is not None else 32)
        dropout = float(getattr(args, "dropout", 0.40) if args is not None else 0.40)
        num_heads = int(getattr(args, "num_heads", 2) if args is not None else 2)
        use_channel_attention = bool(getattr(args, "use_channel_attention", True) if args is not None else True)
        self.use_physics_dynamics = bool(getattr(args, "use_physics_dynamics", False) if args is not None else False)
        self.use_diffusion_residual = bool(getattr(args, "use_diffusion_residual", False) if args is not None else False)
        self.use_negative_anchor_head = bool(getattr(args, "use_negative_anchor_head", False) if args is not None else False)
        self.use_view_gated_fusion = bool(getattr(args, "use_view_gated_fusion", False) if args is not None else False)
        self.use_two_expert_router = bool(getattr(args, "use_two_expert_router", False) if args is not None else False)
        self.use_feature_separated_two_expert = bool(getattr(args, "use_feature_separated_two_expert", False) if args is not None else False)
        self.use_a9v8_lcbo = bool(getattr(args, "use_a9v8_lcbo", False) if args is not None else False)
        self.use_n6_dual_view_ema = bool(getattr(args, "use_n6_dual_view_ema", False) if args is not None else False)
        self.use_v3_qbc = bool(getattr(args, "use_v3_qbc", False) if args is not None else False)
        self.use_v3_rcc = bool(getattr(args, "use_v3_rcc", False) if args is not None else False)
        self.v3_qbc_profile = get_v3_qbc_profile(
            getattr(args, "v3_qbc_profile", "BCR_BC_ONLY") if args is not None else "BCR_BC_ONLY"
        )
        self.v3_rcc_profile = get_v3_rcc_profile(getattr(args, "v3_rcc_profile", "R0_BASE") if args is not None else "R0_BASE")
        self.eval_score_fusion_gamma = float(getattr(args, "eval_score_fusion_gamma", 0.10) if args is not None else 0.10)
        self.two_expert_router_mode = str(getattr(args, "two_expert_router_mode", "center_learned") if args is not None else "center_learned").lower()
        self.diffusion_score_residual = bool(getattr(args, "diffusion_score_residual", False) if args is not None else False)
        self.diffusion_center_mode = str(getattr(args, "diffusion_center_mode", "all") if args is not None else "all").lower()
        self.positive_label = str(getattr(args, "positive_label", "nez") if args is not None else "nez").lower()
        if self.use_n6_dual_view_ema:
            incompatible = {
                "positive_label": self.positive_label != "nez",
                "use_two_expert_router": self.use_two_expert_router,
                "use_feature_separated_two_expert": self.use_feature_separated_two_expert,
                "use_a9v8_lcbo": self.use_a9v8_lcbo,
                "use_negative_anchor_head": self.use_negative_anchor_head,
                "use_diffusion_residual": self.use_diffusion_residual,
                "use_broad_ez_mil_loss": bool(getattr(args, "use_broad_ez_mil_loss", False)),
                "group_robust_mode": str(getattr(args, "group_robust_mode", "none")).lower() != "none",
            }
            invalid = [name for name, enabled in incompatible.items() if enabled]
            if invalid:
                raise ValueError(f"N6 dual-view configuration is incompatible with: {invalid}")
        if self.use_a9v8_lcbo and self.positive_label != "ez":
            raise ValueError("A9v8 LCBO requires --positive_label ez so broad/core logits have EZ semantics.")
        if self.use_v3_qbc or self.use_v3_rcc:
            if self.positive_label != "ez":
                raise ValueError("V3-RCC/QBC requires the V3 TrueBest --positive_label ez contract.")
            incompatible = {
                "use_n6_dual_view_ema": self.use_n6_dual_view_ema,
                "use_two_expert_router": self.use_two_expert_router,
                "use_feature_separated_two_expert": self.use_feature_separated_two_expert,
                "use_a9v8_lcbo": self.use_a9v8_lcbo,
                "use_diffusion_residual": self.use_diffusion_residual,
                "use_view_gated_fusion": self.use_view_gated_fusion,
            }
            invalid = sorted(key for key, value in incompatible.items() if value)
            if invalid:
                raise ValueError(f"V3-QBC is incompatible with: {invalid}")

        # Each PRQ/BCR run instantiates this model separately; no branch weights
        # are shared across their independently trained encoders.
        self.b0_encoder = WindowGraphSpectralEncoder(
            model_dim=self.model_dim,
            num_heads=num_heads,
            dropout=dropout,
            use_channel_attention=use_channel_attention,
        )

        temporal_kwargs = dict(
            model_dim=self.model_dim,
            pooling=str(getattr(args, "temporal_pooling", "mean") if args is not None else "mean"),
            channel_pooling_mode=str(getattr(args, "channel_pooling_mode", "mean") if args is not None else "mean"),
            topk_fraction=float(getattr(args, "temporal_topk_fraction", 0.20) if args is not None else 0.20),
            top_p=float(getattr(args, "temporal_pooling_top_p", 0.10) if args is not None else 0.10),
            tau=float(getattr(args, "temporal_pooling_tau", 0.10) if args is not None else 0.10),
            lse_pool_tau=float(getattr(args, "lse_pool_tau", 1.0) if args is not None else 1.0),
            early_pool_frac=float(getattr(args, "early_pool_frac", 0.25) if args is not None else 0.25),
        )
        record_pooling_kwargs = dict(
            pooling=str(getattr(args, "record_pooling", "mean") if args is not None else "mean"),
            top_p=float(getattr(args, "record_pooling_top_p", 0.30) if args is not None else 0.30),
            alpha=float(getattr(args, "record_pooling_alpha", 0.70) if args is not None else 0.70),
        )
        classifier_kwargs = dict(
            input_dim=self.seizure_aggregator_output_dim(args) if self.use_feature_separated_two_expert else 0,
            num_heads=num_heads,
            dropout=dropout,
            use_patient_relative_z=bool(getattr(args, "use_patient_relative_z", True) if args is not None else True),
            positive_label=self.positive_label,
            use_a9v8_lcbo=self.use_a9v8_lcbo,
            eval_score_fusion_gamma=self.eval_score_fusion_gamma,
        )

        # ---- feature-separated two-expert (A9v6) ----
        if self.use_feature_separated_two_expert:
            if self.two_expert_router_mode not in {"center_learned", "center_fixed"}:
                raise ValueError(f"Unsupported two_expert_router_mode={self.two_expert_router_mode!r}.")
            self._static_feature_slice_spec = str(getattr(args, "two_expert_static_feature_slice", "0:6"))
            self._anchor_feature_slice_spec = str(getattr(args, "two_expert_anchor_feature_slice", "all"))

            agg_out_dim = self.seizure_aggregator_output_dim(args)

            # static expert modules
            self.physics_encoder_static = NeuralDynamicsResidualEncoder(args, model_dim=self.model_dim)
            self.physics_gate_static = nn.Linear(2 * self.model_dim, self.model_dim)
            nn.init.zeros_(self.physics_gate_static.weight)
            nn.init.constant_(self.physics_gate_static.bias, float(getattr(args, "physics_gate_init", -3.0) if args is not None else -3.0))
            self.m1_norm_static = nn.LayerNorm(self.model_dim)
            self.temporal_encoder_static = ChannelTemporalEncoder(**temporal_kwargs)
            self.seizure_aggregator_static = CrossSeizureMILAggregator(model_dim=self.model_dim, **record_pooling_kwargs)
            classifier_kwargs["input_dim"] = agg_out_dim
            self.static_channel_classifier = PatientChannelClassifier(**classifier_kwargs)

            # anchor expert modules
            self.physics_encoder_anchor = NeuralDynamicsResidualEncoder(args, model_dim=self.model_dim)
            self.physics_gate_anchor = nn.Linear(2 * self.model_dim, self.model_dim)
            nn.init.zeros_(self.physics_gate_anchor.weight)
            nn.init.constant_(self.physics_gate_anchor.bias, float(getattr(args, "physics_gate_init", -3.0) if args is not None else -3.0))
            self.m1_norm_anchor = nn.LayerNorm(self.model_dim)
            self.temporal_encoder_anchor = ChannelTemporalEncoder(**temporal_kwargs)
            self.seizure_aggregator_anchor = CrossSeizureMILAggregator(model_dim=self.model_dim, **record_pooling_kwargs)
            self.channel_classifier = PatientChannelClassifier(**classifier_kwargs)

            # anchor head (on anchor expert only)
            if self.use_negative_anchor_head:
                self.negative_anchor_head = NegativeAnchorHead.from_args(args, input_dim=agg_out_dim)

            # lambda fusion
            self._init_two_expert_fusion(args)
            return  # feature-separated path fully initialised

        # ---- single-expert physics / diffusion / classifier ----
        if self.use_physics_dynamics and not self.use_view_gated_fusion:
            self.physics_encoder = NeuralDynamicsResidualEncoder(args, model_dim=self.model_dim)
        if self.use_physics_dynamics or self.use_view_gated_fusion:
            self.physics_gate = nn.Linear(2 * self.model_dim, self.model_dim)
            nn.init.zeros_(self.physics_gate.weight)
            nn.init.constant_(self.physics_gate.bias, float(getattr(args, "physics_gate_init", -3.0) if args is not None else -3.0))
        if self.use_view_gated_fusion:
            self.view_gated_fusion = ViewGatedFusion(args, model_dim=self.model_dim)
        if self.use_diffusion_residual:
            self.diffusion_encoder = DiffusionSourceResidualEncoder(args, model_dim=self.model_dim)
            if self.diffusion_score_residual:
                self.diffusion_score_gate_raw = nn.Parameter(
                    torch.tensor(float(getattr(args, "diffusion_center_gate_init", -4.0) if args is not None else -4.0))
                )
            else:
                self.diffusion_gate = nn.Linear(2 * self.model_dim, self.model_dim)
                nn.init.zeros_(self.diffusion_gate.weight)
                nn.init.constant_(self.diffusion_gate.bias, float(getattr(args, "diffusion_gate_init", -4.0) if args is not None else -4.0))
        self.m1_norm = nn.LayerNorm(self.model_dim)
        self.final_norm = nn.LayerNorm(self.model_dim)
        self.temporal_encoder = ChannelTemporalEncoder(**temporal_kwargs)
        self.seizure_aggregator = CrossSeizureMILAggregator(model_dim=self.model_dim, **record_pooling_kwargs)
        if self.use_diffusion_residual and self.diffusion_score_residual:
            self.diffusion_score_head = nn.Linear(self.seizure_aggregator.output_dim, 1)

        agg_out_dim = self.seizure_aggregator.output_dim
        classifier_kwargs["input_dim"] = agg_out_dim
        self.channel_classifier = PatientChannelClassifier(**classifier_kwargs)

        if self.use_n6_dual_view_ema:
            self.raw_window_encoder = RawWindowEncoder(
                model_dim=self.model_dim,
                dropout=float(getattr(args, "raw_encoder_dropout", 0.20)),
                chunk_size=int(getattr(args, "raw_encoder_chunk_size", 4096)),
                gradient_checkpointing=bool(getattr(args, "raw_encoder_gradient_checkpointing", True)),
            )
            self.raw_temporal_encoder = ChannelTemporalEncoder(**temporal_kwargs)
            self.raw_seizure_aggregator = CrossSeizureMILAggregator(model_dim=self.model_dim, **record_pooling_kwargs)
            self.raw_channel_classifier = PatientChannelClassifier(**classifier_kwargs)
            self.feature_fusion_projection = nn.Identity()
            self.raw_fusion_projection = nn.Identity()
            self.feature_fusion_norm = nn.LayerNorm(agg_out_dim)
            self.raw_fusion_norm = nn.LayerNorm(agg_out_dim)
            self.fusion_gate = nn.Sequential(
                nn.Linear(3 * agg_out_dim + 2, agg_out_dim),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(agg_out_dim, 1),
            )
            initial_gate = float(getattr(args, "n6_initial_feature_gate", 0.75))
            gate_fraction = (initial_gate - 0.05) / 0.90
            if not 0.0 < gate_fraction < 1.0:
                raise ValueError("n6_initial_feature_gate must be strictly inside [0.05, 0.95]")
            nn.init.zeros_(self.fusion_gate[-1].weight)
            nn.init.constant_(self.fusion_gate[-1].bias, torch.logit(torch.tensor(gate_fraction)).item())

        # ---- A9v5 same-embedding two-head router ----
        if self.use_two_expert_router:
            self._init_two_expert_fusion(args)
            self.static_channel_classifier = PatientChannelClassifier(**classifier_kwargs)

        if self.use_negative_anchor_head and not self.use_feature_separated_two_expert:
            self.negative_anchor_head = NegativeAnchorHead.from_args(args, input_dim=agg_out_dim)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def seizure_aggregator_output_dim(args: Any | None) -> int:
        """Return the output dim of CrossSeizureMILAggregator without
        materialising one.  Keep in sync with CrossSeizureMILAggregator."""
        base = int(getattr(args, "model_dim", 32) if args is not None else 32)
        return base * 2  # CrossSeizureMILAggregator doubles model_dim

    def _init_two_expert_fusion(self, args: Any) -> None:
        """Shared lambda-fusion initialisation for A9v5 and A9v6."""
        if self.two_expert_router_mode not in {"center_learned", "center_fixed"}:
            raise ValueError(f"Unsupported two_expert_router_mode={self.two_expert_router_mode!r}.")
        self.register_buffer(
            "two_expert_lambda_fixed",
            torch.tensor(
                [
                    float(getattr(args, "two_expert_lambda_hup", 1.0) if args is not None else 1.0),
                    float(getattr(args, "two_expert_lambda_lzu", 0.5) if args is not None else 0.5),
                    float(getattr(args, "two_expert_lambda_multicenter", 1.0) if args is not None else 1.0),
                    float(getattr(args, "two_expert_lambda_pediatric", 0.0) if args is not None else 0.0),
                    0.5,
                ],
                dtype=torch.float32,
            ),
        )
        if self.two_expert_router_mode == "center_learned":
            self.two_expert_gate_raw = nn.Parameter(
                torch.tensor(
                    [
                        float(getattr(args, "two_expert_gate_init_hup", 1.0) if args is not None else 1.0),
                        float(getattr(args, "two_expert_gate_init_lzu", 0.5) if args is not None else 0.5),
                        float(getattr(args, "two_expert_gate_init_multicenter", 1.0) if args is not None else 1.0),
                        float(getattr(args, "two_expert_gate_init_pediatric", -3.0) if args is not None else -3.0),
                        -6.0,
                    ],
                    dtype=torch.float32,
                )
            )

    def _scores_from_logits(self, logits: torch.Tensor, channel_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        scores = torch.sigmoid(logits).masked_fill(~channel_mask, 0.0)
        if self.positive_label == "ez":
            score_ez = scores
            score_nez = 1.0 - scores
        elif self.positive_label == "nez":
            score_nez = scores
            score_ez = 1.0 - scores
        else:
            raise ValueError(f"Unsupported positive_label={self.positive_label!r}; expected 'ez' or 'nez'.")
        return {
            "logits": logits,
            "scores": scores,
            "score_nez": score_nez.masked_fill(~channel_mask, 0.0),
            "score_ez": score_ez.masked_fill(~channel_mask, 0.0),
        }

    def _diffusion_center_mask(self, batch: dict[str, Any], logits: torch.Tensor) -> torch.Tensor:
        mode = self.diffusion_center_mode
        center_id = batch.get("center_id")
        if center_id is None:
            center_id = torch.full((logits.shape[0],), 4, dtype=torch.long, device=logits.device)
        else:
            center_id = center_id.to(logits.device).long()
        if mode == "all":
            patient_mask = torch.ones_like(center_id, dtype=torch.bool)
        elif mode == "lzu_only":
            patient_mask = center_id == 1
        elif mode == "non_hup":
            patient_mask = center_id != 0
        elif mode == "none":
            patient_mask = torch.zeros_like(center_id, dtype=torch.bool)
        else:
            raise ValueError(f"Unsupported diffusion_center_mode={mode!r}.")
        return patient_mask[:, None].expand_as(logits)

    def _center_id(self, batch: dict[str, Any], batch_size: int, device: torch.device) -> torch.Tensor:
        center_id = batch.get("center_id")
        if center_id is None:
            return torch.full((batch_size,), 4, dtype=torch.long, device=device)
        return center_id.to(device).long().clamp(0, 4)

    def _two_expert_lambda(self, batch: dict[str, Any], logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        center_id = self._center_id(batch, logits.shape[0], logits.device)
        if self.two_expert_router_mode == "center_learned":
            lambda_patient = torch.sigmoid(self.two_expert_gate_raw[center_id])
        else:
            lambda_patient = self.two_expert_lambda_fixed[center_id].to(logits.device, dtype=logits.dtype)
        return lambda_patient.to(dtype=logits.dtype), center_id

    def _two_expert_center_diagnostics(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.two_expert_router_mode == "center_learned":
            lambdas = torch.sigmoid(self.two_expert_gate_raw).to(device=logits.device, dtype=logits.dtype)
            return {
                "two_expert_lambda_hup": lambdas[0],
                "two_expert_lambda_lzu": lambdas[1],
                "two_expert_lambda_multicenter": lambdas[2],
                "two_expert_lambda_pediatric": lambdas[3],
                "two_expert_gate_l2_loss": self.two_expert_gate_raw.square().mean(),
                "two_expert_entropy_loss": -(
                    lambdas * torch.log(lambdas.clamp_min(1e-7))
                    + (1.0 - lambdas) * torch.log((1.0 - lambdas).clamp_min(1e-7))
                ).mean(),
            }
        lambdas = self.two_expert_lambda_fixed.to(device=logits.device, dtype=logits.dtype)
        zero = logits.sum() * 0.0
        return {
            "two_expert_lambda_hup": lambdas[0],
            "two_expert_lambda_lzu": lambdas[1],
            "two_expert_lambda_multicenter": lambdas[2],
            "two_expert_lambda_pediatric": lambdas[3],
            "two_expert_gate_l2_loss": zero,
            "two_expert_entropy_loss": zero,
        }

    def _ez_score_from_logits(self, logits: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
        return self._scores_from_logits(logits, channel_mask)["score_ez"]

    # ------------------------------------------------------------------
    # forward
    # ------------------------------------------------------------------

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        if self.use_n6_dual_view_ema:
            return self._forward_n6_dual_view(batch)
        # ---- feature-separated two-expert (A9v6) ----
        if self.use_feature_separated_two_expert:
            return self._forward_feature_separated(batch)

        # ---- single-expert path (original + A9v5 two-head) ----
        return self._forward_single_expert(batch)

    def _forward_n6_dual_view(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Run independent feature/raw branches and channel-wise late fusion."""
        required = ("raw_windows", "raw_window_mask", "raw_seizure_channel_mask")
        missing = [key for key in required if key not in batch]
        if missing:
            raise KeyError(f"N6 dual-view batch is missing raw tensors: {missing}")
        channel_mask = batch["channel_mask"].to(dtype=torch.bool)
        seizure_channel_mask = batch["seizure_channel_mask"].to(dtype=torch.bool)
        h_b0 = self.b0_encoder(batch["b0_features"], None, seizure_channel_mask)
        branch_losses: dict[str, torch.Tensor] = {}
        if self.use_physics_dynamics:
            h_dyn, physics_losses = self.physics_encoder(
                batch["physics_features"], batch.get("window_mask"), seizure_channel_mask, batch.get("window_centers"),
            )
            gate = torch.sigmoid(self.physics_gate(torch.cat([h_b0, h_dyn], dim=-1)))
            feature_windows = self.m1_norm(h_b0 + gate * h_dyn)
            branch_losses.update(physics_losses)
            branch_losses["physics_gate_mean"] = gate.mean()
        else:
            feature_windows = self.m1_norm(h_b0)
        feature_seizure, feature_temporal_attention = self.temporal_encoder(
            feature_windows, seizure_channel_mask, window_mask=batch.get("window_mask"),
        )
        feature_patient, feature_seizure_attention = self.seizure_aggregator(
            feature_seizure, batch["seizure_mask"], seizure_channel_mask,
        )
        feature_output = self.channel_classifier(feature_patient, channel_mask)
        logits_feature = feature_output["logits"]

        raw_mask = batch["raw_window_mask"].to(device=logits_feature.device, dtype=torch.bool)
        raw_seizure_channel_mask = batch["raw_seizure_channel_mask"].to(device=logits_feature.device, dtype=torch.bool)
        raw_channel_available = raw_mask.any(dim=(1, 3)) & channel_mask
        raw_embeddings = self.raw_window_encoder(batch["raw_windows"], raw_mask).permute(0, 1, 3, 2, 4)
        raw_seizure, raw_temporal_attention = self.raw_temporal_encoder(
            raw_embeddings, raw_seizure_channel_mask, window_mask=raw_mask.any(dim=2),
        )
        raw_patient, raw_seizure_attention = self.raw_seizure_aggregator(
            raw_seizure, batch["seizure_mask"], raw_seizure_channel_mask,
        )
        raw_output = self.raw_channel_classifier(raw_patient, raw_channel_available)
        logits_raw = raw_output["logits"].masked_fill(~raw_channel_available, 0.0)

        feature_embedding = self.feature_fusion_projection(feature_patient)
        raw_embedding = self.raw_fusion_projection(raw_patient)
        feature_norm = self.feature_fusion_norm(feature_embedding)
        raw_norm = self.raw_fusion_norm(raw_embedding)
        gate_input = torch.cat([
            feature_norm,
            raw_norm,
            (feature_norm - raw_norm).abs(),
            torch.sigmoid(logits_feature).unsqueeze(-1),
            torch.sigmoid(logits_raw).unsqueeze(-1),
        ], dim=-1)
        fusion_gate = 0.05 + 0.90 * torch.sigmoid(self.fusion_gate(gate_input).squeeze(-1))
        fusion_gate = torch.where(raw_channel_available, fusion_gate, torch.ones_like(fusion_gate)).masked_fill(~channel_mask, 0.0)
        logits = (fusion_gate * logits_feature + (1.0 - fusion_gate) * logits_raw).masked_fill(~channel_mask, 0.0)
        output = self._scores_from_logits(logits, channel_mask)
        output.update({
            "logits": logits,
            "logits_feature": logits_feature,
            "logits_raw": logits_raw,
            "score_nez_feature": self._scores_from_logits(logits_feature, channel_mask)["score_nez"],
            "score_ez_feature": self._scores_from_logits(logits_feature, channel_mask)["score_ez"],
            "score_nez_raw": self._scores_from_logits(logits_raw, raw_channel_available)["score_nez"],
            "score_ez_raw": self._scores_from_logits(logits_raw, raw_channel_available)["score_ez"],
            "score_nez_fused": output["score_nez"],
            "fusion_gate_feature": fusion_gate,
            "raw_channel_available": raw_channel_available,
            "feature_patient_channel_embedding": feature_patient,
            "raw_patient_channel_embedding": raw_patient,
            "patient_channel_embedding": fusion_gate.unsqueeze(-1) * feature_embedding + (1.0 - fusion_gate).unsqueeze(-1) * raw_embedding,
            "seizure_channel_embedding": feature_seizure,
            "feature_temporal_attention": feature_temporal_attention,
            "raw_temporal_attention": raw_temporal_attention,
            "feature_seizure_attention": feature_seizure_attention,
            "raw_seizure_attention": raw_seizure_attention,
        })
        output.update(branch_losses)
        return output

    # ------------------------------------------------------------------
    # A9v6: feature-separated two-expert forward
    # ------------------------------------------------------------------

    def _forward_feature_separated(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        channel_mask = batch["channel_mask"]
        seizure_channel_mask = batch["seizure_channel_mask"]
        phys_all = batch.get("physics_features")
        if phys_all is None:
            raise KeyError("Batch is missing physics_features for feature-separated two-expert.")

        phys_dim = phys_all.shape[-1]
        static_slice = _parse_feature_slice(
            str(getattr(self, "_static_feature_slice_spec", "0:6")), phys_dim,
        )
        anchor_slice = _parse_feature_slice(
            str(getattr(self, "_anchor_feature_slice_spec", "all")), phys_dim,
        )

        # ---- shared b0 ----
        h_b0 = self.b0_encoder(batch["b0_features"], None, seizure_channel_mask)

        # ==================================================================
        # Expert-S: static physics features only
        # ==================================================================
        phys_static = phys_all[..., static_slice]

        h_dyn_s, losses_s = self.physics_encoder_static(
            phys_static, batch.get("window_mask"), seizure_channel_mask, batch.get("window_centers"),
        )
        gate_s = torch.sigmoid(self.physics_gate_static(torch.cat([h_b0, h_dyn_s], dim=-1)))
        h_static = self.m1_norm_static(h_b0 + gate_s * h_dyn_s)

        sez_s, tmp_s = self.temporal_encoder_static(h_static, seizure_channel_mask, window_mask=batch.get("window_mask"))
        pat_s, sw_s = self.seizure_aggregator_static(sez_s, batch["seizure_mask"], seizure_channel_mask)
        static_output = self.static_channel_classifier(pat_s, channel_mask)
        logit_static = static_output["logits"]

        # ==================================================================
        # Expert-A: full S5 physics + anchor
        # ==================================================================
        phys_anchor = phys_all[..., anchor_slice]

        h_dyn_a, losses_a = self.physics_encoder_anchor(
            phys_anchor, batch.get("window_mask"), seizure_channel_mask, batch.get("window_centers"),
        )
        gate_a = torch.sigmoid(self.physics_gate_anchor(torch.cat([h_b0, h_dyn_a], dim=-1)))
        h_anchor = self.m1_norm_anchor(h_b0 + gate_a * h_dyn_a)

        sez_a, tmp_a = self.temporal_encoder_anchor(h_anchor, seizure_channel_mask, window_mask=batch.get("window_mask"))
        pat_a, sw_a = self.seizure_aggregator_anchor(sez_a, batch["seizure_mask"], seizure_channel_mask)
        anchor_output = self.channel_classifier(pat_a, channel_mask)
        logit_anchor_base = anchor_output["logits"]

        anchor_extra: dict[str, torch.Tensor] = {}
        if self.use_negative_anchor_head:
            anchor_extra = self.negative_anchor_head(
                pat_a, batch["labels_ez"], channel_mask, logit_anchor_base,
                positive_label=self.positive_label, center_id=batch.get("center_id"),
            )
            logit_anchor_final = anchor_extra["logits"]
        else:
            logit_anchor_final = logit_anchor_base

        # ==================================================================
        # Lambda fusion
        # ==================================================================
        lambda_patient, center_id = self._two_expert_lambda(batch, logit_anchor_final)
        lambda_channel = lambda_patient[:, None]
        final_logits = (1.0 - lambda_channel) * logit_static + lambda_channel * logit_anchor_final
        final_scores = self._scores_from_logits(final_logits, channel_mask)

        anchor_scores_base = self._scores_from_logits(logit_anchor_base, channel_mask)

        # ==================================================================
        # Outputs
        # ==================================================================
        zero = final_logits.sum() * 0.0
        output: dict[str, torch.Tensor] = {}
        output.update(anchor_extra)
        output.update(final_scores)
        output.update({
            "logits": final_logits,
            "logits_static": logit_static,
            "logits_anchor": logit_anchor_final,
            "score_ez_static": static_output["score_ez"],
            "score_ez_anchor": self._scores_from_logits(logit_anchor_final, channel_mask)["score_ez"],
            "score_ez_anchor_final": self._scores_from_logits(logit_anchor_final, channel_mask)["score_ez"],
            "score_ez_final": final_scores["score_ez"],
            "score_ez_base": anchor_scores_base["score_ez"],
            "two_expert_lambda_patient": lambda_patient,
            "two_expert_lambda_per_patient": lambda_patient,
            "two_expert_lambda_mean": lambda_patient.mean(),
            "expert_s_physics_gate_mean": gate_s.mean(),
            "expert_a_physics_gate_mean": gate_a.mean(),
            "two_expert_static_feature_dim": torch.tensor(
                static_slice.stop - static_slice.start if static_slice.start is not None else phys_dim,
                dtype=torch.float32, device=final_logits.device,
            ),
            "two_expert_anchor_feature_dim": torch.tensor(
                anchor_slice.stop - anchor_slice.start if anchor_slice.start is not None else phys_dim,
                dtype=torch.float32, device=final_logits.device,
            ),
        })
        output.update(self._two_expert_center_diagnostics(final_logits))

        # forward physics branch losses
        for key, value in losses_s.items():
            output[f"expert_s_{key}"] = value
        for key, value in losses_a.items():
            output[f"expert_a_{key}"] = value

        # pediatric preserve loss
        ped_mask = center_id == 3
        if torch.any(ped_mask):
            output["two_expert_pediatric_preserve_loss"] = F.mse_loss(
                final_scores["score_ez"][ped_mask], static_output["score_ez"][ped_mask],
            )
        else:
            output["two_expert_pediatric_preserve_loss"] = zero

        output.update({
            "patient_channel_embedding": pat_a,
            "task_embedding": pat_a,
        })
        return output

    # ------------------------------------------------------------------
    # single-expert forward (original + A9v5 two-head)
    # ------------------------------------------------------------------

    def _forward_single_expert(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        seizure_channel_mask = batch["seizure_channel_mask"]
        h_b0 = self.b0_encoder(batch["b0_features"], None, seizure_channel_mask)
        branch_losses: dict[str, torch.Tensor] = {}
        if self.use_view_gated_fusion:
            if "physics_features" not in batch:
                raise KeyError("Batch is missing physics_features while use_view_gated_fusion=True.")
            h_dyn, view_losses = self.view_gated_fusion(batch["physics_features"], seizure_channel_mask)
            gate = torch.sigmoid(self.physics_gate(torch.cat([h_b0, h_dyn], dim=-1)))
            h_m1 = self.m1_norm(h_b0 + gate * h_dyn)
            branch_losses.update(view_losses)
            branch_losses["physics_gate_mean"] = gate.mean()
        elif self.use_physics_dynamics:
            if "physics_features" not in batch:
                raise KeyError("Batch is missing physics_features while use_physics_dynamics=True.")
            h_dyn, physics_losses = self.physics_encoder(
                batch["physics_features"], batch.get("window_mask"), seizure_channel_mask, batch.get("window_centers"),
            )
            gate = torch.sigmoid(self.physics_gate(torch.cat([h_b0, h_dyn], dim=-1)))
            h_m1 = self.m1_norm(h_b0 + gate * h_dyn)
            branch_losses.update(physics_losses)
            branch_losses["physics_gate_mean"] = gate.mean()
        else:
            h_m1 = self.m1_norm(h_b0)
        h_diff_for_score: torch.Tensor | None = None
        if self.use_diffusion_residual:
            if "physics_features" not in batch:
                raise KeyError("Batch is missing physics_features while use_diffusion_residual=True.")
            h_diff, diffusion_losses = self.diffusion_encoder(
                batch["physics_features"], batch.get("diffusion_adjacency"),
                batch.get("window_mask"), seizure_channel_mask, batch.get("window_centers"),
            )
            branch_losses.update(diffusion_losses)
            if self.diffusion_score_residual:
                h_diff_for_score = h_diff
                fused = h_m1
                branch_losses["diffusion_gate_mean"] = torch.sigmoid(self.diffusion_score_gate_raw)
            else:
                gate_diff = torch.sigmoid(self.diffusion_gate(torch.cat([h_m1, h_diff], dim=-1)))
                fused = self.final_norm(h_m1 + gate_diff * h_diff)
                branch_losses["diffusion_gate_mean"] = gate_diff.mean()
        else:
            fused = h_m1
        seizure_channel_embedding, temporal_weights = self.temporal_encoder(
            fused, seizure_channel_mask, window_mask=batch.get("window_mask"),
        )
        n_invalid_windows = torch.tensor(
            float(getattr(self.temporal_encoder, "last_n_channels_all_windows_invalid", 0)),
            dtype=fused.dtype,
            device=fused.device,
        )
        quality_aggregation = str(getattr(self.args, "quality_weight_record_aggregation", "weighted_mean_std")).lower()
        quality_weights = None
        if bool(getattr(self.args, "use_edf_quality_weighting", False)) and quality_aggregation != "none":
            quality_weights = batch.get("record_quality_weight_active", batch.get("record_quality_weight"))
        patient_channel_embedding, seizure_weights = self.seizure_aggregator(
            seizure_channel_embedding, batch["seizure_mask"], seizure_channel_mask, seizure_weights=quality_weights,
        )
        if self.use_a9v8_lcbo:
            output = self.channel_classifier(patient_channel_embedding, batch["channel_mask"])
            output.update({
                "patient_channel_embedding": patient_channel_embedding,
                "task_embedding": patient_channel_embedding,
                "seizure_channel_embedding": seizure_channel_embedding,
                "temporal_attention": temporal_weights,
                "seizure_attention": seizure_weights,
                "n_channels_all_windows_invalid": n_invalid_windows,
            })
            output.update(branch_losses)
            return output
        output = self.channel_classifier(patient_channel_embedding, batch["channel_mask"])
        if h_diff_for_score is not None:
            graph_seizure_embedding, _ = self.temporal_encoder(
                h_diff_for_score, seizure_channel_mask, window_mask=batch.get("window_mask"),
            )
            graph_patient_embedding, _ = self.seizure_aggregator(
                graph_seizure_embedding, batch["seizure_mask"], seizure_channel_mask, seizure_weights=quality_weights,
            )
            graph_delta_logits = self.diffusion_score_head(graph_patient_embedding - patient_channel_embedding).squeeze(-1)
            graph_delta_logits = graph_delta_logits.masked_fill(~batch["channel_mask"], 0.0)
            base_logits = output["logits"]
            applied_mask = self._diffusion_center_mask(batch, base_logits) & batch["channel_mask"]
            gate_score = torch.sigmoid(self.diffusion_score_gate_raw)
            fused_logits = base_logits + gate_score * graph_delta_logits.masked_fill(~applied_mask, 0.0)
            output.update(self._scores_from_logits(fused_logits, batch["channel_mask"]))
            lzu_mask = (batch.get("center_id", torch.full((base_logits.shape[0],), 4, device=base_logits.device)).to(base_logits.device) == 1)[:, None]
            non_lzu_mask = (~lzu_mask) & batch["channel_mask"]
            branch_losses["diffusion_score_gate"] = gate_score
            branch_losses["diffusion_applied_patient_count"] = applied_mask.any(dim=1).float().sum()
            branch_losses["diffusion_residual_l2_loss"] = (
                graph_delta_logits[batch["channel_mask"]].square().mean() if torch.any(batch["channel_mask"]) else graph_delta_logits.sum() * 0.0
            )
            branch_losses["mean_abs_graph_delta_lzu"] = (
                graph_delta_logits[lzu_mask.expand_as(graph_delta_logits) & batch["channel_mask"]].abs().mean()
                if torch.any(lzu_mask.expand_as(graph_delta_logits) & batch["channel_mask"]) else graph_delta_logits.sum() * 0.0
            )
            branch_losses["mean_abs_graph_delta_non_lzu"] = (
                (fused_logits - base_logits)[non_lzu_mask].abs().mean() if torch.any(non_lzu_mask) else graph_delta_logits.sum() * 0.0
            )
            output["graph_delta_logits"] = graph_delta_logits
        # ---- A9v5 same-embedding two-head ----
        if self.use_two_expert_router:
            static_output = self.static_channel_classifier(patient_channel_embedding, batch["channel_mask"])
            anchor_logits_base = output["logits"]
            anchor_scores_base = self._scores_from_logits(anchor_logits_base, batch["channel_mask"])
            anchor_output_extra: dict[str, torch.Tensor] = {}
            if self.use_negative_anchor_head:
                anchor_output_extra = self.negative_anchor_head(
                    patient_channel_embedding, batch["labels_ez"], batch["channel_mask"],
                    anchor_logits_base, positive_label=self.positive_label, center_id=batch.get("center_id"),
                )
                anchor_logits_final = anchor_output_extra["logits"]
            else:
                anchor_logits_final = anchor_logits_base
            anchor_scores_final = self._scores_from_logits(anchor_logits_final, batch["channel_mask"])
            lambda_patient, center_id = self._two_expert_lambda(batch, anchor_logits_final)
            lambda_channel = lambda_patient[:, None]
            final_logits = (1.0 - lambda_channel) * static_output["logits"] + lambda_channel * anchor_logits_final
            final_scores = self._scores_from_logits(final_logits, batch["channel_mask"])
            output.update(anchor_output_extra)
            output.update(final_scores)
            output.update({
                "score_ez_static": static_output["score_ez"],
                "score_ez_anchor": anchor_scores_final["score_ez"],
                "score_ez_final": final_scores["score_ez"],
                "score_ez_base": anchor_scores_base["score_ez"],
                "two_expert_lambda_patient": lambda_patient,
                "two_expert_lambda_per_patient": lambda_patient,
                "two_expert_lambda_mean": lambda_patient.mean(),
            })
            output.update(self._two_expert_center_diagnostics(final_logits))
            ped_mask = center_id == 3
            output["two_expert_pediatric_preserve_loss"] = (
                F.mse_loss(final_scores["score_ez"][ped_mask], static_output["score_ez"][ped_mask])
                if torch.any(ped_mask) else final_logits.sum() * 0.0
            )
            output.update(branch_losses)
            output.update({
                "patient_channel_embedding": patient_channel_embedding,
                "task_embedding": patient_channel_embedding,
                "seizure_channel_embedding": seizure_channel_embedding,
                "temporal_attention": temporal_weights,
                "seizure_attention": seizure_weights,
                "n_channels_all_windows_invalid": n_invalid_windows,
            })
            return output
        if self.use_negative_anchor_head:
            anchor_output = self.negative_anchor_head(
                patient_channel_embedding, batch["labels_ez"], batch["channel_mask"],
                output["logits"], positive_label=self.positive_label, center_id=batch.get("center_id"),
            )
            output.update(anchor_output)
            output.update(self._scores_from_logits(anchor_output["logits"], batch["channel_mask"]))
            output["score_ez_base"] = anchor_output["score_ez_base"]
            output["score_ez_anchor"] = anchor_output["score_ez_anchor"]
            output["score_ez_final"] = output["score_ez"]
        if self.use_v3_qbc or self.use_v3_rcc:
            configured_logits = output["logits"]
            # BCR and V3-RCC use EZ-positive supervised logits.  BCR's formal
            # CDEL input is therefore exactly 1 - sigmoid(e_BCR).
            final_nez_logit = -configured_logits
            output.update(self._scores_from_logits(-final_nez_logit, batch["channel_mask"]))
            output["base_nez_logit"] = final_nez_logit
            output["final_nez_logit"] = final_nez_logit
            output["ez_semantic_logit"] = -final_nez_logit
            output["contextual_channel_embedding"] = patient_channel_embedding
            output["decision_rule"] = "score_only_decoder_applied_outside_forward"
        output.update({
            "patient_channel_embedding": patient_channel_embedding,
            "task_embedding": patient_channel_embedding,
            "seizure_channel_embedding": seizure_channel_embedding,
            "temporal_attention": temporal_weights,
            "seizure_attention": seizure_weights,
            "n_channels_all_windows_invalid": n_invalid_windows,
        })
        output.update(branch_losses)
        return output


__all__ = ["NeuroEZCModel"]
