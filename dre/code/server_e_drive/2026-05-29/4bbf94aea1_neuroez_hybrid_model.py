from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import torch
import torch.nn.functional as F
from torch import nn

try:
    from .graph_spectral_encoder import WindowGraphSpectralEncoder
    from .patient_channel_ranker import PatientChannelRanker
    from .seizure_aggregator import CrossSeizureMILAggregator
    from .temporal_encoder import ChannelTemporalEncoder
except ImportError:
    from graph_spectral_encoder import WindowGraphSpectralEncoder
    from patient_channel_ranker import PatientChannelRanker
    from seizure_aggregator import CrossSeizureMILAggregator
    from temporal_encoder import ChannelTemporalEncoder


@dataclass
class NeuroEZConfig:
    model_dim: int = 32
    num_heads: int = 2
    dropout: float = 0.40
    temporal_encoder: str = "mean"
    channel_layers: int = 1
    seizure_pooling: str = "attention"
    use_adjacency_message_passing: bool = True
    use_channel_attention: bool = True
    use_raw_cnn: bool = False
    raw_cnn_channels: int = 48
    raw_cnn_max_samples: int = 8192
    raw_cnn_weight: float = 0.25
    fusion: str = "gated"
    use_graph_edge_dropout: bool = False
    graph_edge_dropout: float = 0.0
    use_learnable_edge_gate: bool = False
    edge_gate_temperature: float = 1.0
    use_disentanglement: bool = False
    use_patient_adversarial: bool = False
    use_physics_prior: bool = False
    physics_logit_bias_beta: float = 0.15
    physics_fusion_weight: float = 0.25
    patient_projection_dim: int = 32
    task_projection_dim: int = 32
    patient_discriminator_hidden: int = 64
    num_patient_domains: int = 0

    @classmethod
    def from_args(cls, args: Any | None) -> "NeuroEZConfig":
        if args is None:
            return cls()
        return cls(
            model_dim=int(getattr(args, "model_dim", 32)),
            num_heads=int(getattr(args, "num_heads", 2)),
            dropout=float(getattr(args, "dropout", 0.40)),
            temporal_encoder=str(getattr(args, "temporal_encoder", "mean")),
            channel_layers=int(getattr(args, "channel_layers", 1)),
            seizure_pooling=str(getattr(args, "seizure_pooling", "attention")),
            use_adjacency_message_passing=bool(getattr(args, "use_adjacency_message_passing", True)),
            use_channel_attention=bool(getattr(args, "use_channel_attention", True)),
            use_raw_cnn=bool(getattr(args, "use_raw_cnn", False)),
            raw_cnn_channels=int(getattr(args, "raw_cnn_channels", 48)),
            raw_cnn_max_samples=int(getattr(args, "raw_cnn_max_samples", 8192)),
            raw_cnn_weight=float(getattr(args, "raw_cnn_weight", 0.25)),
            fusion=str(getattr(args, "fusion", "gated")),
            use_graph_edge_dropout=bool(getattr(args, "use_graph_edge_dropout", False)),
            graph_edge_dropout=float(getattr(args, "graph_edge_dropout", 0.0)),
            use_learnable_edge_gate=bool(getattr(args, "use_learnable_edge_gate", False)),
            edge_gate_temperature=float(getattr(args, "edge_gate_temperature", 1.0)),
            use_disentanglement=bool(getattr(args, "use_disentanglement", False)),
            use_patient_adversarial=bool(getattr(args, "use_patient_adversarial", False)),
            use_physics_prior=bool(getattr(args, "use_physics_prior", False)),
            physics_logit_bias_beta=float(getattr(args, "physics_logit_bias_beta", 0.15)),
            physics_fusion_weight=float(getattr(args, "physics_fusion_weight", 0.25)),
            patient_projection_dim=int(getattr(args, "patient_projection_dim", 32)),
            task_projection_dim=int(getattr(args, "task_projection_dim", 32)),
            patient_discriminator_hidden=int(getattr(args, "patient_discriminator_hidden", 64)),
            num_patient_domains=int(getattr(args, "num_patient_domains", 0)),
        )


class RawChannelCNNEncoder(nn.Module):
    """Per-channel waveform encoder for centered peri-onset iEEG clips."""

    def __init__(self, model_dim: int = 96, base_channels: int = 48, dropout: float = 0.25, max_samples: int = 8192) -> None:
        super().__init__()
        c = int(base_channels)
        self.max_samples = int(max_samples)
        self.net = nn.Sequential(
            nn.Conv1d(1, c, kernel_size=15, stride=4, padding=7, bias=False),
            nn.GroupNorm(1, c),
            nn.GELU(),
            nn.Conv1d(c, c, kernel_size=9, stride=4, padding=4, groups=max(1, c // 8), bias=False),
            nn.GroupNorm(1, c),
            nn.GELU(),
            nn.Conv1d(c, c * 2, kernel_size=7, stride=2, padding=3, bias=False),
            nn.GroupNorm(1, c * 2),
            nn.GELU(),
            nn.Conv1d(c * 2, c * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.GroupNorm(1, c * 2),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(c * 2, int(model_dim)),
            nn.LayerNorm(int(model_dim)),
        )

    def forward(self, raw_x: torch.Tensor, seizure_channel_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, s, c, l = raw_x.shape
        x = raw_x.reshape(b * s * c, 1, l)
        if self.max_samples > 0 and l > self.max_samples:
            x = F.interpolate(x, size=self.max_samples, mode="linear", align_corners=False)
        h = self.proj(self.net(x)).reshape(b, s, c, -1)
        if seizure_channel_mask is not None:
            h = h * seizure_channel_mask.float().unsqueeze(-1)
        return h


class GatedModalityFusion(nn.Module):
    def __init__(self, model_dim: int, dropout: float = 0.25) -> None:
        super().__init__()
        dim = int(model_dim)
        self.graph_proj = nn.Linear(dim, dim)
        self.raw_proj = nn.Linear(dim, dim)
        self.gate = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(dim, dim),
            nn.Sigmoid(),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, graph_h: torch.Tensor, raw_h: torch.Tensor) -> torch.Tensor:
        graph_h = self.graph_proj(graph_h)
        raw_h = self.raw_proj(raw_h)
        gate = self.gate(torch.cat([graph_h, raw_h], dim=-1))
        return self.norm(gate * graph_h + (1.0 - gate) * raw_h)


class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_value: float) -> torch.Tensor:
        ctx.lambda_value = float(lambda_value)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_value * grad_output, None


def gradient_reverse(x: torch.Tensor, lambda_value: float = 1.0) -> torch.Tensor:
    return GradientReversalFunction.apply(x, float(lambda_value))


class PatientDiscriminator(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_patient_domains: int) -> None:
        super().__init__()
        self.num_patient_domains = int(num_patient_domains)
        if self.num_patient_domains > 1:
            self.net = nn.Sequential(
                nn.Linear(int(input_dim), int(hidden_dim)),
                nn.GELU(),
                nn.Dropout(0.10),
                nn.Linear(int(hidden_dim), self.num_patient_domains),
            )
        else:
            self.net = None

    def forward(self, x: torch.Tensor) -> torch.Tensor | None:
        if self.net is None:
            return None
        return self.net(x)


def _masked_channel_mean(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    mask = channel_mask.float().unsqueeze(-1)
    return (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def _masked_scalar_zscore(x: torch.Tensor, channel_mask: torch.Tensor) -> torch.Tensor:
    mask = channel_mask.float()
    count = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = (x * mask).sum(dim=1, keepdim=True) / count
    var = (((x - mean) * mask) ** 2).sum(dim=1, keepdim=True) / count
    z = (x - mean) / torch.sqrt(var + 1e-5)
    return z * mask


class NeuroEZHybridModel(nn.Module):
    """Patient-level neural pipeline for inverse EZ channel ranking.

    Inputs use the patient-level batch produced by ``collate_patient_ez_batch``.
    V3-small output ``scores`` are p_NEZ. EZ ranking is ascending p_NEZ, or
    equivalently descending ``score_ez = 1 - p_NEZ``.
    """

    def __init__(self, args: Any | None = None) -> None:
        super().__init__()
        cfg = NeuroEZConfig.from_args(args)
        self.cfg = cfg
        self.window_encoder = WindowGraphSpectralEncoder(
            model_dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            use_message_passing=cfg.use_adjacency_message_passing,
            use_channel_attention=cfg.use_channel_attention,
            use_graph_edge_dropout=cfg.use_graph_edge_dropout,
            graph_edge_dropout=cfg.graph_edge_dropout,
            use_learnable_edge_gate=cfg.use_learnable_edge_gate,
            edge_gate_temperature=cfg.edge_gate_temperature,
        )
        self.temporal_encoder = ChannelTemporalEncoder(
            model_dim=cfg.model_dim,
            encoder_type=cfg.temporal_encoder,
            num_layers=cfg.channel_layers,
            dropout=cfg.dropout,
        )
        self.raw_encoder = RawChannelCNNEncoder(
            model_dim=cfg.model_dim,
            base_channels=cfg.raw_cnn_channels,
            dropout=cfg.dropout,
            max_samples=cfg.raw_cnn_max_samples,
        ) if cfg.use_raw_cnn else None
        self.fusion = GatedModalityFusion(cfg.model_dim, dropout=cfg.dropout) if cfg.use_raw_cnn else nn.Identity()
        self.seizure_aggregator = CrossSeizureMILAggregator(
            model_dim=cfg.model_dim,
            dropout=cfg.dropout,
            pooling=cfg.seizure_pooling,
        )
        self.use_disentanglement = bool(cfg.use_disentanglement)
        self.use_patient_adversarial = bool(cfg.use_patient_adversarial)
        ranker_dim = int(cfg.task_projection_dim) if self.use_disentanglement else int(cfg.model_dim)
        self.task_projection = nn.Sequential(
            nn.Linear(cfg.model_dim, ranker_dim),
            nn.GELU(),
            nn.LayerNorm(ranker_dim),
        ) if self.use_disentanglement else nn.Identity()
        self.patient_projection = nn.Sequential(
            nn.Linear(cfg.model_dim, int(cfg.patient_projection_dim)),
            nn.GELU(),
            nn.LayerNorm(int(cfg.patient_projection_dim)),
        ) if self.use_disentanglement else None
        self.orthogonal_patient_projection = nn.Linear(
            int(cfg.patient_projection_dim),
            ranker_dim,
        ) if self.use_disentanglement and int(cfg.patient_projection_dim) != ranker_dim else nn.Identity()
        self.task_patient_discriminator = PatientDiscriminator(
            input_dim=ranker_dim,
            hidden_dim=cfg.patient_discriminator_hidden,
            num_patient_domains=cfg.num_patient_domains,
        ) if self.use_patient_adversarial else None
        self.patient_specific_discriminator = PatientDiscriminator(
            input_dim=int(cfg.patient_projection_dim),
            hidden_dim=cfg.patient_discriminator_hidden,
            num_patient_domains=cfg.num_patient_domains,
        ) if self.use_disentanglement else None
        self.patient_ranker = PatientChannelRanker(
            model_dim=ranker_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
        self.physics_encoder = nn.Sequential(
            nn.Linear(1, ranker_dim),
            nn.GELU(),
            nn.Dropout(float(cfg.dropout)),
            nn.Linear(ranker_dim, ranker_dim),
            nn.LayerNorm(ranker_dim),
        ) if cfg.use_physics_prior else None

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = batch["features"]
        adjacency = batch.get("adjacency")
        if not self.cfg.use_adjacency_message_passing:
            adjacency = None
        seizure_channel_mask = batch["seizure_channel_mask"]
        seizure_mask = batch["seizure_mask"]
        channel_mask = batch["channel_mask"]
        window_mask = batch.get("window_mask")

        window_embeddings = self.window_encoder(features, adjacency, seizure_channel_mask)
        seizure_channel_embedding, temporal_attention = self.temporal_encoder(
            window_embeddings,
            seizure_channel_mask,
            window_mask=window_mask,
        )
        if self.raw_encoder is not None:
            raw_embedding = self.raw_encoder(batch["raw_x"], seizure_channel_mask)
            fused_embedding = self.fusion(seizure_channel_embedding, raw_embedding)
            raw_weight = min(max(float(self.cfg.raw_cnn_weight), 0.0), 1.0)
            seizure_channel_embedding = (1.0 - raw_weight) * seizure_channel_embedding + raw_weight * fused_embedding
        else:
            raw_embedding = torch.zeros_like(seizure_channel_embedding)

        patient_channel_embedding, seizure_attention = self.seizure_aggregator(
            seizure_channel_embedding,
            seizure_mask,
            seizure_channel_mask,
        )
        task_embedding = self.task_projection(patient_channel_embedding)
        physics_risk = batch.get("physics_risk")
        physics_z = None
        if self.physics_encoder is not None and physics_risk is not None:
            physics_risk = physics_risk.to(task_embedding.device).clamp(0.0, 1.0) * channel_mask.float()
            physics_z = _masked_scalar_zscore(physics_risk, channel_mask)
            physics_embedding = self.physics_encoder(physics_risk.unsqueeze(-1))
            task_embedding = task_embedding + float(self.cfg.physics_fusion_weight) * physics_embedding
        patient_specific_embedding = self.patient_projection(patient_channel_embedding) if self.patient_projection is not None else None
        patient_specific_for_orthogonality = (
            self.orthogonal_patient_projection(patient_specific_embedding)
            if patient_specific_embedding is not None
            else None
        )
        ranker_output = self.patient_ranker(task_embedding, channel_mask)
        if self.physics_encoder is not None and physics_risk is not None and physics_z is not None:
            base_logits = ranker_output["logits"]
            logits = base_logits - float(self.cfg.physics_logit_bias_beta) * physics_z
            logits = logits.masked_fill(~channel_mask, -1e9)
            scores = torch.sigmoid(logits)
            ranker_output.update(
                {
                    "base_logits": base_logits,
                    "logits": logits,
                    "scores": scores,
                    "score_nez": scores,
                    "score_ez": 1.0 - scores,
                    "score_mass": torch.sigmoid(logits.masked_fill(~channel_mask, -20.0)).sum(dim=1),
                    "physics_risk": physics_risk,
                    "physics_risk_z": physics_z,
                }
            )
        task_patient_logits = None
        patient_specific_logits = None
        if self.task_patient_discriminator is not None:
            pooled_task = _masked_channel_mean(task_embedding, channel_mask)
            task_patient_logits = self.task_patient_discriminator(gradient_reverse(pooled_task, 1.0))
        if self.patient_specific_discriminator is not None and patient_specific_embedding is not None:
            pooled_patient = _masked_channel_mean(patient_specific_embedding, channel_mask)
            patient_specific_logits = self.patient_specific_discriminator(pooled_patient)

        graph_sparsity_loss = self.window_encoder.graph_sparsity_loss
        if graph_sparsity_loss is None:
            graph_sparsity_loss = patient_channel_embedding.sum() * 0.0
        ranker_output.update(
            {
                "patient_channel_embedding": patient_channel_embedding,
                "task_embedding": task_embedding,
                "patient_specific_embedding": patient_specific_embedding,
                "patient_specific_for_orthogonality": patient_specific_for_orthogonality,
                "task_patient_logits": task_patient_logits,
                "patient_specific_logits": patient_specific_logits,
                "graph_sparsity_loss": graph_sparsity_loss,
                "seizure_channel_embedding": seizure_channel_embedding,
                "raw_embedding": raw_embedding,
                "temporal_attention": temporal_attention,
                "seizure_attention": seizure_attention,
            }
        )
        return ranker_output


__all__ = [
    "GatedModalityFusion",
    "NeuroEZConfig",
    "NeuroEZHybridModel",
    "RawChannelCNNEncoder",
    "gradient_reverse",
]
