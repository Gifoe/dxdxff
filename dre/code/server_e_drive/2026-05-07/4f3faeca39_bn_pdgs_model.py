from __future__ import annotations

from typing import Any

import torch
from torch import nn

from ..config import BNPDGSConfig
from .graph_spectral_encoder import WindowGraphSpectralEncoder
from .patient_channel_ranker import PatientChannelRanker
from .seizure_aggregator import CrossSeizureMILAggregator
from .temporal_encoder import ChannelTemporalEncoder


class BNPDGSModel(nn.Module):
    def __init__(self, args: Any | None = None) -> None:
        super().__init__()
        cfg = BNPDGSConfig.from_args(args)
        self.cfg = cfg
        self.window_encoder = WindowGraphSpectralEncoder(
            model_dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
            use_message_passing=cfg.use_adjacency_message_passing,
        )
        self.temporal_encoder = ChannelTemporalEncoder(
            model_dim=cfg.model_dim,
            encoder_type=cfg.temporal_encoder,
            num_layers=cfg.channel_layers,
            dropout=cfg.dropout,
        )
        self.seizure_aggregator = CrossSeizureMILAggregator(
            model_dim=cfg.model_dim,
            dropout=cfg.dropout,
            pooling=cfg.seizure_pooling,
        )
        self.patient_ranker = PatientChannelRanker(
            model_dim=cfg.model_dim,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )

    def forward(self, batch: dict[str, Any]) -> dict[str, torch.Tensor]:
        features = batch["features"]
        adjacency = batch.get("adjacency")
        if not self.cfg.use_adjacency_message_passing:
            adjacency = None
        seizure_channel_mask = batch["seizure_channel_mask"]
        seizure_mask = batch["seizure_mask"]
        channel_mask = batch["channel_mask"]
        seizure_quality = batch.get("seizure_quality") if self.cfg.use_seizure_quality_weighting else None
        topology_features = batch.get("topology_features") if self.cfg.use_topology_features else None

        window_embeddings = self.window_encoder(features, adjacency, seizure_channel_mask)
        seizure_channel_embedding, temporal_attention = self.temporal_encoder(window_embeddings, seizure_channel_mask)
        patient_channel_embedding, seizure_attention = self.seizure_aggregator(
            seizure_channel_embedding,
            seizure_mask,
            seizure_channel_mask,
            seizure_quality,
        )
        ranker_output = self.patient_ranker(patient_channel_embedding, channel_mask, topology_features)
        ranker_output.update(
            {
                "patient_channel_embedding": patient_channel_embedding,
                "temporal_attention": temporal_attention,
                "seizure_attention": seizure_attention,
            }
        )
        return ranker_output


__all__ = ["BNPDGSModel"]
