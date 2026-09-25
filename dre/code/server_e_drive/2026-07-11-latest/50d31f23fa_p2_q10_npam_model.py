from __future__ import annotations

from typing import Any

import torch
from torch import nn

from .bounded_graph_residual import BoundedPhaseGraphResidual
from .channel_pam import ChannelOutcomePAM
from .graph_stability import compute_graph_stability
from .masks import masked_mean, masked_std
from .network_burden import compute_network_burden
from .phase_network import PhaseNetworkEncoder
from .profiles import get_profile
from .simple_q10 import compute_simple_q10


class P2Q10NPAMModel(nn.Module):
    """Task-2 head consuming one P2 backbone export and optional raw graphs."""

    FORBIDDEN_INPUT_TOKENS = ("coordinate", "mni", "fsaverage", "soz", "resect", "true_k", "true_ez")

    def __init__(self, embedding_dim: int, profile: str = "M0_PAM") -> None:
        super().__init__()
        self.embedding_dim = int(embedding_dim)
        self.profile = get_profile(profile)
        self.channel_pam = ChannelOutcomePAM(self.embedding_dim)
        self.network_encoder = PhaseNetworkEncoder(include_deltas=self.profile.phase_contrast) if self.profile.network_stats else None
        self.graph_residual = BoundedPhaseGraphResidual(self.embedding_dim) if self.profile.graph_residual else None
        seizure_input_dim = 4 * self.embedding_dim + 2
        if self.profile.network_stats:
            seizure_input_dim += 32
        if self.profile.graph_residual:
            seizure_input_dim += 64
        self.seizure_projection = nn.Sequential(nn.LayerNorm(seizure_input_dim), nn.Linear(seizure_input_dim, 128), nn.GELU(), nn.Dropout(0.20))
        self.seizure_outcome_head = nn.Linear(128, 1)
        self.stability_projection = nn.Sequential(nn.Linear(7, 16), nn.GELU()) if self.profile.graph_stability else None
        patient_dim = 128 * 2 + 1 + (16 if self.profile.graph_stability else 0)
        self.outcome_head = nn.Sequential(
            nn.LayerNorm(patient_dim), nn.Linear(patient_dim, 64), nn.GELU(), nn.Dropout(0.30),
            nn.Linear(64, 16), nn.GELU(), nn.Dropout(0.10), nn.Linear(16, 1),
        )

    @classmethod
    def assert_label_blind_inputs(cls, mapping: dict[str, Any]) -> None:
        forbidden = [key for key in mapping if any(token in str(key).lower() for token in cls.FORBIDDEN_INPUT_TOKENS)]
        if forbidden:
            raise ValueError(f"Forbidden Task-2 model inputs: {sorted(forbidden)}")
        if any(str(key).lower() in {"center", "center_id", "outcome", "outcome_label"} for key in mapping):
            raise ValueError("center and outcome fields are audit/target metadata, not model inputs")

    def forward(self, p2: dict[str, torch.Tensor], graphs: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
        self.assert_label_blind_inputs(p2)
        if graphs is not None:
            self.assert_label_blind_inputs(graphs)
        required = {
            "seizure_channel_embedding", "seizure_nez_probability", "phase_channel_embedding",
            "seizure_mask", "seizure_channel_mask", "channel_mask", "phase_mask",
        }
        missing = sorted(required - set(p2))
        if missing:
            raise KeyError(f"P2 export missing required tensors: {missing}")
        seizure_mask = p2["seizure_mask"].bool()
        seizure_channel_mask = p2["seizure_channel_mask"].bool() & seizure_mask.unsqueeze(-1)
        channel_mask = p2["channel_mask"].bool()
        q10 = compute_simple_q10(p2["seizure_nez_probability"], seizure_mask, seizure_channel_mask, channel_mask)
        pam = self.channel_pam(p2["seizure_channel_embedding"], q10["simple_q10_nez_z"], seizure_channel_mask)
        pieces = [pam["global_embedding"], pam["channel_std_embedding"], pam["risk_embedding"], pam["risk_global_contrast"], pam["node_failure_burden"].unsqueeze(-1), pam["additive_failure_evidence"].unsqueeze(-1)]
        network: dict[str, torch.Tensor] = {}
        graph_output: dict[str, torch.Tensor] = {}
        adjacency = None
        # P2 may export four temporal embeddings; raw functional graphs are fixed to
        # preictal/onset/spread and therefore consume only the first three.
        graph_phase_embedding = p2["phase_channel_embedding"][:, :, :3]
        phase_channel_mask = p2["phase_mask"].bool()[:, :, :3] & seizure_channel_mask.unsqueeze(2)
        phase_valid = phase_channel_mask.sum(dim=-1) >= 4
        if self.profile.network_stats or self.profile.graph_residual or self.profile.graph_stability:
            if graphs is None or "adjacency" not in graphs:
                raise KeyError(f"Profile {self.profile.name} requires label-free raw functional graphs")
            adjacency = graphs["adjacency"]
            graph_phase_valid = graphs.get("graph_valid", phase_valid).bool() & phase_valid
            phase_channel_mask &= graphs.get("phase_channel_mask", phase_channel_mask).bool()
            network = compute_network_burden(adjacency, pam["risk_membership"], seizure_channel_mask, phase_channel_mask)
            network["graph_valid"] &= graph_phase_valid
            network["network_burden"] = network["network_burden"].masked_fill(~network["graph_valid"].unsqueeze(-1), 0.0)
        if self.network_encoder is not None:
            phase_network = self.network_encoder(network["network_burden"], network["graph_valid"])
            network.update(phase_network)
            pieces.append(network["network_stat_embedding"])
        if self.graph_residual is not None:
            graph_output = self.graph_residual(graph_phase_embedding, adjacency, phase_channel_mask, network["graph_valid"])
            pieces.append(graph_output["graph_embedding"])
        seizure_input = torch.cat(pieces, dim=-1)
        seizure_embedding = self.seizure_projection(seizure_input).masked_fill(~seizure_mask.unsqueeze(-1), 0.0)
        seizure_logit_success = self.seizure_outcome_head(seizure_embedding).squeeze(-1).masked_fill(~seizure_mask, 0.0)
        patient_mean = masked_mean(seizure_embedding, seizure_mask, dim=1)
        patient_std = masked_std(seizure_embedding, seizure_mask, dim=1)
        n_seizures = seizure_mask.sum(dim=1).to(seizure_embedding.dtype)
        patient_pieces = [patient_mean, patient_std, torch.log1p(n_seizures).unsqueeze(-1)]
        stability: dict[str, torch.Tensor] = {}
        if self.stability_projection is not None:
            stability = compute_graph_stability(pam["risk_membership"], adjacency, seizure_mask, seizure_channel_mask, network["graph_valid"])
            stability["graph_stability_embedding"] = self.stability_projection(stability["graph_stability_vector"])
            patient_pieces.append(stability["graph_stability_embedding"])
        patient_embedding = torch.cat(patient_pieces, dim=-1)
        outcome_logit_success = self.outcome_head(patient_embedding).squeeze(-1)
        outcome_probability_success = torch.sigmoid(outcome_logit_success)
        result = {
            "outcome_logit_success": outcome_logit_success,
            "outcome_probability_success": outcome_probability_success,
            "outcome_probability_failure": 1.0 - outcome_probability_success,
            "seizure_outcome_embedding": seizure_embedding,
            "seizure_outcome_logit_success": seizure_logit_success,
            "seizure_channel_valid": seizure_channel_mask,
            "patient_outcome_embedding": patient_embedding,
            "q10_residual": pam["q10_adjustment"],
            **q10,
            **pam,
            **network,
            **graph_output,
            **stability,
        }
        return result


__all__ = ["P2Q10NPAMModel"]
