from __future__ import annotations

import math
from typing import Any

import torch
from torch import nn

from outcome_hifos.leakage_guard import assert_label_blind_tree

from .channel_encoder import ChannelWindowEncoder
from .core_assignment import CompetitiveCoreAssigner
from .core_bank import PatientConditionedCanonicalCoreBank
from .focality import WindowFocalityDescriptor
from .hierarchical_pool import HierarchicalAttentionMIL, HierarchicalStatisticalPool
from .patient_context import PatientContextEncoder, _masked_token_statistics
from .patient_head import PatientOutcomeHead
from .recurrence import CrossSeizureCoreRecurrence
from .seizure_encoder import SeizureTrajectorySummary
from .transport_encoder import TemporalCoreTransportEncoder, adjacent_core_transport
from .unbalanced_ot import UnbalancedSinkhornTransport


class _AttentionPatientPool(nn.Module):
    def __init__(self, model_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.zeros(int(model_dim)))
        nn.init.normal_(self.query, std=0.02)
        self.projection = nn.Sequential(nn.Linear(int(model_dim) * 5, int(model_dim) * 4), nn.GELU(), nn.LayerNorm(int(model_dim) * 4))

    def forward(self, values: torch.Tensor, valid: torch.Tensor, *, use_attention: bool) -> torch.Tensor:
        statistics = _masked_token_statistics(values, valid)
        batch, _, _, _, dim = values.shape
        flat = values.reshape(batch, -1, dim)
        mask = valid.reshape(batch, -1)
        if use_attention:
            scores = torch.einsum("bnd,d->bn", flat, self.query) / math.sqrt(max(dim, 1))
            scores = scores.masked_fill(~mask, float("-inf"))
            weights = torch.softmax(scores, dim=1) * mask.to(values.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-8)
            pooled = torch.einsum("bn,bnd->bd", weights, flat)
        else:
            pooled = statistics[:, :dim]
        return self.projection(torch.cat([statistics, pooled], dim=-1))


class HiFOSPACT(nn.Module):
    def __init__(self, *, variant: str, input_dim: int, config: dict[str, Any]) -> None:
        super().__init__()
        self.variant = str(variant)
        self.model_dim = int(config.get("model_dim", 64))
        self.num_cores = int(config.get("num_cores", 4))
        dropout = float(config.get("dropout", 0.1))
        self.channel_encoder = ChannelWindowEncoder(input_dim, self.model_dim, dropout)
        self.patient_pool = _AttentionPatientPool(self.model_dim)
        if self.variant == "H2_HIER_POOL":
            self.hierarchical_baseline = HierarchicalStatisticalPool(self.model_dim)
        elif self.variant == "H3_ATTENTION_MIL":
            self.hierarchical_baseline = HierarchicalAttentionMIL(self.model_dim)
        self.patient_context = PatientContextEncoder(self.model_dim)
        self.uses_cores = self.variant in {"H4_MULTI_CORE", "H5_ANCHORED_CORE", "H6_ANCHORED_UOT_DESC", "H7_TRANSPORT_GRAPH", "H8_RECURRENCE", "H9_FM_RECURRENCE"}
        self.uses_transport = self.variant in {"H6_ANCHORED_UOT_DESC", "H7_TRANSPORT_GRAPH", "H8_RECURRENCE", "H9_FM_RECURRENCE"}
        self.uses_recurrence = self.variant in {"H8_RECURRENCE", "H9_FM_RECURRENCE"}
        if self.uses_cores:
            self.core_bank = PatientConditionedCanonicalCoreBank(
                self.num_cores,
                self.model_dim,
                conditioned=self.variant != "H4_MULTI_CORE",
            )
            self.core_assigner = CompetitiveCoreAssigner(
                self.model_dim,
                self.num_cores,
                temperature=float(config.get("core_temperature", 0.2)),
            )
            self.focality = WindowFocalityDescriptor()
            core_input_dim = self.model_dim + self.num_cores * 2 + WindowFocalityDescriptor.output_dim * 2
            self.core_projection = nn.Sequential(
                nn.Linear(core_input_dim, self.model_dim * 4),
                nn.GELU(),
                nn.LayerNorm(self.model_dim * 4),
            )
            if self.uses_transport:
                self.uot_solver = UnbalancedSinkhornTransport(
                    epsilon=float(config.get("uot_epsilon", 0.1)),
                    tau=float(config.get("uot_tau", 1.0)),
                    iterations=int(config.get("sinkhorn_iterations", 20)),
                    tolerance=float(config.get("sinkhorn_tolerance", 1e-5)),
                )
                self.transport_encoder = TemporalCoreTransportEncoder(
                    self.model_dim,
                    self.num_cores,
                    WindowFocalityDescriptor.output_dim,
                    graph_mode=self.variant in {"H7_TRANSPORT_GRAPH", "H8_RECURRENCE", "H9_FM_RECURRENCE"},
                )
                self.seizure_summary = SeizureTrajectorySummary(self.model_dim)
                if self.uses_recurrence:
                    self.recurrence = CrossSeizureCoreRecurrence(self.model_dim, self.uot_solver)
        self.patient_head = PatientOutcomeHead(self.model_dim * 4, dropout=float(config.get("head_dropout", dropout)))

    @staticmethod
    def _masked_mean_std(values: torch.Tensor, mask: torch.Tensor, dimensions: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
        weight = mask.to(values.dtype)
        while weight.ndim < values.ndim:
            weight = weight.unsqueeze(-1)
        count = weight.sum(dim=dimensions).clamp_min(1.0)
        mean = (values * weight).sum(dim=dimensions) / count
        expanded_mean = mean
        for dimension in sorted(dimensions):
            expanded_mean = expanded_mean.unsqueeze(dimension)
        variance = ((values - expanded_mean).square() * weight).sum(dim=dimensions) / count
        return mean, torch.sqrt(variance.clamp_min(1e-8))

    def forward(self, batch: dict[str, torch.Tensor], *, persist_diagnostics: bool = True) -> dict[str, torch.Tensor]:
        assert_label_blind_tree(batch, stage="model_forward")
        valid = batch["window_channel_mask"].bool()
        encoded = self.channel_encoder(batch["feature_x"], valid)
        if not self.uses_cores:
            hierarchy = self.hierarchical_baseline(
                encoded,
                valid,
                batch["window_mask"].bool(),
                batch["seizure_mask"].bool(),
                batch["window_centers"],
            )
            return {"logits": self.patient_head(hierarchy["patient_embedding"]), **hierarchy}

        context = self.patient_context(encoded, valid)
        anchors = self.core_bank(context)
        assignment = self.core_assigner(encoded, anchors, valid)
        focality = self.focality(assignment.core_masses, assignment.background_mass, assignment.responsibilities, anchors, valid)
        window_valid = valid.any(dim=-1)
        if self.uses_transport:
            transport = adjacent_core_transport(
                assignment.core_states,
                assignment.core_masses,
                assignment.responsibilities,
                window_valid,
                self.uot_solver,
            )
            seizure_embeddings, transported_states = self.transport_encoder(
                assignment.core_states,
                assignment.core_masses,
                focality,
                transport,
                window_valid,
            )
            window_weight = window_valid[..., None, None].to(encoded.dtype)
            seizure_core_states = (transported_states * window_weight).sum(dim=2) / window_weight.sum(dim=2).clamp_min(1.0)
            mass_weight = window_valid.unsqueeze(-1).to(encoded.dtype)
            seizure_core_masses = (assignment.core_masses * mass_weight).sum(dim=2) / mass_weight.sum(dim=2).clamp_min(1.0)
            if self.uses_recurrence:
                patient_embedding, recurrence_distances, seizure_attention = self.recurrence(
                    seizure_embeddings,
                    seizure_core_states,
                    seizure_core_masses,
                    batch["seizure_mask"].bool(),
                )
            else:
                patient_embedding, seizure_attention = self.seizure_summary(seizure_embeddings, batch["seizure_mask"].bool())
                recurrence_distances = encoded.new_zeros((encoded.shape[0], encoded.shape[1], encoded.shape[1]))
            output = {
                "logits": self.patient_head(patient_embedding),
                "patient_embedding": patient_embedding,
                "seizure_embeddings": seizure_embeddings,
                "core_anchors": anchors,
                "core_states": transported_states,
                "core_masses": assignment.core_masses,
                "background_mass": assignment.background_mass,
                "responsibilities": assignment.responsibilities,
                "focality_descriptors": focality,
                "transport_plans": transport.plans,
                "transport_costs": transport.costs,
                "transport_descriptors": transport.descriptors,
                "recurrence_distances": recurrence_distances,
                "seizure_attention": seizure_attention,
            }
            if not persist_diagnostics:
                output = {key: value for key, value in output.items() if key in {"logits", "patient_embedding", "core_masses", "seizure_embeddings"}}
            return output
        mass_weight = assignment.core_masses.unsqueeze(-1)
        state_numerator = (assignment.core_states * mass_weight).sum(dim=(1, 2, 3))
        state_denominator = mass_weight.sum(dim=(1, 2, 3)).clamp_min(1e-8)
        state_summary = state_numerator / state_denominator
        mass_mean, mass_std = self._masked_mean_std(assignment.core_masses, window_valid, (1, 2))
        focal_mean, focal_std = self._masked_mean_std(focality, window_valid, (1, 2))
        patient_embedding = self.core_projection(torch.cat([state_summary, mass_mean, mass_std, focal_mean, focal_std], dim=-1))
        output = {
            "logits": self.patient_head(patient_embedding),
            "patient_embedding": patient_embedding,
            "core_anchors": anchors,
            "core_states": assignment.core_states,
            "core_masses": assignment.core_masses,
            "background_mass": assignment.background_mass,
            "responsibilities": assignment.responsibilities,
            "focality_descriptors": focality,
        }
        if not persist_diagnostics:
            output = {key: value for key, value in output.items() if key in {"logits", "patient_embedding", "core_masses"}}
        return output


__all__ = ["HiFOSPACT"]
