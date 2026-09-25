from __future__ import annotations

import math
from typing import Any

import torch

from .abnormality_dispersion import DISPERSION_FEATURES, compute_abnormality_dispersion
from .clean_nez_prototype import CleanNEZPrototype
from .functional_graph import NETWORK_PHASES
from .nez_consensus import compute_nez_consensus
from .outside_nez_residual import OUTSIDE_RESIDUAL_FEATURES, compute_outside_nez_residual
from .seizure_nez_prototype import SEIZURE_STABILITY_FEATURES, compute_seizure_nez_stability, seizure_prototype_probability
from .virtual_target_network import VIRTUAL_NETWORK_FEATURES, compute_virtual_target_network


BASE_VIEW_FEATURES = (
    "outside_final_nez_mean", "outside_final_nez_q10", "outside_final_nez_min",
    "outside_direct_nez_mean", "outside_direct_nez_q10", "outside_direct_nez_min",
    "inside_final_nez_mean", "inside_direct_nez_mean", "inside_outside_final_gap",
)
CONSENSUS_SUMMARY_FEATURES = ("view_disagreement_mean", "reliability_weight_mean")
ALL_SCALAR_FEATURES = BASE_VIEW_FEATURES + OUTSIDE_RESIDUAL_FEATURES + CONSENSUS_SUMMARY_FEATURES + SEIZURE_STABILITY_FEATURES + DISPERSION_FEATURES + VIRTUAL_NETWORK_FEATURES


def scalar_features_for_profile(profile: str) -> tuple[str, ...]:
    key = str(profile).upper()
    # Primary heads use a compact predeclared subset. Highly correlated
    # quantiles/complements remain available in audit outputs only.
    if key == "R0_NEZ_OUTSIDE": return ("outside_final_nez_mean", "outside_final_nez_q10", "outside_direct_nez_mean", "outside_direct_nez_q10", "inside_outside_final_gap")
    r1 = ("outside_nez_q10", "outside_residual_top10_mean", "outside_suspicious_fraction_05", "inside_abnormality_mean", "inside_outside_reliable_gap", "view_disagreement_mean", "reliability_weight_mean")
    if key == "R1_NEZ_CONSENSUS": return r1
    r2 = r1 + ("outside_persistent_top10", "outside_worstcase_top10", "outside_abnormal_seizure_fraction", "outside_temporal_std")
    if key == "R2_PERSISTENT_RESIDUAL": return r2
    if key == "R3_VIRTUAL_RESECTION": return r2 + ("residual_edge_mass_spread", "persistent_residual_edge_mass_spread", "residual_spectral_radius_ratio_spread", "residual_largest_component_fraction_spread", "hub_miss_ratio_spread", "virtual_disruption_spread", "global_abnormality_entropy_normalized", "outside_abnormal_cluster_count", "outside_largest_cluster_fraction")
    from .monotone_outcome_head import MONOTONE_FEATURES
    if key.startswith(("R4_", "R5_", "R6_")): return MONOTONE_FEATURES
    raise ValueError(f"Unsupported NVR profile: {profile}")


def _q10(value: torch.Tensor) -> torch.Tensor:
    return torch.quantile(value, .10)


def _view_features(q_final: torch.Tensor, q_direct: torch.Tensor, target: torch.Tensor, valid: torch.Tensor) -> dict[str, torch.Tensor]:
    rows = {name: [] for name in BASE_VIEW_FEATURES}
    for p in range(q_final.shape[0]):
        outside, inside = valid[p] & ~target[p], valid[p] & target[p]
        if not outside.any() or not inside.any(): raise ValueError("NVR requires non-empty target and outside channel sets")
        fo, fi, do, di = q_final[p][outside], q_final[p][inside], q_direct[p][outside], q_direct[p][inside]
        values = {"outside_final_nez_mean": fo.mean(), "outside_final_nez_q10": _q10(fo), "outside_final_nez_min": fo.min(),
                  "outside_direct_nez_mean": do.mean(), "outside_direct_nez_q10": _q10(do), "outside_direct_nez_min": do.min(),
                  "inside_final_nez_mean": fi.mean(), "inside_direct_nez_mean": di.mean(), "inside_outside_final_gap": fi.mean()-fo.mean()}
        for name in rows: rows[name].append(values[name])
    return {name: torch.stack(values) for name, values in rows.items()}


def _phase_strength(adjacency: torch.Tensor, phase_mask: torch.Tensor, graph_valid: torch.Tensor) -> torch.Tensor:
    b, s, phases, channels, _ = adjacency.shape
    output = adjacency.new_zeros((b, phases, channels))
    counts = adjacency.new_zeros((b, phases, channels))
    for p in range(b):
        for seizure in range(s):
            for phase in range(phases):
                if not bool(graph_valid[p, seizure, phase]): continue
                mask = phase_mask[p, seizure, phase].bool()
                output[p, phase] += adjacency[p, seizure, phase].sum(-1) * mask
                counts[p, phase] += mask
    return output / counts.clamp_min(1)


def build_nvr_features(
    evidence: dict[str, torch.Tensor], patient_prototype: CleanNEZPrototype,
    seizure_prototype: CleanNEZPrototype, *, target_override: torch.Tensor | None = None,
    channel_mask_override: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Create NVR evidence without labels, center, coordinates, SOZ, or resection inputs."""
    patient_embedding = evidence["patient_channel_embedding"].float()
    valid = (channel_mask_override if channel_mask_override is not None else evidence["channel_mask"]).bool()
    target = (target_override if target_override is not None else evidence["clinical_target_mask"]).bool() & valid
    q_final = torch.sigmoid(evidence["final_nez_logit"].float())
    q_direct = torch.sigmoid(evidence["direct_nez_logit"].float())
    q_proto = patient_prototype.probability_nez(patient_embedding.detach().cpu()).to(patient_embedding.device)
    consensus = compute_nez_consensus(q_final, q_direct, q_proto)
    residual = compute_outside_nez_residual(consensus["q_consensus_nez"], consensus["reliable_abnormality"], target, valid)

    seizure_embedding = evidence["seizure_channel_embedding"].float()
    seizure_valid = evidence["seizure_mask"].bool().unsqueeze(-1) & evidence["seizure_channel_mask"].bool() & valid.unsqueeze(1)
    q_seizure = seizure_prototype_probability(seizure_embedding, seizure_valid, seizure_prototype)
    stability = compute_seizure_nez_stability(q_seizure, seizure_valid, target, valid)

    adjacency = evidence.get("graph_adjacency")
    if adjacency is None:
        b, s, c = seizure_valid.shape
        adjacency = patient_embedding.new_zeros((b, s, len(NETWORK_PHASES), c, c))
        phase_mask = seizure_valid.unsqueeze(2).expand(-1, -1, len(NETWORK_PHASES), -1)
        graph_valid = torch.zeros((b, s, len(NETWORK_PHASES)), dtype=torch.bool, device=valid.device)
    else:
        phase_mask = evidence["graph_phase_channel_mask"].bool() & valid[:, None, None, :]
        graph_valid = evidence["graph_valid"].bool()
    virtual = compute_virtual_target_network(adjacency, target, consensus["reliable_abnormality"], stability["persistent_abnormality"], phase_mask, graph_valid)
    dispersion = compute_abnormality_dispersion(consensus["reliable_abnormality"], target, valid, adjacency, graph_valid)
    strengths = _phase_strength(adjacency, phase_mask, graph_valid)
    view = _view_features(q_final, q_direct, target, valid)

    scalars: dict[str, torch.Tensor] = {}
    for mapping in (view, residual, stability, dispersion, virtual):
        for key, value in mapping.items():
            if key in ALL_SCALAR_FEATURES: scalars[key] = value
    scalars["view_disagreement_mean"] = (consensus["view_disagreement"] * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    scalars["reliability_weight_mean"] = (consensus["reliability_weight"] * valid).sum(-1) / valid.sum(-1).clamp_min(1)
    for key in ALL_SCALAR_FEATURES:
        if key not in scalars: scalars[key] = patient_embedding.new_zeros(patient_embedding.shape[0])

    channel_scalars = torch.stack((
        q_final, q_direct, q_proto, consensus["q_consensus_nez"], consensus["view_disagreement"], consensus["reliability_weight"],
        consensus["reliable_abnormality"], stability["persistent_abnormality"], stability["worstcase_abnormality"], target.float(),
        strengths[:, 0], strengths[:, 1], strengths[:, 2],
    ), dim=-1)
    return {"scalars": scalars, "patient_channel_embedding": patient_embedding, "channel_scalars": channel_scalars,
            "target": target, "channel_mask": valid, "reliable_abnormality": consensus["reliable_abnormality"],
            "q_final_nez": q_final, "q_direct_nez": q_direct, "q_proto_nez": q_proto,
            "q_consensus_nez": consensus["q_consensus_nez"], "persistent_abnormality": stability["persistent_abnormality"],
            "worstcase_abnormality": stability["worstcase_abnormality"], "virtual_target_network_audit": virtual["virtual_target_network_audit"],
            "seizure_nez_stability_audit": stability["seizure_nez_stability_audit"]}


def make_counterfactual_targets(features: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target, valid = features["target"].clone(), features["channel_mask"].bool()
    plus, minus, cf_valid = target.clone(), target.clone(), torch.zeros(target.shape[0], dtype=torch.bool, device=target.device)
    risk = features["reliable_abnormality"]
    for p in range(target.shape[0]):
        outside = torch.where(valid[p] & ~target[p] & (risk[p] > .7))[0]
        inside = torch.where(valid[p] & target[p])[0]
        if outside.numel() and inside.numel():
            plus[p, outside[torch.argmax(risk[p, outside])]] = True
            minus[p, inside[torch.argmax(risk[p, inside])]] = False
            if plus[p].any() and (~plus[p] & valid[p]).any() and minus[p].any() and (~minus[p] & valid[p]).any(): cf_valid[p] = True
    return plus, minus, cf_valid


__all__ = ["ALL_SCALAR_FEATURES", "BASE_VIEW_FEATURES", "build_nvr_features", "make_counterfactual_targets", "scalar_features_for_profile"]
