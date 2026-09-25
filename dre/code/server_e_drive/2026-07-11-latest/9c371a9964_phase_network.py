from __future__ import annotations

import torch
from torch import nn


def assemble_phase_network_features(
    signatures: torch.Tensor,
    phase_valid: torch.Tensor,
    *,
    include_deltas: bool,
) -> dict[str, torch.Tensor]:
    if signatures.ndim < 3 or signatures.shape[-2] < 2:
        raise ValueError("signatures must end with [phase,feature] and contain at least two phases")
    n_phases, n_features = signatures.shape[-2:]
    phase_ok = phase_valid.bool()
    flattened = signatures.reshape(*signatures.shape[:-2], n_phases * n_features)
    if not include_deltas:
        vector = torch.cat((flattened, phase_ok.to(signatures.dtype)), dim=-1)
        return {"network_feature_vector": vector, "phase_delta": signatures.new_zeros((*signatures.shape[:-2], n_phases - 1, n_features)), "delta_valid": phase_ok[..., 1:] & False}
    delta_valid = phase_ok[..., :-1] & phase_ok[..., 1:]
    deltas = signatures[..., 1:, :] - signatures[..., :-1, :]
    deltas = deltas.masked_fill(~delta_valid.unsqueeze(-1), 0.0)
    vector = torch.cat((flattened, deltas.reshape(*deltas.shape[:-2], (n_phases - 1) * n_features), phase_ok.to(signatures.dtype), delta_valid.to(signatures.dtype)), dim=-1)
    return {"network_feature_vector": vector, "phase_delta": deltas, "delta_valid": delta_valid}


class PhaseNetworkEncoder(nn.Module):
    def __init__(self, *, include_deltas: bool, n_phases: int = 3, n_features: int = 5) -> None:
        super().__init__()
        self.include_deltas = bool(include_deltas)
        input_dim = n_phases * n_features + n_phases
        if include_deltas:
            input_dim += (n_phases - 1) * n_features + (n_phases - 1)
        self.projection = nn.Sequential(nn.LayerNorm(input_dim), nn.Linear(input_dim, 32), nn.GELU(), nn.Dropout(0.10))

    def forward(self, signatures: torch.Tensor, phase_valid: torch.Tensor) -> dict[str, torch.Tensor]:
        output = assemble_phase_network_features(signatures, phase_valid, include_deltas=self.include_deltas)
        output["network_stat_embedding"] = self.projection(output["network_feature_vector"])
        return output


__all__ = ["PhaseNetworkEncoder", "assemble_phase_network_features"]
