from __future__ import annotations

import torch


def compute_nez_consensus(q_final_nez: torch.Tensor, q_direct_nez: torch.Tensor, q_proto_nez: torch.Tensor, temperature: float = .15) -> dict[str, torch.Tensor]:
    views = torch.stack((q_final_nez, q_direct_nez, q_proto_nez), dim=-1).clamp(0, 1)
    consensus = views.median(dim=-1).values
    disagreement = (views - consensus.unsqueeze(-1)).abs().median(dim=-1).values
    reliability = torch.exp(-disagreement / float(temperature)).clamp(0, 1)
    abnormality = ((1 - consensus) * reliability).clamp(0, 1)
    return {"q_consensus_nez": consensus, "view_disagreement": disagreement, "reliability_weight": reliability, "reliable_abnormality": abnormality}


__all__ = ["compute_nez_consensus"]
