from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import HybridA1RawModel, PatientRelativeHead, RawTinyPatientModel, scores_from_nez_logits
from seizure_aggregator import CrossSeizureMILAggregator
from temporal_encoder import ChannelTemporalEncoder


class FakeA1(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.b0_encoder = nn.Linear(4, 32)
        self.temporal = ChannelTemporalEncoder(model_dim=32, pooling="mean")
        self.seizure_aggregator = CrossSeizureMILAggregator(model_dim=32, pooling="mean")
        self.classifier = nn.Linear(64, 1)

    def forward(self, batch: dict) -> dict:
        w = self.b0_encoder(batch["b0_features"])
        s, _ = self.temporal(w, batch["seizure_channel_mask"], window_mask=batch["window_mask"])
        u, _ = self.seizure_aggregator(s, batch["seizure_mask"], batch["seizure_channel_mask"])
        out = scores_from_nez_logits(self.classifier(u).squeeze(-1), batch["channel_mask"])
        out["patient_channel_embedding"] = u
        return out


def batch() -> dict:
    torch.manual_seed(42)
    b, s, c, w, t = 1, 2, 4, 3, 500
    return {
        "raw_windows": torch.randn(b, s, c, w, t),
        "raw_window_mask": torch.ones(b, s, c, w, dtype=torch.bool),
        "b0_features": torch.randn(b, s, w, c, 4),
        "window_mask": torch.ones(b, s, w, dtype=torch.bool),
        "seizure_mask": torch.ones(b, s, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(b, s, c, dtype=torch.bool),
        "raw_seizure_channel_mask": torch.ones(b, s, c, dtype=torch.bool),
        "channel_mask": torch.ones(b, c, dtype=torch.bool),
    }


def run() -> None:
    x = batch()
    for relative in (False, True):
        model = RawTinyPatientModel(patient_relative=relative).eval()
        y = model(x)
        assert y["logits"].shape == (1, 4)
        assert torch.isfinite(y["logits"]).all()
        if relative:
            perm = torch.tensor([2, 0, 3, 1])
            xp = copy.deepcopy(x)
            xp["raw_windows"] = x["raw_windows"][:, :, perm]
            xp["raw_window_mask"] = x["raw_window_mask"][:, :, perm]
            xp["seizure_channel_mask"] = x["seizure_channel_mask"][:, :, perm]
            xp["raw_seizure_channel_mask"] = x["raw_seizure_channel_mask"][:, :, perm]
            xp["channel_mask"] = x["channel_mask"][:, perm]
            yp = model(xp)
            assert torch.allclose(yp["logits"], y["logits"][:, perm], atol=1e-5)
    a1 = FakeA1().eval()
    reference = a1(x)["logits"]
    hybrid = HybridA1RawModel(a1).eval()
    replay = hybrid(x)["logits"]
    assert torch.max(torch.abs(replay - reference)).item() < 1e-6
    assert torch.equal((replay > 0), (reference > 0))
    head = PatientRelativeHead(64)
    p = torch.rand(1, 4, 64)
    z, rank = head.relative_parts(p, torch.ones(1, 4, dtype=torch.bool))
    assert z.shape == p.shape and rank.shape == (1, 4, 1)
    assert torch.equal(torch.sort(rank[0, :, 0]).values, torch.tensor([0., 1/3, 2/3, 1.]))
    print("STAGE_A_MODEL_UNIT_TEST_PASS", flush=True)


if __name__ == "__main__":
    run()
