"""Shape, masking, architecture and beta-zero identity smoke tests."""
import torch

from late_tf import TFScorer, fused_margin, robust_channel_normalize


def test_tf_independence_and_beta_zero_identity():
    torch.manual_seed(42)
    model = TFScorer().eval()
    assert sum(p.numel() for p in model.parameters()) == 45377
    batch = {
        "tf_log_power": torch.randn(2, 2, 59, 8, 32),
        "window_mask": torch.ones(2, 2, 59, dtype=torch.bool),
        "seizure_channel_mask": torch.ones(2, 2, 8, dtype=torch.bool),
        "channel_mask": torch.ones(2, 8, dtype=torch.bool),
    }
    batch["seizure_channel_mask"][0, 1, 7] = False
    result = model(batch)
    assert result["margin_tf"].shape == (2, 8)
    a1 = torch.randn(2, 8)
    assert torch.equal(fused_margin(a1, result["margin_tf"], batch["channel_mask"], 0), a1)
    z = robust_channel_normalize(result["margin_tf"], batch["channel_mask"])
    assert torch.isfinite(z).all()
    assert not any("a1" in key.lower() for key in model.state_dict())
