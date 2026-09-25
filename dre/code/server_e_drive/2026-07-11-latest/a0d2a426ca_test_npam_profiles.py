import torch

from conftest import synthetic_p2
from neuroez_c.task2.p2_q10_npam_model import P2Q10NPAMModel
from neuroez_c.task2.profiles import get_profile, profile_names


def test_ordered_profiles_and_synthetic_forward():
    expected = ("M0_PAM", "M1_NETWORK_STATS", "M2_PHASE_NETWORK", "M3_GRAPH_RESIDUAL", "M4_GRAPH_STABILITY", "M5_LIMITED_FINETUNE")
    assert profile_names()[:6] == expected
    p2, graphs = synthetic_p2(dim=8)
    for name in expected:
        model = P2Q10NPAMModel(8, name).eval()
        with torch.no_grad():
            out = model(p2, graphs if get_profile(name).network_stats else None)
        assert out["outcome_logit_success"].shape == (2,)
        assert torch.allclose(out["outcome_probability_failure"], 1 - out["outcome_probability_success"])
        assert torch.all((out["outcome_probability_failure"] >= 0) & (out["outcome_probability_failure"] <= 1))
        if name == "M0_PAM":
            assert "network_burden" not in out
        if name in expected[1:]:
            assert out["network_burden"].shape[-2:] == (3, 5)
        assert torch.allclose(out["q10_residual"], out["q10_adjustment"])
