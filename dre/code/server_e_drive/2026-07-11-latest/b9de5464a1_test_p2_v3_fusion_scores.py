import numpy as np
import pytest

from neuroez_c.p2_v3_conservative_fusion import conservative_probability_fusion


def test_probability_fusion_is_fixed_and_bounded():
    nez, ez = conservative_probability_fusion(np.array([.2, .8]), np.array([.6, .4]))
    assert np.allclose(nez, [.24, .76]) and np.allclose(ez, [.76, .24])


def test_probability_fusion_rejects_bad_weights_and_logits_as_probabilities():
    with pytest.raises(ValueError, match="sum"):
        conservative_probability_fusion(np.array([.2]), np.array([.4]), p2_weight=.9, v3_weight=.2)
    with pytest.raises(ValueError, match="\[0, 1\]"):
        conservative_probability_fusion(np.array([2.0]), np.array([.4]))


def test_probability_and_logit_fusion_are_not_interchangeable():
    score, _ = conservative_probability_fusion(np.array([.9]), np.array([.5]))
    assert score[0] != .9 * 2.197224577 + .1 * 0.0
