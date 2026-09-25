from neuroez_c.p23_profiles import get_p23_profile, profile_names


def test_profile_progression_and_default_causal_policy():
    assert not get_p23_profile("P5_FULL").causal
    assert get_p23_profile("P6_FULL_WITH_CAUSAL").causal
    assert get_p23_profile("P1_TEMPORAL").temporal
    assert len(profile_names()) == 8
    lite = get_p23_profile("P23_LITE")
    assert lite.temporal and lite.seizure_tail and lite.bounded_fusion
    assert not lite.noise_aware and not lite.causal
    assert lite.min_seizures_for_tail == 2


def test_profile_ablation_progression_is_explicit():
    p1 = get_p23_profile("P1_TEMPORAL")
    p2 = get_p23_profile("P2_TEMPORAL_Q10")
    p3 = get_p23_profile("P3_BOUNDED_FUSION")
    p4 = get_p23_profile("P4_NOISE_AWARE")
    assert p1.temporal and not p1.seizure_tail and not p1.bounded_fusion
    assert p2.seizure_tail and not p2.bounded_fusion
    assert p3.bounded_fusion and not p3.noise_aware
    assert p4.noise_aware and not p4.causal
