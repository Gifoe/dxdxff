import numpy as np
import pandas as pd
import pytest

from P2_V3_AAAI_ABLATIONS import ABLATION_STATUS
from P2_V3_AAAI_ABLATIONS.bootstrap import paired_bootstrap
from P2_V3_AAAI_ABLATIONS.fusion_operators import (
    logit_fusion, patient_rank_fusion, probability_fusion, shuffle_v3_within_patient, stable_shuffle_seed,
)
from P2_V3_AAAI_ABLATIONS.input_loader import config_difference, load_optional_model
from P2_V3_AAAI_ABLATIONS.metrics import aggregate, evaluate
from P2_V3_AAAI_ABLATIONS.score_semantics import as_probability, canonical_probability_columns, sigmoid
from P2_V3_AAAI_ABLATIONS.threshold_protocol import select_fold_threshold, threshold_candidates


def frame():
    return pd.DataFrame({
        "subject_id": ["hup:a"] * 3 + ["lzu:b"] * 3,
        "center": ["hup"] * 3 + ["lzu"] * 3,
        "outer_fold": [1] * 6,
        "channel_name": ["A", "B", "C", "A", "B", "C"],
        "label_nez": [0, 1, 1, 0, 0, 1], "label_ez": [1, 0, 0, 1, 1, 0],
        "p2_score_nez": [.1, .6, .9, .2, .4, .8], "v3_score_nez": [.2, .7, .8, .1, .5, .9],
    })


def test_nez_one_ez_zero_contract():
    result = canonical_probability_columns(frame().rename(columns={"p2_score_nez": "score_nez"}))
    assert (result.label_nez + result.label_ez).eq(1).all()


def test_invalid_label_contract_fails():
    bad = frame().rename(columns={"p2_score_nez": "score_nez"}); bad.loc[0, "label_ez"] = 0
    with pytest.raises(ValueError): canonical_probability_columns(bad)


def test_probability_range_checked():
    with pytest.raises(ValueError): as_probability([1.1])


def test_logit_sigmoid_conversion():
    assert np.allclose(sigmoid([0.0]), [0.5])


def test_v3_weight_zero_equals_p2():
    data = frame(); assert np.array_equal(probability_fusion(data.p2_score_nez, data.v3_score_nez, v3_weight=0), data.p2_score_nez)


def test_v3_weight_one_equals_v3():
    data = frame(); assert np.array_equal(probability_fusion(data.p2_score_nez, data.v3_score_nez, v3_weight=1), data.v3_score_nez)


def test_locked_probability_formula():
    data = frame(); actual = probability_fusion(data.p2_score_nez, data.v3_score_nez, v3_weight=.1)
    assert np.allclose(actual, .9 * data.p2_score_nez + .1 * data.v3_score_nez)


def test_probability_shape_mismatch_fails():
    with pytest.raises(ValueError): probability_fusion([.1], [.1, .2], v3_weight=.1)


def test_logit_fusion_is_finite_at_boundaries():
    assert np.isfinite(logit_fusion([0, 1], [1, 0], v3_weight=.1)).all()


def test_logit_and_probability_fusion_are_distinct():
    assert not np.allclose(probability_fusion([.01], [.9], v3_weight=.1), logit_fusion([.01], [.9], v3_weight=.1))


def test_rank_fusion_in_unit_interval():
    result = patient_rank_fusion(frame(), v3_weight=.1); assert np.all((result >= 0) & (result <= 1))


def test_rank_fusion_is_patient_local():
    data = frame(); base = patient_rank_fusion(data, v3_weight=.1)
    data.loc[data.subject_id.eq("lzu:b"), ["p2_score_nez", "v3_score_nez"]] += 100
    changed = patient_rank_fusion(data, v3_weight=.1)
    assert np.array_equal(base[:3], changed[:3])


def test_threshold_grid_is_exact():
    grid = threshold_candidates(); assert len(grid) == 201 and grid[0] == 0 and grid[-1] == 1


def test_threshold_step_cannot_change():
    with pytest.raises(ValueError): threshold_candidates(.01)


def test_threshold_selects_exactly_one():
    data = frame(); data["score"] = data.p2_score_nez
    _, search = select_fold_threshold(data, score_column="score"); assert search.selected.sum() == 1


def test_threshold_tie_prefers_closest_to_point5():
    data = frame(); data["score"] = .5
    threshold, _ = select_fold_threshold(data, score_column="score"); assert abs(threshold - .5) <= .005


def test_test_labels_cannot_change_validation_threshold():
    validation = frame(); validation["score"] = validation.p2_score_nez
    first, _ = select_fold_threshold(validation, score_column="score")
    test = frame(); test["label_nez"] = 1 - test.label_nez
    second, _ = select_fold_threshold(validation, score_column="score")
    assert first == second


def test_patient_macro_not_channel_pooled():
    data = frame(); data["score"] = data.p2_score_nez
    rows = evaluate(data, score_column="score", threshold=.5, experiment="x", analysis_status=ABLATION_STATUS)
    assert aggregate(rows).patient_macro_f1.iloc[0] == pytest.approx(rows.patient_macro_f1.mean())


def test_true_k_not_used_for_formal_prediction():
    data = frame(); data["score"] = data.p2_score_nez
    rows = evaluate(data, score_column="score", threshold=.5, experiment="x", analysis_status=ABLATION_STATUS)
    assert rows.true_count_used_for_prediction.eq(False).all()


def test_single_class_auprc_is_nan():
    data = frame().iloc[:3].copy(); data.label_nez = 1; data.label_ez = 0; data["score"] = data.p2_score_nez
    rows = evaluate(data, score_column="score", threshold=.5, experiment="x", analysis_status=ABLATION_STATUS)
    assert np.isnan(rows.patient_ez_auprc.iloc[0])


def test_shuffle_preserves_patient_multiset():
    data = frame(); shuffled, _ = shuffle_v3_within_patient(data, experiment="x", fold=1, partition="test", repeat=0, seed=42)
    for subject, group in data.groupby("subject_id"):
        assert np.array_equal(np.sort(group.v3_score_nez), np.sort(shuffled[shuffled.subject_id.eq(subject)].v3_score_nez))


def test_shuffle_preserves_keys_and_labels():
    data = frame(); shuffled, _ = shuffle_v3_within_patient(data, experiment="x", fold=1, partition="test", repeat=0, seed=42)
    assert data[["subject_id", "channel_name", "label_nez"]].equals(shuffled[["subject_id", "channel_name", "label_nez"]])


def test_shuffle_seed_is_stable():
    assert stable_shuffle_seed("x", 1, 42) == stable_shuffle_seed("x", 1, 42)


def test_shuffle_partition_changes_seed():
    assert stable_shuffle_seed("x", "validation") != stable_shuffle_seed("x", "test")


def test_bootstrap_is_patient_level_and_deterministic():
    data = frame(); data["score"] = data.p2_score_nez
    a = evaluate(data, score_column="score", threshold=.5, experiment="a", analysis_status=ABLATION_STATUS)
    b = a.copy(); b.patient_macro_f1 -= .1
    first = paired_bootstrap(a, b, comparison="x", repeats=20, seed=42); second = paired_bootstrap(a, b, comparison="x", repeats=20, seed=42)
    assert first == second and first["n_patients"] == 2


def test_missing_optional_component_is_explicit():
    model, audit = load_optional_model(None, {}, model_name="P2_TEMPORAL_NO_Q10")
    assert model is None and audit["status"] == "UNAVAILABLE_MISSING_INPUT"


def test_config_difference_does_not_claim_pure_q10_without_proof():
    audit = config_difference([{"model": "a", "config": {"x": 1}}, {"model": "b", "config": {"x": 2}}])
    assert audit["pure_q10_ablation_confirmed"] is False

