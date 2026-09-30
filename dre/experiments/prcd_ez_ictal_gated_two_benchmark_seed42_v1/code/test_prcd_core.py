import numpy as np
from sklearn.metrics import average_precision_score, balanced_accuracy_score, f1_score, roc_auc_score

from prcd_core import (GROUPS, INPUT_DIM, KERNELS, POOLED_DIM, RECORD_DIM,
                       average_tie_rank, dictionary_record, fine_spectrum,
                       fit_biases, make_dictionary, patient_coordinates,
                       pool_records)
from run_ictal import ranking_metrics, threshold_metrics


def test_dictionary_is_deterministic_zero_sum_and_normalized():
    left, right = make_dictionary(), make_dictionary()
    assert len(left) == GROUPS and all(len(group) == KERNELS for group in left)
    for a, b in zip(sum(left, []), sum(right, [])):
        assert a.dimensions == b.dimensions and a.dilation == b.dilation
        assert np.array_equal(a.weights, b.weights)
        assert np.allclose(a.weights.sum(axis=1), 0, atol=1e-7)
        assert np.isclose(np.linalg.norm(a.weights), 1, atol=1e-6)
        assert max(a.dimensions) < INPUT_DIM and a.dilation in (1, 2, 4)


def test_fine_spectrum_shape_and_finite():
    rng = np.random.default_rng(42)
    wave = rng.normal(size=(3, 15000)).astype(np.float32)
    values, mask = fine_spectrum(wave, 250.0, np.ones((3, 59), dtype=bool))
    assert values.shape == mask.shape == (3, 59, 16)
    assert mask.all() and np.isfinite(values).all()
    short, short_mask = fine_spectrum(wave[:, :14250], 250.0,
                                      np.c_[np.ones((3, 56), dtype=bool), np.zeros((3, 3), dtype=bool)])
    assert short.shape == (3, 59, 16)
    assert not short_mask[:, -3:].any() and np.all(short[:, -3:] == 0)


def test_competitive_statistics_and_pooling():
    rng = np.random.default_rng(1)
    dictionary = make_dictionary()
    sample = rng.normal(size=(12, 59, 52)).astype(np.float32)
    mask = np.ones_like(sample, dtype=bool)
    biases = fit_biases(sample, mask, dictionary)
    output = dictionary_record(sample[:4], mask[:4], dictionary, biases)
    assert output.shape == (4, RECORD_DIM) and np.isfinite(output).all()
    # Occupancy sums are one for every group/view.
    for view in range(2):
        for group in range(48):
            start = view * 672 + group * 14
            assert np.isclose(output[:, start:start + 6].sum(axis=1), 1).all()
    records = np.stack((output, output + .1))
    pooled = pool_records(records, np.ones((2, 4), dtype=bool))
    assert pooled.shape == (4, POOLED_DIM) and np.isfinite(pooled).all()
    partial = mask[:4].copy(); partial[:, -3:] = False
    shortened = dictionary_record(sample[:4], partial, dictionary, biases)
    assert shortened.shape == (4, RECORD_DIM) and np.isfinite(shortened).all()


def test_average_tie_rank_and_patient_coordinates():
    values = np.asarray([[1, 5], [1, 7], [3, 6]], dtype=np.float32)
    rank = average_tie_rank(values)
    assert np.allclose(rank[:, 0], [-.5, -.5, 1])
    one = average_tie_rank(values[:1])
    assert np.array_equal(one, np.zeros_like(one))
    absolute = np.zeros((4, POOLED_DIM), dtype=np.float32)
    absolute[:3, :2] = values
    coordinate = patient_coordinates(absolute, np.asarray([1, 1, 1, 0], dtype=bool))
    assert coordinate.shape == (4, POOLED_DIM * 3)
    assert np.allclose(coordinate[:3, POOLED_DIM:POOLED_DIM + 2], values - np.median(values, axis=0))
    assert np.allclose(coordinate[:3, POOLED_DIM * 2:POOLED_DIM * 2 + 2], rank)


def test_fast_metrics_match_sklearn_including_ties():
    rng = np.random.default_rng(42)
    for size in (4, 11, 100):
        for _ in range(10):
            labels = rng.integers(0, 2, size)
            if len(np.unique(labels)) < 2:
                continue
            scores = np.round(rng.random(size), 1)
            rank = ranking_metrics(labels, scores)
            assert np.isclose(rank["auroc"], roc_auc_score(labels, scores))
            assert np.isclose(rank["ap"], average_precision_score(labels, scores))
            threshold = threshold_metrics(labels, scores, .5); prediction = scores >= .5
            assert np.isclose(threshold["macro_f1"], f1_score(labels, prediction, average="macro"))
            assert np.isclose(threshold["balanced_accuracy"], balanced_accuracy_score(labels, prediction))
