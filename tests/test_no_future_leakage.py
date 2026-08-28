"""Causality regression tests for the G1 CPU feature extractor."""

from __future__ import annotations

import math

import numpy as np

from reva_dlm.features import extract_causal_features


def _trajectory():
    tokens = [
        np.array([9, 9, 9, 9]),
        np.array([1, 9, 9, 9]),
        np.array([1, 2, 9, 9]),
        np.array([1, 2, 3, 9]),
        np.array([1, 2, 4, 9]),
        np.array([7, 7, 7, 7]),
    ]
    committed = [
        np.array([0, 0, 0, 0], dtype=bool),
        np.array([1, 0, 0, 0], dtype=bool),
        np.array([1, 1, 0, 0], dtype=bool),
        np.array([1, 1, 1, 0], dtype=bool),
        np.array([1, 1, 1, 1], dtype=bool),
        np.array([0, 0, 0, 0], dtype=bool),
    ]
    answers = [None, "11", "12", "11", "13", "future answer"]
    texts = ["", "work 11", "work 12", "reconsider 11", "answer 13", "future"]
    return tokens, committed, answers, texts


def test_changing_every_future_history_does_not_change_feature_t():
    checkpoint = 3
    histories = _trajectory()
    expected = extract_causal_features(
        *histories, checkpoint, window=3, total_steps=256
    )

    changed = tuple(list(history) for history in histories)
    changed[0][4:] = [np.array([100, 200]), np.array([300])]
    changed[1][4:] = [np.array([0, 0]), np.array([1])]
    changed[2][4:] = ["entirely different", None]
    changed[3][4:] = ["mutated suffix A", "mutated suffix B"]

    observed = extract_causal_features(
        *changed, checkpoint, window=3, total_steps=256
    )
    assert observed.keys() == expected.keys()
    assert observed == expected


def test_appending_future_states_does_not_change_feature_t():
    checkpoint = 3
    histories = _trajectory()
    expected = extract_causal_features(
        *histories, checkpoint, window=16, total_steps=256
    )

    extended = tuple(list(history) for history in histories)
    for step in range(20):
        extended[0].append(np.array([step, step + 1, step + 2]))
        extended[1].append(np.array([step % 2, 1, 0], dtype=bool))
        extended[2].append(f"future-{step}")
        extended[3].append(f"arbitrary future text {step}")

    assert (
        extract_causal_features(
            *extended, checkpoint, window=16, total_steps=256
        )
        == expected
    )


def test_features_are_finite_numeric_and_have_no_label_or_future_names():
    features = extract_causal_features(
        *_trajectory(), 4, window=4, total_steps=256
    )
    forbidden = ("correct", "ground_truth", "label", "final", "future")

    assert features
    assert all(not any(term in name.lower() for term in forbidden) for name in features)
    assert all(isinstance(value, float) for value in features.values())
    assert all(math.isfinite(value) for value in features.values())
