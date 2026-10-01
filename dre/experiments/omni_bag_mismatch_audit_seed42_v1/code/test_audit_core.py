import numpy as np
import pandas as pd

from audit_core import metrics, select_indices


def test_sampling_is_reproducible_without_replacement_and_uses_all_when_short():
    first = select_indices(20, 5, 42, "patient/example.edf")
    second = select_indices(20, 5, 42, "patient/example.edf")
    assert np.array_equal(first, second)
    assert len(np.unique(first)) == 5
    assert np.array_equal(select_indices(3, 5, 42, "patient/example.edf"), np.arange(3))


def test_sampling_is_shared_when_called_with_same_edf():
    assert np.array_equal(select_indices(20, 5, 42, "x.edf"), select_indices(20, 5, 42, "x.edf"))


def test_ranking_metrics_have_known_values():
    frame = pd.DataFrame({
        "patient": ["a", "a", "b", "b"], "channel": ["1", "2", "1", "2"],
        "y": [1, 0, 0, 1], "score": [.9, .1, .8, .7],
    })
    out = metrics(frame)
    assert out["auroc"] == .75
    assert out["top1"] == .5
    assert out["mrr"] == .75
    assert out["patient_equal_ap"] == .75
