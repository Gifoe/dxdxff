"""Pre-unseal checks for historical fixed-query evaluation semantics."""

import numpy as np

from evaluate_ictal_frozen import (fixed_query, fixed_query_rows,
                                   original_channel_order, paired_query_bootstrap,
                                   query_metrics)


def test_historical_membership_and_paired_20_queries():
    assert fixed_query(12, 1, "sub-test", 0).tolist() == [8, 3, 4, 6, 7, 5]
    labels = [0, 1] * 6
    raw = {"sub-test": {"labels": labels, "scores": np.linspace(.1, .9, 12).tolist()}}
    left = fixed_query_rows(raw, 1)
    right = fixed_query_rows(raw, 1)
    assert len(left) == 20
    estimates = paired_query_bootstrap(left, right, ["sub-test"],
                                       np.zeros((8, 1), dtype=np.int64))
    assert all(value["mean_delta"] == 0 for value in estimates.values())


def test_nonestimable_rank_and_threshold_tie_rule():
    row = query_metrics([0, 0, 0, 0], [.1, .2, .3, .4])
    assert row["auroc"] is None and row["ap"] is None
    assert np.isclose(query_metrics([0, 1, 1, 0], [.2, .5, .8, .1])["macro_f1"],
                      (2 / 3 + 4 / 5) / 2)


def test_historical_query_uses_original_channel_order(tmp_path):
    source = tmp_path / "synthetic.npz"
    np.savez(source, channel_names=["Z", "A", "M", "B"], labels=[1, 0, 1, 0])

    class Bank:
        def path(self, patient):
            assert patient == "synthetic"
            return source

    # Training metrics store alphabetically: A, B, M, Z.
    ordered = original_channel_order(Bank(), "synthetic",
                                     {"labels": [0, 0, 1, 1],
                                      "scores": [.1, .2, .8, .9]})
    assert ordered == {"labels": [1, 0, 1, 0],
                       "scores": [.9, .1, .8, .2]}
