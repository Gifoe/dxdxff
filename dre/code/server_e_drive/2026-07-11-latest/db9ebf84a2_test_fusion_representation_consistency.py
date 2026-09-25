from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from outcome_hifos.training.late_fusion import fit_cross_fitted_stacker


def _inner(representation: str = "raw_logit") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": [f"p{i}" for i in range(8)],
            "outcome": [0, 0, 0, 0, 1, 1, 1, 1],
            "raw_logit": np.linspace(-2, 2, 8),
            "role": "inner_oof",
            "outer_fold_idx": 1,
            "inner_fold_idx": [0, 1] * 4,
            "seed": 42,
            "input_representation": representation,
        }
    )


def _outer(representation: str = "raw_logit") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["z0", "z1"],
            "feature_raw_logit": [-0.5, 0.5],
            "fm_raw_logit": [-0.2, 0.8],
            "role": "outer_test",
            "outer_fold_idx": 1,
            "seed": 42,
            "feature_input_representation": representation,
            "fm_input_representation": representation,
        }
    )


def test_stacker_uses_raw_logits_for_inner_and_outer() -> None:
    result = fit_cross_fitted_stacker(_inner(), _inner(), _outer())
    assert result.input_representation == "raw_logit"
    assert result.outer_predictions["fusion_probability"].between(0.0, 1.0).all()


@pytest.mark.parametrize("inner_rep,outer_rep", [("probability", "raw_logit"), ("calibrated_probability", "probability")])
def test_stacker_rejects_inner_outer_representation_mismatch(inner_rep: str, outer_rep: str) -> None:
    with pytest.raises(ValueError, match="representation"):
        fit_cross_fitted_stacker(_inner(inner_rep), _inner(inner_rep), _outer(outer_rep))
