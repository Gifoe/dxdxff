from __future__ import annotations

import pandas as pd
import pytest

from outcome_hifos.training.outer_cv import select_best_profile


def test_candidate_profiles_use_inner_oof_metrics_and_deterministic_tie_breaks() -> None:
    candidates = pd.DataFrame(
        [
            {"candidate_profile": "large", "macro_f1": 0.7, "auroc": 0.8, "brier": 0.2, "parameter_count": 200},
            {"candidate_profile": "compact", "macro_f1": 0.7, "auroc": 0.8, "brier": 0.2, "parameter_count": 100},
            {"candidate_profile": "worse", "macro_f1": 0.6, "auroc": 0.9, "brier": 0.1, "parameter_count": 50},
        ]
    )
    selected = select_best_profile(candidates)
    assert selected["candidate_profile"] == "compact"


def test_candidate_selection_rejects_outer_test_rows() -> None:
    candidates = pd.DataFrame(
        [{"candidate_profile": "bad", "role": "outer_test", "macro_f1": 1.0, "auroc": 1.0, "brier": 0.0, "parameter_count": 1}]
    )
    with pytest.raises(ValueError, match="inner_oof"):
        select_best_profile(candidates)
