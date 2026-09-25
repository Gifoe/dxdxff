from __future__ import annotations

import pandas as pd
import pytest

from outcome_hifos.training.experiment_runner import _validate_resume_predictions


def _rows() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "subject_id": ["a"],
            "config_hash": ["config"],
            "fold_ledger_hash": ["fold"],
            "cohort_id": ["feature_full"],
            "subject_set_hash": ["subjects"],
        }
    )


def test_resume_accepts_only_matching_config_fold_and_cohort_hashes() -> None:
    _validate_resume_predictions(_rows(), config_hash="config", fold_hash="fold", cohort_id="feature_full", subject_set_hash="subjects")
    for column, value in (("config_hash", "old"), ("fold_ledger_hash", "old"), ("cohort_id", "old"), ("subject_set_hash", "old")):
        stale = _rows()
        stale.loc[0, column] = value
        with pytest.raises(ValueError, match="resume artifact"):
            _validate_resume_predictions(stale, config_hash="config", fold_hash="fold", cohort_id="feature_full", subject_set_hash="subjects")
