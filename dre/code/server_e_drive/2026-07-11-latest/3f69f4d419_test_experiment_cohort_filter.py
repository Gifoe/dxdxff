from __future__ import annotations

from outcome_hifos.training.experiment_runner import _configured_cohort_subjects


def test_configured_task2_cohort_subjects_are_explicit_and_unique() -> None:
    assert _configured_cohort_subjects({"cohort": {"primary_subject_ids": ["p2", "p1", "p2"]}}) == ["p1", "p2"]
    assert _configured_cohort_subjects({}) is None
