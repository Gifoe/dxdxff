from __future__ import annotations

import numpy as np
import pandas as pd

from outcome_hifos.baselines.patient_summary import build_patient_summary_matrix
from outcome_hifos.baselines.protocol import build_task2_primary_cohort
from outcome_hifos.dataset import OutcomePatientExample


def _example(subject: str, target: int) -> OutcomePatientExample:
    return OutcomePatientExample(
        subject_id=subject,
        center="c1",
        target=float(target),
        canonical_channels=("A1", "A2"),
        model_input={
            "feature_runs": [np.asarray([[[1.0], [2.0]], [[3.0], [4.0]]], dtype=np.float32)],
            "window_centers": [np.asarray([-1.0, 1.0], dtype=np.float32)],
            "seizure_channel_mask": [np.asarray([True, True])],
            "canonical_index": np.asarray([0, 1]),
        },
        side_metadata={"run_ids": ("r1",)},
    )


def test_primary_cohort_is_old90_success_plus_all_valid_failures() -> None:
    outcome = pd.DataFrame(
        [
            {"subject_id": "s1", "center": "c1", "outcome_group": "success", "outcome_label": 1},
            {"subject_id": "s2", "center": "c1", "outcome_group": "success", "outcome_label": 1},
            {"subject_id": "s3", "center": "c2", "outcome_group": "success", "outcome_label": 1},
            {"subject_id": "f1", "center": "c1", "outcome_group": "failure", "outcome_label": 0},
            {"subject_id": "u1", "center": "c2", "outcome_group": "unknown", "outcome_label": np.nan},
        ]
    )
    cohort, audit = build_task2_primary_cohort(outcome, old90_subjects={"s1", "s2"})
    assert set(cohort["subject_id"]) == {"s1", "s2", "f1"}
    assert set(cohort.loc[cohort["outcome_label"] == 1, "subject_id"]) == {"s1", "s2"}
    assert audit["extra_success_excluded"] == 1


def test_patient_summary_is_label_blind_and_excludes_counts_by_default() -> None:
    frame, manifest = build_patient_summary_matrix([_example("s1", 1), _example("f1", 0)])
    forbidden = ("ez", "soz", "resect", "outcome", "engel", "success", "label", "count")
    feature_names = manifest["feature_names"]
    assert feature_names
    assert not any(any(fragment in name.lower() for fragment in forbidden) for name in feature_names)
    assert frame.shape[0] == 2
    assert frame["subject_id"].tolist() == ["f1", "s1"]
