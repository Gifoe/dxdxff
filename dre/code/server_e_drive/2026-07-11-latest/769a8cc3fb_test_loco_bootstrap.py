import pandas as pd
import pytest

from task1_confirmatory.loco_bootstrap import METRICS, patient_bootstrap_summary, patient_seed_average


def _rows():
    rows = []
    for seed in (42, 52, 62):
        for model, offset in (("PRQ-Net", 0.0), ("BCR-Net", 0.1), ("CDEL", 0.2)):
            for subject, value in (("p1", 0.4), ("p2", 0.8)):
                rows.append({"held_out_center": "lzu", "model": model, "subject_id": subject, "center": "lzu", "outer_fold": 1, "training_seed": seed, **{metric: value + offset for metric in METRICS}})
    return pd.DataFrame(rows)


def test_seed_average_happens_before_patient_bootstrap_and_is_deterministic():
    averaged = patient_seed_average(_rows(), seeds=(42, 52, 62))
    assert len(averaged) == 6
    first = patient_bootstrap_summary(averaged, repeats=100, seed=42)
    second = patient_bootstrap_summary(averaged, repeats=100, seed=42)
    pd.testing.assert_frame_equal(first, second)
    row = first.loc[(first.model == "CDEL") & (first.metric == "patient_macro_f1")].iloc[0]
    assert row.n_unique_patients == 2
    assert row.resampling_unit == "patient_after_three_seed_mean"
    assert row.ci_low <= row.estimate <= row.ci_high


def test_incomplete_seed_or_duplicate_patient_is_rejected():
    frame = _rows().iloc[:-1]
    with pytest.raises(ValueError, match="once for every requested seed"):
        patient_seed_average(frame, seeds=(42, 52, 62))
    averaged = patient_seed_average(_rows(), seeds=(42, 52, 62))
    duplicate = pd.concat([averaged, averaged.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate patients"):
        patient_bootstrap_summary(duplicate, repeats=10)
