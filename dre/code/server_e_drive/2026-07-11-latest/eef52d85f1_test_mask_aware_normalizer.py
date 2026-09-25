from __future__ import annotations

import numpy as np

from outcome_hifos.normalization import FoldNormalizer


def test_invalid_padding_never_changes_fold_statistics_or_output() -> None:
    valid = np.array([[True, False], [True, False]])
    clean = np.array([[[1.0], [0.0]], [[3.0], [0.0]]], dtype=np.float32)
    poisoned = clean.copy()
    poisoned[:, 1] = 99999.0

    first = FoldNormalizer().fit([clean], valid_masks=[valid], subject_ids=["train"])
    second = FoldNormalizer().fit([poisoned], valid_masks=[valid], subject_ids=["train"])
    np.testing.assert_array_equal(first.median, second.median)
    np.testing.assert_array_equal(first.scale, second.scale)
    transformed = second.transform(poisoned, valid_mask=valid)
    assert np.count_nonzero(transformed[~valid]) == 0


def test_patient_relative_uses_all_valid_runs_and_is_shift_invariant() -> None:
    runs = [
        np.array([[[1.0], [99999.0]], [[3.0], [99999.0]]], dtype=np.float32),
        np.array([[[5.0], [99999.0]]], dtype=np.float32),
    ]
    masks = [np.array([[True, False], [True, False]]), np.array([[True, False]])]
    normalizer = FoldNormalizer().fit(runs, valid_masks=masks, subject_ids=["train", "train"])
    original = normalizer.patient_relative_many(runs, masks)
    shifted = normalizer.patient_relative_many([run + 100.0 for run in runs], masks)
    for left, right, mask in zip(original, shifted, masks):
        np.testing.assert_allclose(left, right, atol=1e-5)
        assert np.count_nonzero(left[~mask]) == 0


def test_fit_subject_audit_excludes_outer_test() -> None:
    train = np.array([[[1.0]], [[2.0]]], dtype=np.float32)
    normalizer = FoldNormalizer().fit(
        [train],
        valid_masks=[np.ones(train.shape[:-1], dtype=bool)],
        subject_ids=["outer-train"],
    )
    assert normalizer.fit_subject_ids == ("outer-train",)
    assert "outer-test" not in normalizer.fit_subject_ids
