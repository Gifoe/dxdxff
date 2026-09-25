from __future__ import annotations

import numpy as np

from outcome_hifos.normalization import FoldNormalizer


def test_fit_records_only_explicit_training_subjects() -> None:
    arrays = [np.asarray([[[1.0, 10.0]], [[3.0, 30.0]]], dtype=np.float32)]
    masks = [np.ones(arrays[0].shape[:-1], dtype=bool)]
    normalizer = FoldNormalizer().fit(arrays, valid_masks=masks, subject_ids=["train-1"])
    transformed = normalizer.transform(arrays[0], valid_mask=masks[0])
    assert normalizer.fit_subject_ids == ("train-1",)
    assert "outer-test-1" not in normalizer.fit_subject_ids
    assert np.isfinite(transformed).all()


def test_patient_relative_view_is_shift_invariant() -> None:
    array = np.asarray([[[1.0], [3.0]], [[5.0], [7.0]]], dtype=np.float32)
    shifted = array + 100.0
    mask = np.ones(array.shape[:-1], dtype=bool)
    normalizer = FoldNormalizer().fit([array], valid_masks=[mask], subject_ids=["train-1"])
    np.testing.assert_allclose(
        normalizer.patient_relative(array, mask),
        normalizer.patient_relative(shifted, mask),
        atol=1e-5,
    )
