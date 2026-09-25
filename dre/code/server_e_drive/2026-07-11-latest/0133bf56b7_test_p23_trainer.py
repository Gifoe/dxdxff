import numpy as np
from types import SimpleNamespace

from neuroez_c.p23_trainer import (
    _flatten,
    _counterfactual_records, _direct_outer_resume_path, _load_direct_outer_resume,
    _inner_resume_path, _load_inner_resume,
    _patient_oracle_macro_f1, _save_inner_resume, _select_global_threshold,
    _save_direct_outer_resume,
)


def _record(scores=(0.8, 0.2), labels=(1, 0)):
    return {
        "subject_id": "p1", "center": "hup", "channel_mask": np.array([True, True]),
        "labels_nez": np.asarray(labels), "score_nez": np.asarray(scores, dtype=np.float32),
        "score_ez": 1.0 - np.asarray(scores, dtype=np.float32),
        "direct_nez_logit": np.array([1.0, -1.0], dtype=np.float32),
        "u_anchor": np.array([0.4, -0.4], dtype=np.float32),
        "u_seizure": np.zeros(2, dtype=np.float32), "u_causal": np.zeros(2, dtype=np.float32),
        "w_anchor": np.array([0.05, 0.05], dtype=np.float32),
        "w_seizure": np.array([0.05, 0.05], dtype=np.float32), "w_causal": np.zeros(2, dtype=np.float32),
        "delta": np.array([0.02, -0.02], dtype=np.float32), "predicted_patient_threshold": 0.5,
    }


def test_threshold_uses_only_provided_oof_records_and_oracle_is_diagnostic_helper():
    threshold, audit = _select_global_threshold([_record()])
    assert 0.2 < threshold < 0.8
    assert audit["inner_oof_macro_f1"] == 1.0
    assert _patient_oracle_macro_f1([_record()]) == 1.0


def test_direct_outer_only_threshold_audit_is_labeled_as_validation_not_inner_oof():
    threshold, audit = _select_global_threshold([_record()], audit_prefix="validation")
    assert 0.2 < threshold < 0.8
    assert audit["validation_macro_f1"] == 1.0
    assert "inner_oof_macro_f1" not in audit


def test_score_level_counterfactual_keeps_saved_threshold_and_never_uses_true_count():
    result = _counterfactual_records([_record()], "C2_NO_ANCHOR")[0]
    assert result["classification_threshold"] == 0.5
    assert result["decision_rule"] == "fixed_nez_probability_threshold"
    assert "true_count" not in result
    assert np.array_equal(result["predicted_nez_mask"], result["score_nez"] >= 0.5)


def test_flatten_reduces_seizure_by_channel_temporal_diagnostics_to_channel_scalars():
    record = _record()
    record.update({
        "canonical_channels": ["A", "B"],
        "predicted_nez_mask": np.array([True, False]),
        "predicted_ez_mask": np.array([False, True]),
        "valid_pre": np.array([[True, False], [False, True]]),
        "temporal_gate": np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32),
    })
    channels, _ = _flatten([record], fold=1)
    assert channels["valid_pre"].tolist() == [0.5, 0.5]
    assert np.allclose(channels["temporal_gate"].to_numpy(), [0.4, 0.6])


def test_completed_inner_fold_can_be_restored_only_for_the_same_subject_ledger(tmp_path):
    args = SimpleNamespace(p23_profile="P5_FULL", model_seed=42)
    path = _inner_resume_path(tmp_path, 1, 2)
    _save_inner_resume(
        path, args=args, outer_fold=1, inner_fold=2, fit_subjects=["p2"], heldout_subjects=["p1"],
        best_epoch=7, history=[{"epoch": 7}], prototype_audit={"source": "fit_only"}, records=[_record()],
    )
    restored = _load_inner_resume(
        path, args=args, outer_fold=1, inner_fold=2, fit_subjects=["p2"], heldout_subjects=["p1"],
    )
    assert restored is not None
    assert restored["best_epoch"] == 7


def test_resume_rejects_a_different_subject_ledger(tmp_path):
    args = SimpleNamespace(p23_profile="P5_FULL", model_seed=42)
    path = _inner_resume_path(tmp_path, 1, 1)
    _save_inner_resume(
        path, args=args, outer_fold=1, inner_fold=1, fit_subjects=["p2"], heldout_subjects=["p1"],
        best_epoch=1, history=[], prototype_audit={}, records=[_record()],
    )
    try:
        _load_inner_resume(path, args=args, outer_fold=1, inner_fold=1, fit_subjects=["p2"], heldout_subjects=["other"])
    except RuntimeError as error:
        assert "does not match" in str(error)
    else:
        raise AssertionError("Expected incompatible resume artifact to fail")


def test_completed_direct_outer_fold_can_be_resumed(tmp_path):
    args = SimpleNamespace(
        p23_profile="P5_FULL", model_seed=42, batch_size=4,
        patient_batch_size=4, epochs=30, patience=6,
    )
    path = _direct_outer_resume_path(tmp_path, 1)
    record = _record()
    _save_direct_outer_resume(
        path, args=args, outer_fold=1, outer_train=["p2"], outer_test=["p1"],
        records=[record], no_temporal_records=[record], summary={"outer_fold": 1},
        history=[{"outer_fold": 1}], stage_rows=[{"outer_fold": 1}],
    )
    restored = _load_direct_outer_resume(
        path, args=args, outer_fold=1, outer_train=["p2"], outer_test=["p1"],
    )
    assert restored is not None
    assert restored["summary"]["outer_fold"] == 1
