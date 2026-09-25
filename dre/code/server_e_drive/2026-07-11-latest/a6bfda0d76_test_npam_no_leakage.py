import pytest

from neuroez_c.task2.p2_q10_npam_model import P2Q10NPAMModel
from neuroez_c.task2.outcomes import normalize_outcome, resolve_cache_outcomes
from neuroez_c.task2.protocol import ProtocolError, assert_checkpoint_safe


def test_checkpoint_subject_overlap_is_fatal():
    with pytest.raises(ProtocolError):
        assert_checkpoint_safe({"A", "B"}, {"B", "C"})
    assert_checkpoint_safe({"A"}, {"B"})


def test_outcome_and_task1_label_semantics_are_not_inverted():
    assert normalize_outcome(1, field="outcome_label_failure") == "failure"
    assert normalize_outcome(0, field="outcome_label_failure") == "success"
    assert normalize_outcome(True, field="surgery_success") == "success"
    assert normalize_outcome("Engel III", field="engel_score") == "failure"
    cache = {
        "patient_index": {
            "x": {"outcome_group": "success"},
            "y": {"engel_score": 3},
        },
        "run_records": [],
    }
    outcomes, _ = resolve_cache_outcomes(cache)
    labels = outcomes.set_index("patient_key")["outcome_label"].to_dict()
    assert labels == {"x": 1, "y": 0}


@pytest.mark.parametrize("field", ["coordinates", "soz", "resect", "true_k", "center", "outcome_label"])
def test_forbidden_model_inputs_are_rejected(field):
    with pytest.raises(ValueError):
        P2Q10NPAMModel.assert_label_blind_inputs({field: 1})
