import numpy as np
import pytest
from neuroez_c.evidence_views import physics_state_features
from task1_clean_nez.dataset import PHYSICS_FEATURE_NAMES,feature_args,validate_feature_names

def test_s5_selection_is_exactly_12_and_ordered():
    names=["junk",*PHYSICS_FEATURE_NAMES]; x=np.arange(2*3*len(names),dtype=np.float32).reshape(2,3,-1); a=feature_args(window_feature_names=names)
    y=physics_state_features(x,np.array([-1,1]),a)
    assert y.shape[-1]==12 and tuple(physics_state_features.selected_feature_names)==PHYSICS_FEATURE_NAMES
def test_missing_s5_feature_fails_fast():
    with pytest.raises(ValueError,match="missing=.*hfo80_150_max_envelope_z"):
        validate_feature_names([{"window_feature_names":list(PHYSICS_FEATURE_NAMES[:-1])}])
