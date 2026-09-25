import numpy as np
import pytest
from neuroez_c.task2.cop.schema import COPSchemaError,adapt_feature_record,inspect_cache_schema


def _lookup():return {"p":{"channel_names":["A1","A2"],"clinical_target_mask":np.array([1,0],dtype=bool)}}


def test_feature_schema_validates_axes_and_times():
    record={"subject_id":"p","run_id":"s","channel_names_norm":["A1","A2"],"sample":{"window_features":np.ones((3,2,1)),"window_feature_names":["high_gamma"],"window_relative_centers_sec":np.array([-1.,1.,11.]),"feature_scale_used_secs":[2.,2.,2.]}}
    value,audit=adapt_feature_record(record,_lookup());assert value.feature_values.shape==(3,2,1) and audit["axis_transform"]=="W,C,F"


def test_missing_required_field_is_explicit():
    with pytest.raises(COPSchemaError,match="MISSING_REQUIRED_FIELD"):adapt_feature_record({"subject_id":"p"},_lookup())
