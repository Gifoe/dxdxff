import numpy as np
import pytest
from neuroez_c.task2.cop.schema import COPSchemaError,FeatureRunRecord,RawRunRecord,align_feature_raw


def test_alignment_reports_channel_match_and_strict_failure():
    f=FeatureRunRecord("p","c","s",["A1","A2"],np.array([0]),np.array([1]),np.ones((1,2,1)),["gamma"],np.ones((1,2),dtype=bool),np.array([1,0],dtype=bool));r=RawRunRecord("p","c","s",np.ones((2,5)),250.,2,["A1","A2"],np.ones(2,dtype=bool),np.array([1,0],dtype=bool))
    assert align_feature_raw([f],[r],strict=True)[0]["match_rate"]==1
    bad=RawRunRecord("p","c","s",np.ones((1,5)),250.,2,["X"],np.ones(1,dtype=bool),np.array([1],dtype=bool))
    with pytest.raises(COPSchemaError):align_feature_raw([f],[bad],strict=True)
