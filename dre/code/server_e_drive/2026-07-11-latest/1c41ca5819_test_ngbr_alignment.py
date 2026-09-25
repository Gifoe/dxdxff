import numpy as np
import pytest

from neuroez_c.task2.ngbr.channel_alignment import align_p2_to_raw,audit_raw_feature_alignment
from neuroez_c.task2.ngbr.schema import FeatureRunRecord,P2NEZRecord,RawRunRecord


def _raw(names):return RawRunRecord("p","c","s",np.ones((len(names),20)),250,10,names,np.ones(len(names),bool),np.array([1]+[0]*(len(names)-1),bool),0,.04)


def test_channel_normalization_and_p2_reorder():
    raw=_raw(["A01","A02","A03","A04"]);p2=P2NEZRecord("p","c",["A4","A3","A2","A1"],np.array([.4,.3,.2,.1]),None,np.ones(4,bool),np.array([0,0,0,1],bool))
    aligned=align_p2_to_raw(p2,raw,strict=True)
    assert np.allclose(aligned.final_nez_probability,[.1,.2,.3,.4])


def test_raw_feature_low_match_fails_with_ids():
    raw=_raw(["A1","A2","A3","A4"]);feature=FeatureRunRecord("p","c","s",np.ones((2,4,1)),["x"],np.array([0,1]),["A1","B2","B3","B4"],np.ones((2,4),bool),np.array([1,0,0,0],bool))
    with pytest.raises(ValueError,match="patient=p, seizure=s"):audit_raw_feature_alignment([feature],[raw],strict=True)
