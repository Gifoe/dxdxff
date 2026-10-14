import numpy as np
from calibrate import apply_map,fit_map

class Core:
    def select_threshold(self,y,s,p):return {'threshold':.3}

def test_known_source_minimum_and_unknown_fallback():
    d=dict(center=np.array(['a','b','a','a','a','b','new']),patient=np.array(['t1','t2','p1','p2','p3','p4','p5']),train=np.array([0,1]),val=np.array([2,3,4,5,6]),y=np.zeros(7))
    mapping,counts=fit_map(Core(),d,np.ones(5)*.4,.5,'SOURCE_MIN3')
    assert mapping=={'a':.3} and counts=={'a':3}
    assert np.array_equal(apply_map(np.array(['a','b','new']),.5,mapping),[.3,.5,.5])

def test_shrinkage_fixed_before_labels_and_no_patient_input():
    d=dict(center=np.array(['a','a']),patient=np.array(['t','v']),train=np.array([0]),val=np.array([1]),y=np.zeros(2))
    mapping,counts=fit_map(Core(),d,np.array([.4]),.5,'SOURCE_SHRINK5')
    assert np.isclose(mapping['a'],(.3+5*.5)/6)
