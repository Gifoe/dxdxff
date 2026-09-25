import numpy as np
from neuroez_c.task2.mosaic.schema import MosaicRunRecord
from neuroez_c.task2.mosaic.fragility_trajectory import FragilityConfig,compute_fragility_trajectory

def test_fragility_trajectory_deterministic():
    rng=np.random.default_rng(2); signal=rng.normal(size=(4,500)); record=MosaicRunRecord("p","lzu","s",signal,25.,250,["A","B","C","D"],np.ones(4,bool),np.array([1,1,0,0],bool))
    cfg=FragilityConfig(angle_count=8); a,_,_=compute_fragility_trajectory(record,cfg); b,_,_=compute_fragility_trajectory(record,cfg)
    assert set(a)=={"preictal","onset","spread"}
    for phase in a: assert np.allclose(a[phase].values,b[phase].values,equal_nan=True) and a[phase].values.shape[1]==4
