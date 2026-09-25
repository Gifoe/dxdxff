import numpy as np
from neuroez_c.task2.mosaic.cii_expert import cii_from_matrices

def test_cii_and_escape_definition():
    m=np.zeros((1,3,3)); m[0,0,2]=.7; m[0,2,0]=.1; m[0,1,2]=.2
    cii,flow=cii_from_matrices(m,np.array([1,1,0],bool))
    assert np.isclose(cii[0,0],.6) and np.isclose(flow["net_causal_escape"][0],.8)
