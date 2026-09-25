import numpy as np
from neuroez_c.task2.mosaic.virtual_intervention import virtual_intervention

def test_virtual_intervention_exact():
    x=np.array([[1.,1.,.2,.4]]); z=np.array([1,1,0,0],bool); q=np.array([.2,.2,.8,.5])
    out=virtual_intervention(x,z,q)
    assert np.isclose(out.capture[0],2/2.6) and np.isclose(out.outside_residual[0],.38)
    assert np.isclose(out.inside_outside_gap[0],.65) and out.outside_extent[0]==0
