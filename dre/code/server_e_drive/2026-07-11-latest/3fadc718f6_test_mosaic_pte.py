import numpy as np
from neuroez_c.task2.mosaic.phase_transfer_entropy import phase_transfer_entropy_matrix

def test_pte_direction_and_diagonal():
    rng=np.random.default_rng(4); x=rng.integers(0,4,2000); y=np.r_[0,x[:-1]]; states=np.vstack([x,y])
    matrix=phase_transfer_entropy_matrix(states,1,bins=4)
    assert np.diag(matrix).sum()==0 and matrix[0,1]>matrix[1,0]
