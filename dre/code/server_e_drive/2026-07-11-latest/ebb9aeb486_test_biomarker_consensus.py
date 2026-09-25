import numpy as np

from neuroez_c.task2.ngbr.biomarker_consensus import consensus_map


def test_consensus_requires_two_and_uses_top_two():
    consensus,count,agreement=consensus_map(np.array([.9,.9]),np.array([np.nan,.8]),np.array([np.nan,.1]),np.array([np.nan,np.nan]))
    assert np.isnan(consensus[0]) and count[0]==1
    assert np.isclose(consensus[1],np.sqrt(.9*.8)) and agreement[1]>0
