import numpy as np

from neuroez_c.task2.ngbr.negative_controls import permute_channel_map,permute_outcomes,permute_target


def test_permutations_preserve_distributions_and_counts():
    values=np.arange(10.);valid=np.ones(10,bool);target=np.array([1,1,1,0,0,0,0,0,0,0],bool);outcome=np.array([0,1]*5)
    assert np.array_equal(np.sort(permute_channel_map(values,valid,4)),values)
    assert permute_target(target,valid,4).sum()==target.sum()
    assert np.array_equal(np.sort(permute_outcomes(outcome,4)),np.sort(outcome))
