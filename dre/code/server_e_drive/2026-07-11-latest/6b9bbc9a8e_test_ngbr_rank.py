import numpy as np

from neuroez_c.task2.ngbr.robust_rank import percentile_rank_channels


def test_rank_ties_nan_and_direction():
    values=np.array([1.,2.,2.,4.,np.nan]);valid=np.array([1,1,1,1,1],bool)
    high=percentile_rank_channels(values,valid,high_is_abnormal=True);low=percentile_rank_channels(values,valid,high_is_abnormal=False)
    assert high[1]==high[2]
    assert np.allclose(high[:4]+low[:4],1)
    assert np.isnan(high[4]) and np.nanmin(high)>=0 and np.nanmax(high)<=1


def test_rank_invalid_with_fewer_than_four_channels():
    assert np.isnan(percentile_rank_channels(np.arange(3.),np.ones(3,bool),high_is_abnormal=True)).all()
