import numpy as np
import pandas as pd
from neuroez_c.task2.cop.nested_elastic_net import deterministic_inner_splits,inner_search,select_inner_threshold


def test_nested_search_is_deterministic_and_threshold_inner_only():
    rng=np.random.default_rng(3);y=np.array([0,1]*15);frame=pd.DataFrame({"feature__x":y+rng.normal(0,.3,30),"feature__z":rng.normal(size=30)});centers=["a","b","c"]*10
    a=inner_search(frame,list(frame),y,centers,outer_fold=1,inner_folds=3,seed=42,quick=True);b=inner_search(frame,list(frame),y,centers,outer_fold=1,inner_folds=3,seed=42,quick=True)
    assert a.C==b.C and a.l1_ratio==b.l1_ratio and np.allclose(a.probability,b.probability) and .2<=a.threshold<=.8
