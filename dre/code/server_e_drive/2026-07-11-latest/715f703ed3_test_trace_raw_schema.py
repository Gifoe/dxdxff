import numpy as np
from neuroez_c.task2.trace_rawmil.preprocessing import preprocess_window
def test_preprocess_shape_and_invalid():
 x,ok=preprocess_window(np.random.default_rng(1).normal(size=500),250);assert ok and x.shape==(500,)
 _,bad=preprocess_window(np.zeros(500),250);assert not bad
