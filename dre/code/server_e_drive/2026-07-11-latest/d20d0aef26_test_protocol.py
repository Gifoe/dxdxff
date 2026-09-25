import numpy as np
from scripts.task1_aaai_cpu.run_all_cpu import f1
def test_ez_nez_direction():
    y=np.array([0,1]); p=np.array([0,1])
    assert f1(y,p,0)==1 and f1(y,p,1)==1
