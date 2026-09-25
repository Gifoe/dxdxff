import numpy as np
from neuroez_c.task2.trace_rawmil.schema import masks
def test_nez_one_ez_zero():
 ez,nez,valid=masks([0,1,np.nan,2]);assert ez.tolist()==[True,False,False,False] and nez.tolist()==[False,True,False,False] and valid.tolist()==[True,True,False,False]
