"""Run on the trusted server with pinned original dependencies available."""
import sys
from pathlib import Path
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from support import load,ROOT,BASE,CompactDelta
P=load('progressive_test_pre',ROOT/'code/preprocessing.py')
from construct_delta import window_delta,aggregate

class Contract(unittest.TestCase):
    def test_no_pr_and_fit_only(self):
        x=np.arange(84,dtype=float).reshape(6,14);patient=np.array(['a','a','b','b','c','c']);tr=np.array([0,1,2,3])
        y,pre=P.prepare(x,patient,tr);xx=x.copy();xx[4:]+=100000
        yy,pp=P.prepare(xx,patient,tr)
        for k in ['mean','scale','var','imputer_statistics']:np.testing.assert_array_equal(pre[k],pp[k])
        np.testing.assert_array_equal(y[tr],yy[tr]);self.assertEqual(pre['fit_patient_ids'],['a','b'])
        self.assertFalse(pre['patient_z']);self.assertGreater(float(np.abs(yy[4:]).max()),100)
    def test_missing_and_single_seizure(self):
        a=np.array([[1,2],[3,4]],np.float32);b=np.array([[5,6],[np.nan,np.nan]],np.float32)
        v,n=aggregate([a,b]);np.testing.assert_array_equal(v,[[3,2,4,2],[3,0,4,0]])
        np.testing.assert_array_equal(n,[[2,2],[1,1]])
    def test_all_genuine_pre(self):
        x=np.array([[[1.,3.]],[[3.,5.]],[[7.,9.]],[[999.,999.]]],np.float32)
        d,p,n=window_delta(x,np.array([-20.,-1.,2.,5.]),np.array([[1],[1],[1],[0]],bool))
        np.testing.assert_array_equal(p,[[2,4]]);self.assertEqual(n.tolist(),[2]);self.assertTrue(np.isnan(d[-1]).all())
        np.testing.assert_array_equal(np.nanmean(d,axis=0),np.array([[5/3,5/3]],np.float32))

if __name__=='__main__':unittest.main()
