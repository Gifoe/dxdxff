import unittest
import numpy as np
from neuroez_c.lzu_adapter_protocol import deterministic_lzu_split,non_lzu_invariance_audit
class LZUProtocolTests(unittest.TestCase):
 def test_split_is_deterministic_and_patientwise(self):
  self.assertEqual(deterministic_lzu_split(['a','b','c','d'],42),deterministic_lzu_split(['a','b','c','d'],42))
 def test_invariance_requires_bitwise_equality(self):
  x=np.array([1.,2.]);a=non_lzu_invariance_audit(x,x.copy(),x,x.copy(),np.array([1,0]),np.array([1,0]));self.assertTrue(a['passed'])
  self.assertFalse(non_lzu_invariance_audit(x,x+.01,x,x,np.array([1,0]),np.array([1,0]))['passed'])
