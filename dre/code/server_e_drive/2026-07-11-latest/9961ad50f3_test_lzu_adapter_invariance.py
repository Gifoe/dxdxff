import unittest,numpy as np
from neuroez_c.lzu_adapter_protocol import non_lzu_invariance_audit
class LZUInvarianceTests(unittest.TestCase):
 def test_bitwise_requirement(self):
  x=np.array([.2,.3]);self.assertTrue(non_lzu_invariance_audit(x,x,x,x,np.array([0,1]),np.array([0,1]))['passed'])
