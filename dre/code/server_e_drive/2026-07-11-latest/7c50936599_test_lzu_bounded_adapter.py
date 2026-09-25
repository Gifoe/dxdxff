import unittest
import torch
from neuroez_c.lzu_bounded_adapter import LZUBoundedResidualAdapter,apply_lzu_adapter
class LZUAdapterTests(unittest.TestCase):
 def test_initial_delta_and_hard_gate(self):
  m=LZUBoundedResidualAdapter(5,max_delta=.15);x=torch.randn(3,5);d=m(x,torch.tensor([True,False,True]));self.assertTrue(torch.equal(d,torch.zeros_like(d)))
 def test_bounds_and_non_lzu_exact(self):
  m=LZUBoundedResidualAdapter(2,max_delta=.15)
  with torch.no_grad():m.net[-1].bias.fill_(100)
  d=m(torch.randn(2,2),torch.tensor([True,False]));self.assertLessEqual(float(d.abs().max().detach()),.150001);base=torch.tensor([.2,.3]);self.assertTrue(torch.equal(apply_lzu_adapter(base,d,torch.tensor([True,False]))[1],base[1]))
