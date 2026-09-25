import unittest
import torch
from neuroez_c.p2_q10_lzu_adapter import P2Q10LZUBoundedAdapter, apply_p2_q10_lzu_adapter


class AdapterTests(unittest.TestCase):
    def test_zero_initialization_is_exact_identity(self):
        model = P2Q10LZUBoundedAdapter(9)
        base = torch.tensor([-.4, .2]); delta = model(torch.randn(2, 9), torch.tensor([True, True]))
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        self.assertTrue(torch.equal(apply_p2_q10_lzu_adapter(base, delta, torch.tensor([True, True])), base))

    def test_bounds_sign_and_external_gate(self):
        model = P2Q10LZUBoundedAdapter(3, max_delta=.15)
        with torch.no_grad(): model.net[-1].bias.fill_(50)
        delta = model(torch.randn(2, 3), torch.tensor([True, False]))
        self.assertGreater(float(delta[0].detach()), 0); self.assertEqual(float(delta[1].detach()), 0.0); self.assertLessEqual(float(delta.abs().max().detach()), .150001)
        with torch.no_grad(): model.net[-1].bias.fill_(-50)
        self.assertLess(float(model(torch.randn(1, 3), torch.tensor([True]))[0].detach()), 0)

    def test_center_is_not_an_input(self):
        self.assertEqual(P2Q10LZUBoundedAdapter(11).net[1].in_features, 11)
