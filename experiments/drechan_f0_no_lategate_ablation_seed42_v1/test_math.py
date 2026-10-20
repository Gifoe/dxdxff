"""Changed arithmetic only; no benchmark accuracy or fabricated outcomes."""
import unittest
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'drechan_f0_complete_ablation_3seed_v1'))
import numpy as np
import torch
from relative_guidance import conditioning_delta
from feature_pr import prepare, relative, query
sys.path.insert(0, str(Path(__file__).resolve().parent))


class MathTests(unittest.TestCase):
    def test_window_formula_valid_support_and_variance_mask(self):
        torch.manual_seed(319)
        delta = torch.randn(2, 5, 4, 7)
        mask = torch.ones(2, 5, 4, dtype=torch.bool)
        mask[0, 0, 1] = False
        scale = torch.arange(1, 8).float()
        keep = torch.tensor([1, 1, 1, 1, 0, 1, 1.])
        result, audit = conditioning_delta(delta, mask, scale, keep, True)
        for s in range(2):
            for w in range(4):
                selected = mask[s, :, w]
                for j in [0, 1, 2, 3, 5, 6]:
                    g = delta[s, selected, w, j].double()/scale[j]
                    q = (g-g.mean())/g.std(unbiased=False)
                    expected = .5*(g+q)
                    actual = result[s, selected, w, j].double()/scale[j]
                    self.assertTrue(torch.allclose(actual, expected, rtol=1e-6, atol=1e-7))
        changed = delta.clone()
        changed[~mask] = float('nan')
        changed[..., 4] = float('nan')
        again, _ = conditioning_delta(changed, mask, scale, keep)
        self.assertTrue(torch.equal(result, again))
        self.assertTrue(torch.equal(result[..., 4], torch.zeros_like(result[..., 4])))
        self.assertEqual(audit['relative_used_coordinates'], int(mask.sum())*6)

    def test_units_and_unsupported_coordinates(self):
        torch.manual_seed(531)
        delta = torch.randn(1, 4, 4, 7)
        mask = torch.ones(1, 4, 4, dtype=torch.bool)
        mask[:, 1:, 0] = False
        delta[:, :, 1] = 2.
        scale = torch.arange(1, 8).float()
        keep = torch.tensor([1, 1, 1, 1, 0, 1, 1.])
        original, audit = conditioning_delta(delta, mask, scale, keep, True)
        factor = torch.tensor([1e-3, 3., 7., 100., 1., 1e-2, 9.])
        changed, _ = conditioning_delta(delta*factor, mask, scale*factor, keep)
        self.assertTrue(torch.allclose(original/scale, changed/(scale*factor), rtol=2e-6, atol=2e-7))
        masked = torch.where(keep.bool(), delta, 0.)
        self.assertTrue(torch.equal(original[:, :, 1], masked[:, :, 1]))
        self.assertTrue(torch.equal(original[:, 0, 0], masked[:, 0, 0]))
        self.assertEqual(audit['singleton_valid_windows'], 1)
        self.assertGreater(audit['global_fallback_coordinates'], 0)

    def test_direct_fit_only_units_and_query(self):
        rng = np.random.default_rng(447)
        raw = rng.normal(size=(36, 14))*np.logspace(-9, 3, 14)
        patient = np.repeat(np.arange(6).astype(str), 6)
        fit = np.arange(24)
        value, pre = prepare(raw, patient, fit)
        changed = raw.copy()
        changed[24:] *= 123.
        _, second = prepare(changed, patient, fit)
        for key in pre:
            if key == 'fit_patient_ids':
                self.assertEqual(pre[key], second[key])
            else:
                self.assertTrue(np.array_equal(pre[key], second[key]), key)
        self.assertTrue(np.array_equal(query(raw, patient, pre), value))
        factor = np.logspace(-3, 3, 14)
        scaled, _ = prepare(raw*factor, patient, fit)
        self.assertTrue(np.allclose(value, scaled, rtol=1e-6, atol=1e-7))

    def test_direct_degenerate_missing_and_nondegenerate_definition(self):
        rng = np.random.default_rng(83)
        raw = rng.normal(size=(12, 14))
        patient = np.repeat(['a', 'b'], 6)
        raw[:6, 0] = 3.
        raw[0, 1] = np.nan
        value = relative(raw, patient, np.ones(14))
        self.assertTrue(np.array_equal(value[:6, 0], np.zeros(6)))
        self.assertTrue(np.isnan(value[0, 1]))
        expected = (raw[6:]-raw[6:].mean(0))/raw[6:].std(0)
        self.assertTrue(np.array_equal(value[6:], expected))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
