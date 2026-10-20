"""Original source body/parity and changed switch/mask/gradient paths only."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest import mock
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
REPOSITORY_EXPERIMENTS = HERE.parent
original_spec = importlib.util.spec_from_file_location


def source_spec(name, path, *args, **kwargs):
    path = Path(path)
    # Local CPU tests resolve the SAME tracked parent files, not a reduced model.
    # On the server the exact original D-drive source files are used unchanged.
    if not path.is_file() and path.parent.name in {
            'm8_6d_no_interaction_ablation_seed42_v1',
            'm8_6d_no_variance_ablation_baselines_seed42_v1',
            'm8_7d_f0_complete_ablation_seed42_v1'}:
        path = REPOSITORY_EXPERIMENTS/path.parent.name/path.name
    return original_spec(name, path, *args, **kwargs)


with mock.patch.object(importlib.util, 'spec_from_file_location', side_effect=source_spec):
    spec = original_spec('test_relative_f0_math', HERE/'model.py')
    MODEL = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = MODEL
    spec.loader.exec_module(MODEL)
film_path = Path(r'D:\m8_7d_raw_film_seed42_v1\model.py')
if not film_path.is_file():
    film_path = REPOSITORY_EXPERIMENTS/'m8_7d_raw_film_seed42_v1/model.py'
film_spec = original_spec('test_original_film_math', film_path)
FILM = importlib.util.module_from_spec(film_spec)
sys.modules[film_spec.name] = FILM
film_spec.loader.exec_module(FILM)


def transform(delta, mask, times, mode, perms):
    return delta.clone()


def stub():
    pass


class Gate(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(7, 32), nn.GELU(), nn.Linear(32, 24))

    def forward(self, value):
        return 1+.25*torch.tanh(self.network(value))


def source():
    torch.manual_seed(51)
    cnn = nn.Sequential(nn.Conv1d(1, 12, 9, 4), nn.GELU(), nn.Conv1d(12, 24, 7, 4),
                        nn.GELU(), nn.AdaptiveAvgPool1d(1))
    result = FILM.SingleHead('F0_AFTER_GELU', cnn, Gate(), stub, 170054)
    # Nonzero CONDITIONING fixture tests sensitivity, not a trained initialization.
    with torch.no_grad():
        result.cnn.conditioner[2].weight.normal_(0, .02)
    return result


def batch():
    torch.manual_seed(63)
    raw = torch.randn(2, 6, 9, 500)
    mask = torch.ones(2, 6, 9, dtype=torch.bool)
    times = torch.tensor([[-4, -2, 0, 2, 4, 6, 8, 12, 18]]*2, dtype=torch.float32)
    return torch.randn(6, 14), raw, mask, times, torch.randn(2, 6, 9, 7), torch.ones(7)


class ModelTests(unittest.TestCase):
    def test_original_reference_and_noPR_global_math_and_gradients_exact(self):
        initial = source()
        args = batch()
        for arm, old_arm in [('B0_ORIGINAL', 'FULL_F0'), ('WO_BOTH_PR', 'WO_BOTH_PR')]:
            original = MODEL.BASE.Ablation(initial, old_arm)
            revised = MODEL.CompleteF0(initial, arm)
            self.assertEqual(set(original.state_dict()), set(revised.state_dict()))
            for name in original.state_dict():
                self.assertTrue(torch.equal(original.state_dict()[name], revised.state_dict()[name]), name)
            a, b = original(*args)['logit'], revised(*args)['logit']
            self.assertTrue(torch.equal(a, b), arm)
            y = torch.tensor([0., 1., 0., 1., 0., 1.])
            la = MODEL.objective(a, y, torch.tensor(1.))
            lb = MODEL.objective(b, y, torch.tensor(1.))
            self.assertTrue(torch.equal(la, lb))
            la.backward()
            lb.backward()
            expected = dict(original.named_parameters())
            for name, parameter in revised.named_parameters():
                self.assertTrue(torch.equal(expected[name].grad, parameter.grad), name)

    def test_all_variants_match_shared_initialization_and_ignore_Variance(self):
        initial = source()
        args = batch()
        for arm in MODEL.ARMS:
            rng = torch.random.get_rng_state().clone()
            model = MODEL.CompleteF0(initial, arm)
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()), arm)
            self.assertEqual(sum(p.numel() for p in model.parameters()), MODEL.PARAMETERS[arm])
            self.assertFalse(model.interaction)
            self.assertEqual(sum(isinstance(layer, nn.Linear) and layer.out_features == 1
                                 for layer in model.modules()), 1)
            expected = initial.head[0].weight[:, :96] if model.branch == 'raw' else initial.head[0].weight[:, 96:] if model.branch == 'phys' else initial.head[0].weight
            self.assertTrue(torch.equal(model.head[0].weight, expected))
            self.assertTrue(torch.equal(model.head[2].weight, initial.head[2].weight))
            use = list(args)
            if arm == 'EARLY_FILM_MISMATCH':
                perm, info = MODEL.channel_mapping(args[2], list('abcdef'), 'fixture', 1)
                use.append(perm)
            a = model(*use)['logit']
            changed = [value.clone() for value in use]
            changed[0][:, 8:10] = float('nan')
            changed[4][..., 4] = float('nan')
            if arm in MODEL.DROP:
                j = MODEL.DROP[arm]
                changed[0][:, 2*j:2*j+2] = float('nan')
                changed[4][..., j] = float('nan')
            self.assertTrue(torch.equal(a, model(*changed)['logit']), arm)
            model.zero_grad(set_to_none=True)
            MODEL.objective(a, torch.tensor([0., 1., 0., 1., 0., 1.]), torch.tensor(1.)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                                for p in model.parameters()), arm)
            self.assertGreater(float(model.head[2].weight.grad.norm()), 0.)

    def test_separate_PR_guidance_and_reference_boundaries(self):
        initial = source()
        args = batch()
        full = MODEL.CompleteF0(initial, 'FULL')
        no_raw = MODEL.CompleteF0(initial, 'WO_RAW_PR')
        raw = no_raw(*args)['raw_summary'].double()
        sd = raw.std(0, unbiased=False)
        normalized = ((raw-raw.mean(0))/torch.where(sd > 1e-8, sd, torch.ones_like(sd))).float()
        self.assertTrue(torch.allclose(full(*args)['raw_summary'], normalized, rtol=1e-5, atol=1e-6))
        no_feature = MODEL.CompleteF0(initial, 'WO_FEATURE_PR')
        self.assertFalse(no_feature.feature_pr)
        self.assertFalse(no_feature.guidance_pr)
        self.assertTrue(no_feature.raw_pr)
        both = MODEL.CompleteF0(initial, 'WO_BOTH_PR')
        self.assertFalse(both.feature_pr or both.guidance_pr or both.raw_pr)
        window = MODEL.CompleteF0(initial, 'WO_WINDOW_PR')
        self.assertTrue(window.feature_pr and window.raw_pr)
        self.assertFalse(window.guidance_pr)
        early = MODEL.CompleteF0(initial, 'WO_EARLY_FILM')
        late = MODEL.CompleteF0(initial, 'WO_LATE_GATE')
        self.assertFalse(early.early)
        self.assertTrue(early.late and early.guidance_pr)
        self.assertTrue(late.early and late.guidance_pr)
        self.assertFalse(late.late)
        no_pre = MODEL.CompleteF0(initial, 'WO_PRE_REFERENCE')
        self.assertFalse(no_pre.pre_reference)
        self.assertTrue(no_pre.raw_pr and no_pre.guidance_pr)

    def test_pure_branches_and_mismatch_support(self):
        initial, args = source(), batch()
        raw = MODEL.CompleteF0(initial, 'RAW_ONLY')
        expected = raw(*args)['logit']
        changed = [v.clone() for v in args]
        changed[0][:] = float('nan'); changed[4][:] = float('nan')
        self.assertTrue(torch.equal(expected, raw(*changed)['logit']))
        phys = MODEL.CompleteF0(initial, 'PHYSIOLOGY_ONLY')
        self.assertTrue(torch.equal(phys(*args)['logit'], phys(args[0])['logit']))
        self.assertEqual(phys.head[0].in_features, 14)
        both = MODEL.CompleteF0(initial, 'WO_BOTH_GUIDANCE')
        self.assertFalse(both.early or both.late or both.guidance_pr)
        self.assertTrue(both.feature_pr and both.raw_pr)
        model = MODEL.CompleteF0(initial, 'EARLY_FILM_MISMATCH')
        perm, info = MODEL.channel_mapping(args[2], list('abcdef'), 'fixture', 1)
        again, _ = MODEL.channel_mapping(args[2], list('abcdef'), 'fixture', 1)
        self.assertTrue(torch.equal(perm, again))
        self.assertEqual(info['changed_valid_windows'], info['eligible_valid_windows'])
        self.assertEqual(info['total_valid_windows'], int(args[2].sum()))
        self.assertEqual(sum(p.numel() for p in model.parameters()), MODEL.PARAMETERS['FULL'])
        self.assertFalse(torch.equal(model(*args, perm)['logit'], model(*args, torch.arange(6).repeat(2,1))['logit']))


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
