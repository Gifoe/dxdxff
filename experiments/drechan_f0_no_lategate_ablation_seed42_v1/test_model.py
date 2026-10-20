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
    def test_all_six_reuse_graphs_forward_loss_gradient_and_state_exact(self):
        initial,args=source(),batch()
        for arm,original in MODEL.REUSE.items():
            old=MODEL.BASE.CompleteF0(initial,original);new=MODEL.NoLateF0(initial,arm)
            self.assertEqual(set(old.state_dict()),set(new.state_dict()))
            for name in old.state_dict():self.assertTrue(torch.equal(old.state_dict()[name],new.state_dict()[name]),(arm,name))
            a,b=old(*args)['logit'],new(*args)['logit'];self.assertTrue(torch.equal(a,b),arm)
            y=torch.tensor([0.,1.,0.,1.,0.,1.]);la=MODEL.objective(a,y,torch.tensor(1.));lb=MODEL.objective(b,y,torch.tensor(1.))
            self.assertTrue(torch.equal(la,lb));la.backward();lb.backward()
            expected=dict(old.named_parameters())
            for name,p in new.named_parameters():self.assertTrue(torch.equal(expected[name].grad,p.grad),(arm,name))

    def test_all_variants_no_late_shared_init_masks_and_gradients(self):
        initial,args=source(),batch()
        for arm in MODEL.ARMS:
            rng=torch.random.get_rng_state().clone();m=MODEL.NoLateF0(initial,arm)
            self.assertTrue(torch.equal(rng,torch.random.get_rng_state()),arm)
            self.assertEqual(sum(p.numel() for p in m.parameters()),MODEL.PARAMETERS[arm])
            if arm not in MODEL.REFERENCES:self.assertFalse(m.late or hasattr(m,'gate'))
            expected=initial.head[0].weight[:,:96] if m.branch=='raw' else initial.head[0].weight[:,96:] if m.branch=='phys' else initial.head[0].weight
            self.assertTrue(torch.equal(m.head[0].weight,expected))
            self.assertEqual(m.head[0].out_features,64);self.assertFalse(m.interaction)
            use=list(args)
            if arm=='EARLY_FILM_MISMATCH':
                perm,_=MODEL.channel_mapping(args[2],list('abcdef'),'fixture',1);use.append(perm)
            out=m(*use)['logit'];change=[x.clone() for x in use]
            change[0][:,8:10]=float('nan');change[4][...,4]=float('nan')
            if arm in MODEL.DROP:
                j=MODEL.DROP[arm];change[0][:,2*j:2*j+2]=float('nan');change[4][...,j]=float('nan')
            self.assertTrue(torch.equal(out,m(*change)['logit']),arm)
            MODEL.objective(out,torch.tensor([0.,1.,0.,1.,0.,1.]),torch.tensor(1.)).backward()
            self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters()),arm)
            self.assertGreater(float(m.head[2].weight.grad.norm()),0)

    def test_branch_isolation_PR_reference_and_alias_boundaries(self):
        initial,args=source(),batch();full=MODEL.NoLateF0(initial,'FULL')
        raw=MODEL.NoLateF0(initial,'RAW_ONLY');a=raw(*args)['logit'];change=[x.clone() for x in args]
        change[0][:]=float('nan');change[4][:]=float('nan')
        self.assertTrue(torch.equal(a,raw(*change)['logit']))
        phys=MODEL.NoLateF0(initial,'PHYSIOLOGY_ONLY')
        self.assertTrue(torch.equal(phys(*args)['logit'],phys(args[0])['logit']))
        nofeature=MODEL.NoLateF0(initial,'WO_FEATURE_PR');self.assertFalse(nofeature.feature_pr or nofeature.guidance_pr);self.assertTrue(nofeature.raw_pr)
        noraw=MODEL.NoLateF0(initial,'WO_RAW_PR');self.assertFalse(noraw.raw_pr);self.assertTrue(noraw.feature_pr and noraw.guidance_pr)
        both=MODEL.NoLateF0(initial,'WO_BOTH_PR');self.assertFalse(both.feature_pr or both.guidance_pr or both.raw_pr)
        window=MODEL.NoLateF0(initial,'WO_WINDOW_PR');self.assertTrue(window.feature_pr and window.raw_pr);self.assertFalse(window.guidance_pr)
        early=MODEL.NoLateF0(initial,'WO_EARLY_FILM');self.assertFalse(early.early or early.late or early.guidance_pr);self.assertTrue(early.feature_pr and early.raw_pr)
        self.assertEqual(MODEL.ALIASES,dict(WO_LATE_GATE='FULL',WO_BOTH_GUIDANCE='WO_EARLY_FILM'))
        pre=MODEL.NoLateF0(initial,'WO_PRE_REFERENCE');self.assertFalse(pre.pre_reference);self.assertTrue(pre.raw_pr)
        mismatch=MODEL.NoLateF0(initial,'EARLY_FILM_MISMATCH');perm,info=MODEL.channel_mapping(args[2],list('abcdef'),'fixture',1)
        self.assertEqual(info['changed_valid_windows'],info['eligible_valid_windows'])
        self.assertFalse(torch.equal(mismatch(*args,perm)['logit'],mismatch(*args,torch.arange(6).repeat(2,1))['logit']))
        self.assertEqual(sum(p.numel() for p in mismatch.parameters()),sum(p.numel() for p in full.parameters()))

if __name__=='__main__':
    torch.set_num_threads(2)
    unittest.main()
