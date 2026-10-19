"""Remove unsupported guidance/gains; compare joint vs branch-separate activation."""
import copy
import importlib.util
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from fusion_math import fuse_hidden

PREVIOUS = Path(r'D:\drechan_pr_component_factorial_3seed_v1')
spec = importlib.util.spec_from_file_location('pbf_component_model', PREVIOUS / 'model.py')
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)
ARMS = ['B0', 'JOINT_PRUNED', 'FULL', 'WO_FEATURE_PR', 'WO_RAW_PR', 'WO_BOTH_PR', 'WO_PRE_REFERENCE']
PARAMETERS = {a: 11425 if a == 'B0' else 9329 for a in ARMS}


def remove_guidance(m):
    """No RNG use; retain exact shared convolution/pooling/head tensors."""
    assert m.early and m.late and not m.interaction
    m.cnn = nn.Sequential(*copy.deepcopy(list(m.cnn.stem.children())), copy.deepcopy(m.cnn.pool))
    del m.gate
    m.early = m.late = False
    return m


class PrunedFusion(BASE.PRFactorial):
    def __init__(self, source, arm):
        assert arm in ARMS[1:]
        super().__init__(source, 'FULL')
        remove_guidance(self)
        self.variant = arm
        self.feature_pr = arm not in ['WO_FEATURE_PR', 'WO_BOTH_PR']
        self.raw_pr = arm not in ['WO_RAW_PR', 'WO_BOTH_PR']
        self.pre_reference = arm != 'WO_PRE_REFERENCE'
        self.split_fusion = arm != 'JOINT_PRUNED'
        self.register_buffer('anchor_mean', torch.zeros(96, dtype=torch.float64))
        self.register_buffer('anchor_scale', torch.ones(96, dtype=torch.float64))
        self.collect_anchor = False
        assert not any('gain' in n or 'conditioner' in n or n.startswith('gate.') for n, _ in self.named_parameters())

    def features(self, *args, **kwargs):
        z, activity = super().features(*args, **kwargs)
        if not self.raw_pr and not self.collect_anchor:
            z = ((z.double() - self.anchor_mean) / self.anchor_scale).to(z.dtype)
        return z, activity

    def classify(self, z, x):
        x = torch.where(self.keep7.repeat_interleave(2).bool(), x, torch.zeros_like(x))
        first = self.head[0]
        a = F.linear(z * math.sqrt(110 / 192), first.weight[:, :96])
        b = F.linear(x * math.sqrt(110 / 28), first.weight[:, 96:])
        hidden = fuse_hidden(self.head, a, b, self.split_fusion)
        return self.head[2](hidden).squeeze(-1)

    @torch.no_grad()
    def set_anchor(self, anchor):
        assert anchor['mean'].shape == anchor['scale'].shape == (96,)
        assert torch.isfinite(anchor['mean']).all() and torch.isfinite(anchor['scale']).all() and (anchor['scale'] > 0).all()
        self.anchor_mean.copy_(anchor['mean'])
        self.anchor_scale.copy_(anchor['scale'])


objective = BASE.objective
