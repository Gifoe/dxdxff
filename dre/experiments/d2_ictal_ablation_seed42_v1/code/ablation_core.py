"""Fixed-name feature masks and a shared-only constructor; no D2 redesign."""
import re
import numpy as np
import torch

ARMS = ['Full D2','w/o Patient Z','w/o Source','ABS Only','w/o DELTA','w/o ZDELTA','w/o RATIO']
NEW_ARMS = ARMS[2:]
KEYS = dict(zip(ARMS, ['full','no_patient_z','no_source','abs_only','no_delta','no_zdelta','no_ratio']))
KEEP = {'Full D2': ['abs','delta','zdelta','ratio'], 'w/o Patient Z': ['abs','delta','zdelta','ratio'],
    'w/o Source': ['abs','delta','zdelta','ratio'], 'ABS Only': ['abs'],
    'w/o DELTA': ['abs','zdelta','ratio'], 'w/o ZDELTA': ['abs','delta','ratio'],
    'w/o RATIO': ['abs','delta','zdelta']}


def parse_names(names):
    assert len(names) == len(set(names)) == 88
    columns = []
    for i, name in enumerate(names):
        match = re.fullmatch(r'(b0|physics)_(abs|delta|zdelta|ratio)__(.+)__seizure_(mean|std)', name)
        assert match is not None, f'Unmapped feature {name}'
        block, view, feature, statistic = match.groups()
        columns.append({'index': i, 'name': name, 'block': block, 'view': view,
            'physiological_feature': feature, 'aggregation': statistic})
    return columns


def mask_for(columns, arm):
    assert arm in ARMS
    mask = np.asarray([c['view'] in KEEP[arm] for c in columns], dtype=bool)
    assert mask.shape == (88,)
    return mask


def masked(x, mask):
    assert x.ndim == 2 and x.shape[1] == 88 and mask.shape == (88,)
    out = np.asarray(x, dtype=np.float32).copy()
    out[:, ~mask] = 0.
    assert np.array_equal(out[:, mask], x[:, mask]) and np.all(out[:, ~mask] == 0)
    assert np.isfinite(out).all()
    return out


def source_free_factory(original):
    class SharedOnly(torch.nn.Module):
        def __init__(self, seed=42):
            super().__init__()
            # Original construction preserves exact shared weights and RNG semantics.
            self.network = original(seed).network
            assert sum(p.numel() for p in self.parameters()) == 8817

        def forward(self, x, g):
            return self.network(x).squeeze(-1)

        def regularizer(self, counts):
            return next(self.parameters()).new_zeros(())
    return SharedOnly
