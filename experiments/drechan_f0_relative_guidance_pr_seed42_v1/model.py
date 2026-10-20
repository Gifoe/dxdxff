"""Original six-feature F0 architecture; only physiological context changes."""
import importlib.util
from pathlib import Path
import sys
from relative_guidance import conditioning_delta

PARENT = Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1')
spec = importlib.util.spec_from_file_location('relative_guidance_original_f0', PARENT/'model.py')
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)

ARMS = ['B0_ORIGINAL', 'FULL', 'WO_FEATURE_PR', 'WO_RAW_PR', 'WO_BOTH_PR',
        'WO_WINDOW_PR', 'WO_EARLY_FILM', 'WO_LATE_GATE', 'WO_PRE_REFERENCE']
PARAMETERS = {arm: 11425 for arm in ARMS}
PARAMETERS.update(WO_EARLY_FILM=10377, WO_LATE_GATE=10377)


class RelativeF0(BASE.BASE.Ablation):
    def __init__(self, source, arm):
        assert arm in ARMS
        mapping = {'WO_EARLY_FILM': 'WO_EARLY_FILM', 'WO_LATE_GATE': 'WO_LATE_GATE',
                   'WO_PRE_REFERENCE': 'WO_RAW_PRE_REFERENCE'}
        # Complete original initializer first; no trained reference initialization.
        super().__init__(source, mapping.get(arm, 'WO_HEAD_INTERACTION'))
        self.interaction = False
        if 'interaction_gain' in self._parameters:
            delattr(self, 'interaction_gain')
        self.arm = arm
        self.raw_pr = arm not in ['WO_RAW_PR', 'WO_BOTH_PR']
        self.feature_pr = arm not in ['WO_FEATURE_PR', 'WO_BOTH_PR']
        self.guidance_pr = arm not in ['B0_ORIGINAL', 'WO_FEATURE_PR', 'WO_BOTH_PR', 'WO_WINDOW_PR']
        self.collect_guidance_audit = False
        self.last_guidance_audit = {}
        assert self.branch == 'joint' and self.keep7[4] == 0
        assert not any('interaction' in name for name, _ in self.named_parameters())
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS[arm]

    def features(self, raw, mask, times, delta, scale, permutation=None):
        assert permutation is None
        if self.guidance_pr:
            delta, self.last_guidance_audit = conditioning_delta(
                delta, mask, scale, self.keep7, self.collect_guidance_audit)
        else:
            self.last_guidance_audit = {}
        # Original CNN/FiLM/Gate/pre-reference/pooling/RawPR/classifier math.
        return super().features(raw, mask, times, delta, scale, permutation)


objective = BASE.objective
