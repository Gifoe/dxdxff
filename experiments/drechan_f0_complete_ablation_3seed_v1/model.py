"""Complete fixed six-measurement F0 matrix; original initialization and widths."""
import importlib.util
from pathlib import Path
import sys
from relative_guidance import conditioning_delta

PARENT = Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1')
spec = importlib.util.spec_from_file_location('complete3_original_no_interaction', PARENT/'model.py')
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)

ARMS = ['B0_ORIGINAL', 'FULL', 'WO_FEATURE_PR', 'WO_RAW_PR', 'WO_BOTH_PR',
        'WO_WINDOW_PR', 'WO_EARLY_FILM', 'WO_LATE_GATE', 'WO_PRE_REFERENCE',
        'WO_BOTH_GUIDANCE', 'EARLY_FILM_MISMATCH', 'RAW_ONLY', 'PHYSIOLOGY_ONLY',
        'DROP_BETA', 'DROP_LOW_GAMMA', 'DROP_PARTIAL_HIGH_GAMMA', 'DROP_RMS',
        'DROP_LINE_LENGTH', 'DROP_SPECTRAL_ENTROPY']
DROP = dict(DROP_BETA=0, DROP_LOW_GAMMA=1, DROP_PARTIAL_HIGH_GAMMA=2,
            DROP_RMS=3, DROP_LINE_LENGTH=5, DROP_SPECTRAL_ENTROPY=6)
PARAMETERS = {a: 11425 for a in ARMS}
PARAMETERS.update(WO_EARLY_FILM=10377, WO_LATE_GATE=10377, WO_BOTH_GUIDANCE=9329,
                  RAW_ONLY=8433, PHYSIOLOGY_ONLY=1025)


class CompleteF0(BASE.BASE.Ablation):
    def __init__(self, source, arm):
        assert arm in ARMS
        mapping = dict(WO_EARLY_FILM='WO_EARLY_FILM', WO_LATE_GATE='WO_LATE_GATE',
                       WO_BOTH_GUIDANCE='WO_BOTH_GUIDANCE', WO_PRE_REFERENCE='WO_RAW_PRE_REFERENCE',
                       EARLY_FILM_MISMATCH='EARLY_FILM_MISMATCH', RAW_ONLY='RAW_ONLY',
                       PHYSIOLOGY_ONLY='PHYSIOLOGY_ONLY')
        mapping.update({a: a for a in DROP})
        # The complete original source initializer is consumed before deletion.
        super().__init__(source, mapping.get(arm, 'WO_HEAD_INTERACTION'))
        self.interaction = False
        if 'interaction_gain' in self._parameters:
            delattr(self, 'interaction_gain')
        self.arm = arm
        self.raw_pr = arm not in ['WO_RAW_PR', 'WO_BOTH_PR']
        self.feature_pr = arm not in ['WO_FEATURE_PR', 'WO_BOTH_PR', 'RAW_ONLY']
        self.guidance_pr = arm not in ['B0_ORIGINAL', 'WO_FEATURE_PR', 'WO_BOTH_PR',
                                       'WO_WINDOW_PR', 'WO_BOTH_GUIDANCE', 'RAW_ONLY', 'PHYSIOLOGY_ONLY']
        self.collect_guidance_audit = False
        self.last_guidance_audit = {}
        assert self.branch == ('raw' if arm == 'RAW_ONLY' else 'phys' if arm == 'PHYSIOLOGY_ONLY' else 'joint')
        # Historical Raw-only has an unused all-one keep buffer. Keep it exact
        # for audited reuse; the Raw-only graph never reads physiological input.
        assert arm == 'RAW_ONLY' or self.keep7[4] == 0
        if arm in DROP:
            assert self.keep7[DROP[arm]] == 0
        assert not any('interaction' in n for n, _ in self.named_parameters())
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS[arm]

    def features(self, raw, mask, times, delta, scale, permutation=None):
        assert (permutation is not None) == (self.arm == 'EARLY_FILM_MISMATCH')
        self.last_guidance_audit = {}
        if self.guidance_pr:
            delta, self.last_guidance_audit = conditioning_delta(
                delta, mask, scale, self.keep7, self.collect_guidance_audit)
        # Relative context is computed correctly BEFORE early-only donor routing.
        # Parent routing leaves late Gate/direct physiology correctly aligned.
        return super().features(raw, mask, times, delta, scale, permutation)


channel_mapping = BASE.channel_mapping
objective = BASE.objective
