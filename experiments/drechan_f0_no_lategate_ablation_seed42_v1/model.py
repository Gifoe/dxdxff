"""Fixed no-Late-Gate main and matched seed42 deletions; immutable parent math."""
import importlib.util
from pathlib import Path
import sys

PARENT = Path(r'D:\drechan_f0_complete_ablation_3seed_v1')
if not PARENT.is_dir():
    PARENT = Path(__file__).resolve().parent.parent/PARENT.name
sys.path.insert(0, str(PARENT))
spec = importlib.util.spec_from_file_location('no_late_original_complete3_math', PARENT/'model.py')
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)
sys.path.insert(0, str(Path(__file__).resolve().parent))

ARMS = ['FULL', 'PREVIOUS_FULL_WITH_LATE', 'B0_ORIGINAL', 'WO_FEATURE_PR', 'WO_RAW_PR', 'WO_BOTH_PR',
        'WO_WINDOW_PR', 'WO_EARLY_FILM', 'WO_PRE_REFERENCE', 'EARLY_FILM_MISMATCH', 'RAW_ONLY',
        'PHYSIOLOGY_ONLY', 'DROP_BETA', 'DROP_LOW_GAMMA', 'DROP_PARTIAL_HIGH_GAMMA', 'DROP_RMS',
        'DROP_LINE_LENGTH', 'DROP_SPECTRAL_ENTROPY']
DROP = BASE.DROP
REFERENCES = {'PREVIOUS_FULL_WITH_LATE', 'B0_ORIGINAL'}
REUSE = dict(FULL='WO_LATE_GATE', PREVIOUS_FULL_WITH_LATE='FULL', B0_ORIGINAL='B0_ORIGINAL',
             WO_EARLY_FILM='WO_BOTH_GUIDANCE', RAW_ONLY='RAW_ONLY', PHYSIOLOGY_ONLY='PHYSIOLOGY_ONLY')
ALIASES = dict(WO_LATE_GATE='FULL', WO_BOTH_GUIDANCE='WO_EARLY_FILM')
PARAMETERS = {a:10377 for a in ARMS}
PARAMETERS.update(PREVIOUS_FULL_WITH_LATE=11425, B0_ORIGINAL=11425, WO_EARLY_FILM=9329,
                  RAW_ONLY=8433, PHYSIOLOGY_ONLY=1025)


class NoLateF0(BASE.CompleteF0):
    def __init__(self, source, arm):
        assert arm in ARMS
        original = REUSE.get(arm, arm)
        super().__init__(source, original)
        if arm not in REFERENCES:
            self.late = False
            if hasattr(self, 'gate'):
                delattr(self, 'gate')
        self.arm = arm
        # After removing the last guidance path no window physiology is read.
        if arm == 'WO_EARLY_FILM':
            self.guidance_pr = False
        assert arm in REFERENCES or (not self.late and not hasattr(self, 'gate'))
        assert sum(p.numel() for p in self.parameters()) == PARAMETERS[arm]
        assert not self.interaction and not any('interaction' in n for n,_ in self.named_parameters())


channel_mapping = BASE.channel_mapping
objective = BASE.objective
