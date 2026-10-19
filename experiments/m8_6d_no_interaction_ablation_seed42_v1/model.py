"""Compose approved six-feature switches with removal of ALL head interaction."""
import importlib.util
from pathlib import Path
import sys

PREVIOUS=Path(r'D:\m8_6d_no_variance_ablation_baselines_seed42_v1')
spec=importlib.util.spec_from_file_location('ni_previous_sixd_math',PREVIOUS/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['FULL_F0','WO_EARLY_FILM','WO_BOTH_GUIDANCE','WO_BOTH_PR','WO_RAW_PRE_REFERENCE','EARLY_FILM_MISMATCH','RAW_ONLY','PHYSIOLOGY_ONLY','DROP_BETA','DROP_LOW_GAMMA','DROP_PARTIAL_HIGH_GAMMA','DROP_RMS','DROP_LINE_LENGTH','DROP_SPECTRAL_ENTROPY']

class Ablation(BASE.Ablation):
    def __init__(self,source,arm):
        assert arm in ARMS
        super().__init__(source,'WO_HEAD_INTERACTION' if arm=='FULL_F0' else arm)
        # Original initializer is constructed first, preserving all shared tensors/RNG.
        self.interaction=False
        if 'interaction_gain' in self._parameters:delattr(self,'interaction_gain')
        self.arm=arm
        assert not self.interaction and not any('interaction' in n for n,_ in self.named_parameters())

channel_mapping=BASE.channel_mapping
objective=BASE.objective
