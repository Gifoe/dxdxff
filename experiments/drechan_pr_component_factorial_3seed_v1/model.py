"""Exact original architecture; only the two existing summary PR operations vary."""
import importlib.util
from pathlib import Path
import sys

PARENT=Path(r'D:\m8_6d_no_interaction_ablation_seed42_v1')
spec=importlib.util.spec_from_file_location('prf_original_model',PARENT/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['FULL','WO_FEATURE_PR','WO_RAW_PR','WO_BOTH_PR']

class PRFactorial(BASE.Ablation):
    def __init__(self,source,arm):
        assert arm in ARMS
        super().__init__(source,'FULL_F0')
        self.variant=arm
        self.feature_pr=arm not in ['WO_FEATURE_PR','WO_BOTH_PR']
        self.raw_pr=arm not in ['WO_RAW_PR','WO_BOTH_PR']
        assert not self.interaction and self.keep7[4]==0

objective=BASE.objective
