"""Identical no-Early/no-Late graph; no new modules or wider head."""
import importlib.util
from pathlib import Path
import sys
PARENT=Path(r'D:\drechan_f0_no_lategate_ablation_seed42_v1')
if not PARENT.is_dir():PARENT=Path(__file__).resolve().parent.parent/PARENT.name
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('noearly5262_parent_model',PARENT/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['WO_EARLY_FILM']
PARAMETERS={'WO_EARLY_FILM':9329}
DROP={}
REFERENCES=set()
objective=BASE.objective

class NoEarlyF0(BASE.NoLateF0):
    def __init__(self,source,arm):
        assert arm=='WO_EARLY_FILM'
        super().__init__(source,arm)
        assert not self.early and not self.late and not self.guidance_pr
        assert not hasattr(self,'gate') and not hasattr(self.cnn,'conditioner')
        assert self.feature_pr and self.raw_pr and self.pre_reference and not self.interaction
        assert self.head[0].out_features==64
        assert sum(p.numel() for p in self.parameters())==9329
