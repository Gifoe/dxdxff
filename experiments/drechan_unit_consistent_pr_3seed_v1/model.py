"""No neural math change: original dual-view body and original objective."""
import importlib.util
from pathlib import Path
import sys

PARENT=Path(r'D:\drechan_raw_dualview_3seed_v1')
sys.path.insert(0,str(PARENT))
spec=importlib.util.spec_from_file_location('ucp_unchanged_dualview_model',PARENT/'model.py')
BASE=importlib.util.module_from_spec(spec);sys.modules[spec.name]=BASE;spec.loader.exec_module(BASE)
ARMS=['B0','PRUNED_REF','LEGACY_DUALVIEW','FULL','WO_FEATURE_PR','WO_RAW_PR','WO_BOTH_PR','WO_GLOBAL_VIEW','WO_PRE_REFERENCE']
REFERENCES={'B0':'B0','PRUNED_REF':'PRUNED_REF','LEGACY_DUALVIEW':'FULL'}
PARAMETERS={a:11425 if a=='B0' else 9329 if a=='PRUNED_REF' else 15473 for a in ARMS}
DualView=BASE.DualView
objective=BASE.objective
