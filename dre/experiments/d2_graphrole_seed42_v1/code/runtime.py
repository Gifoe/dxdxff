"""Read-only, hash-locked original D2 optimization helpers."""
import sys
from pathlib import Path
from common import sha
CORE=Path(r'E:\DRE-nips\new-pipeline\7-11\pr_uncertainty_aware_supervision_seed42_v1\code')
assert sha(CORE/'uas_core.py')=='1b1ccb0a361e6245cb2e6089347e147036aac90437c0992194f6db7bebf1503a'
sys.path.insert(0,str(CORE))
from uas_core import seed_all,state_hash,restore_rng,rng_state,select_threshold,torch_write,json_write,groups,patient_z
