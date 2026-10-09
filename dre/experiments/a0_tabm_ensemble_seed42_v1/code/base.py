"""Hash-locked original A0 preprocessing, metrics and state helpers."""
import sys
import hashlib
from pathlib import Path
prior=Path(__file__).resolve().parents[2]/'pr_uncertainty_aware_supervision_seed42_v1'/'code'
assert hashlib.sha256((prior/'uas_core.py').read_bytes()).hexdigest()=='1b1ccb0a361e6245cb2e6089347e147036aac90437c0992194f6db7bebf1503a'
sys.path.insert(0,str(prior))
from uas_core import PRMLP,prepare,sha,json_write,torch_write,seed_all,state_hash,rng_state,restore_rng,select_threshold,patient_metrics,METRICS
