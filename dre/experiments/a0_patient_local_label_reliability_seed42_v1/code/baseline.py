"""Import immutable validated PR-UAS A0 utilities; never its soft-target trainer."""
import hashlib
import sys
from pathlib import Path

PRIOR_CODE = Path(__file__).resolve().parents[2] / 'pr_uncertainty_aware_supervision_seed42_v1' / 'code'
CORE_SHA = '1b1ccb0a361e6245cb2e6089347e147036aac90437c0992194f6db7bebf1503a'
assert hashlib.sha256((PRIOR_CODE / 'uas_core.py').read_bytes()).hexdigest() == CORE_SHA
sys.path.insert(0, str(PRIOR_CODE))
from uas_core import (PRMLP, METRICS, prepare, patient_metrics, select_threshold,
                      sha, json_write, torch_write, seed_all, state_hash,
                      groups, rng_state, restore_rng, train_model)
