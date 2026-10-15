"""Hash only completed anonymous initial audits; not a full-goal completion gate."""
import json
import sys
from pathlib import Path
sys.path.insert(0,r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code')
from io_utils import sha,write
pub=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime\public')
v=json.loads((pub/'INITIAL_AUDITS_INDEPENDENT_VERIFICATION.json').read_text())
assert v['status']=='PASS' and not v['complete_M8_feature_discovery_goal']
assert all(p.suffix in ['.json','.csv'] for p in pub.iterdir())
write(pub/'PUBLIC_INITIAL_SHA256.json',{p.name:sha(p) for p in pub.iterdir() if p.name!='PUBLIC_INITIAL_SHA256.json'})
print('ANONYMOUS_INITIAL_ARTIFACT_HASHES_WRITTEN_FULL_GOAL_NOT_COMPLETE')
