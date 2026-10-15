"""Final read-only scientific source/protected-file gate; publish no private paths."""
import sys
import json
from pathlib import Path
sys.path.insert(0,r'E:\DRE-nips\new-pipeline\7-11\raw_delta_encoder_seed42_v1\code')
from io_utils import sha,write
TASK=Path(__file__).resolve().parents[1];OUT=Path(r'D:\m8_physiology_feature_discovery_seed42_v1_runtime')
def jr(p):return json.loads(p.read_text())
old=jr(OUT/'old7_matched_M8/INPUT_FREEZE_PRIVATE.json')
new=jr(OUT/'integrated_M8_v2/MODEL_FREEZE_PRIVATE.json')
final=jr(OUT/'final_Outer/FROZEN_BEFORE_OUTER_PRIVATE.json')
phases={'old7':old['source'],'integrated':new['binding']['source'],'final_prediction':final['binding']['source']}
for phase,files in phases.items():
    for name,h in files.items():assert sha(TASK/name)==h,(phase,name)
for path,h in old['protected'].items():assert sha(path)==h
assert sha(OUT/'integrated_M8_v2/MODEL_FREEZE_PRIVATE.json')==final['binding']['integrated_freeze']
assert sha(OUT/'old7_matched_M8/INPUT_FREEZE_PRIVATE.json')==final['binding']['old7_freeze']
assert sha(Path(r'D:\m8_raw_backbones_pr_seed42_v1\evaluate.py'))==final['binding']['metric_engine_sha256']
assert jr(OUT/'final_Outer/public/INDEPENDENT_FINAL_VERIFICATION.json')['status']=='PASS'
write(OUT/'public/FROZEN_SCIENTIFIC_SOURCE_AUDIT.json',dict(status='PASS',source_sha256=sha(__file__),
    phase_source_sha256=phases,protected_historical_artifacts_count=len(old['protected']),protected_historical_artifacts_unchanged=True,
    final_integrated_and_old_model_freezes_unchanged=True,original_metric_engine_unchanged=True,
    no_model_data_or_threshold_changes_after_final_freeze=True,no_training_or_inference_called=True))
print('ALL_FROZEN_SCIENTIFIC_SOURCES_AND_HISTORICAL_PROTECTED_ARTIFACTS_UNCHANGED_PASS')
