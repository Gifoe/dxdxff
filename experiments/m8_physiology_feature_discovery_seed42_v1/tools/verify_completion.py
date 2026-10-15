"""Full release evidence audit, not a test that demands a favorable scientific result."""
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
def jr(p):return json.loads(p.read_text(encoding='utf-8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):
    with p.open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
audit=jr(root/'GOAL_COMPLETION_AUDIT.json')
assert audit['status']=='ALL_RESEARCH_REQUIREMENTS_VERIFIED'
assert audit['primary_improvement_established'] is False and audit['global_optimality_claimed'] is False
assert not audit['remaining_required_work']
assert len(audit['requirements'])==11 and {r['id'] for r in audit['requirements']}=={f'R{i:02d}' for i in range(1,12)}
for r in audit['requirements']:
    assert r['status']=='VERIFIED_WITH_EXPLICIT_LIMITS' and r['evidence'] and r['interpretation_limit']
    for p in r['evidence']:assert p in audit['release_sha256'] and (root/p).is_file(),(r['id'],p)
for p,h in audit['release_sha256'].items():assert sha(root/p)==h,p
for name in ['verify_public_initial.py','verify_phase2_public.py','verify_phase3_public.py','verify_final_public.py']:
    subprocess.run([sys.executable,str(root/'tools'/name)],check=True)
assert len(rows(root/'final_results/outer/main_results.csv'))==26
assert len(rows(root/'final_results/outer/fold_metrics.csv'))==260
assert len(rows(root/'final_results/outer/center_metrics.csv'))==208
assert len(rows(root/'final_results/outer/metric_coverage.csv'))==806
assert len(rows(root/'final_results/outer/paired_intervals.csv'))==1023
assert len(rows(root/'phase3_results/old7_development/MODEL_SELECTION.csv'))==80
assert len(rows(root/'final_results/integrated/MODEL_SELECTION.csv'))==45
ind=jr(root/'final_results/outer/INDEPENDENT_FINAL_VERIFICATION.json')
assert (ind['arms'],ind['armfolds'],ind['patient_rows'],ind['aliases_exact'])==(26,130,2080,9)
assert ind['all_patient_cache_inputs_independently_reconstructed'] and ind['all33_paired_bootstrap_comparisons_reproduced']
assert ind['metric_process_did_not_import_torch'] and ind['no_training_or_predictor_forward_called']
compute=jr(root/'final_results/compute/COMPUTE_AUDIT.json')
assert compute['total_parameters']==[11018,11690] and compute['parameter_ratio']<=1.1
assert compute['offline_feature_cost_separately_measured'] and compute['D1_epoch_timing_missing_reported_not_imputed']
report=(root/'REPORT.md').read_text(encoding='utf-8')
for phrase in ['EXPLORATORY_REPEATED_TEST','不是七项都已证明必需','同步提升F1','不是完整端到端FLOPs','不支持“七项缺一不可”或“原7D最优”']:
    assert phrase in report,phrase
for p in root.rglob('*'):
    if not p.is_file() or '__pycache__' in p.parts:continue
    assert p.suffix.lower() not in ['.pkl','.npz','.pt','.edf','.npy','.log','.err'],p.name
print('M8_FEATURE_DISCOVERY_FULL_SCOPE_COMPLETION_AUDIT_PASS_PRIMARY_GATE_NOT_MET_NO_GLOBAL_OPTIMALITY_CLAIM')
