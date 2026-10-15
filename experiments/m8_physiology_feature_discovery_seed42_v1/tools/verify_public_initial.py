"""Local aggregate/source/privacy gate for the completed initial phases only."""
import csv
import hashlib
import json
from pathlib import Path

root=Path(__file__).resolve().parents[1];pub=root/'initial_results'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(name):
    with (pub/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
for name,h in json.loads((pub/'PUBLIC_INITIAL_SHA256.json').read_text()).items():assert sha(pub/name)==h,name
for status in ['OLD7_AUDIT_STATUS.json','CANDIDATE_EXTRACTION_STATUS.json','CANDIDATE_SCREEN_STATUS.json']:
    q=json.loads((pub/status).read_text())
    for name,h in q['source'].items():assert sha(root/name)==h,name
    assert q['full_goal_complete'] is False
v=json.loads((pub/'INITIAL_AUDITS_INDEPENDENT_VERIFICATION.json').read_text())
assert v['status']=='PASS' and v['candidate_linear_probes_replayed']==390 and v['complete_M8_feature_discovery_goal'] is False
assert sha(root/'tools/verify_initial_audits.py')==v['verifier_sha256']
assert len(rows('OLD7_INNER_PROBE_METRICS.csv'))==225 and len(rows('CANDIDATE_INNER_PROBE_METRICS.csv'))==390
assert len(rows('OLD7_PATIENT_ISOLATION_AUDIT.csv'))==15
assert len(rows('CANDIDATE_PROVISIONAL_ELIGIBILITY.csv'))==55
coverage=next(r for r in rows('CANDIDATE_COVERAGE.csv') if r['center']=='ALL')
assert [int(coverage[k]) for k in ['patients','channels','seizures','valid_channel_windows']]==[80,7635,256,1471965]
for p in pub.glob('*.csv'):
    with p.open(newline='',encoding='utf-8') as f:columns=next(csv.reader(f))
    assert not set(columns)&{'patient','channel','path','checkpoint','probability','prediction','y','indices'}
print('INITIAL_PHASES_PUBLIC_SOURCE_COUNTS_PRIVACY_PASS_FULL_GOAL_STILL_ACTIVE')
