"""Stage A only. No torch, labels, model, checkpoint, training or test reader."""
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path
import numpy as np
import pandas as pd


def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()


def write(path,obj):
 path.parent.mkdir(parents=True,exist_ok=True)
 path.write_text(json.dumps(obj,indent=2,allow_nan=False),encoding='utf-8')


def decide(fraction,centers,source_verified,cohort_unchanged,mask_valid):
 return bool(fraction>=.95 and all(centers) and source_verified and cohort_unchanged and mask_valid)


def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-runtime',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 here=Path(__file__).parents[1];lockpath=here/'PROTOCOL_LOCK.json';lock=json.loads(lockpath.read_text())
 assert lock['stage_A']['canonical_channel_valid_fraction_min']==.95 and not lock['new_outer_access_authorized']
 root=a.audit_runtime;out=a.output;out.mkdir(parents=True,exist_ok=True)
 old=json.loads((root/'public/DATA_AVAILABILITY_GATE.json').read_text())
 freeze=json.loads((root/'AUDIT_FREEZE_PRIVATE.json').read_text())
 with (root/'RAW_PSD_PRIVATE.pkl').open('rb') as f:raw=pickle.load(f)
 with (root/'EDF_PLAN_PRIVATE.pkl').open('rb') as f:plan=pickle.load(f)
 disc=json.loads((root/'DISCOVERY_PRIVATE.json').read_text())
 metadata={tuple(r['key']):r for r in disc['raw_primary']['records']}
 assert len(raw)==len(plan)==256 and len({r['patient'] for r in raw})==80
 groups=['ALL']+sorted({r['center'] for r in raw});sets={g:{n:set() for n in ['all','geometric','paired','onset','hardware','verified']} for g in groups}
 recordcounts={g:0 for g in groups}
 hpmax=lock['stage_A']['verified_hardware_highpass_max_hz'];lpmin=lock['stage_A']['verified_hardware_lowpass_min_hz']
 formats=Counter()
 for i,(r,q) in enumerate(zip(raw,plan)):
  assert r['key']==q['key']
  with (root/f'paired_private/record_{i:03d}.pkl').open('rb') as f:pair=pickle.load(f)
  m=metadata[tuple(r['key'])];fs=float(m['raw_sfreq']);lo=int(m['raw_valid_start']);hi=lo+int(m['raw_valid_samples'])
  pre=np.rint((np.array(lock['stage_B_if_authorized']['preictal_anchors'])+30)*fs).astype(int)
  post=np.rint((np.array(lock['stage_B_if_authorized']['post_onset_anchors'])+30)*fs).astype(int)
  n=int(round(4*fs));preok=((pre>=lo)&(pre+n<=hi)).sum();postok=((post>=lo)&(post+n<=hi)).sum()
  geom=preok>=2 and postok>=2
  continuous=False
  if q.get('source'):
   with Path(q['source']).open('rb') as f:reserved=f.read(256)[192:236].decode('latin1').strip()
   formats[(r['center'],reserved or 'classic_EDF')]+=1
   continuous=not reserved.startswith('EDF+D')
  for g in ['ALL',r['center']]:
   recordcounts[g]+=1
   for j,name in enumerate(r['names']):
    identity=(r['patient'],name);s=sets[g];s['all'].add(identity)
    if geom:s['geometric'].add(identity)
    if pair.get('paired'):s['paired'].add(identity)
    if q.get('onset_independently_matched'):s['onset'].add(identity)
    hp=q.get('hp',[]);lp=q.get('lp',[])
    hardware=bool(len(hp)>j and hp[j] is not None and hp[j]<=hpmax and lp[j] is not None and lp[j]>=lpmin)
    if hardware:s['hardware'].add(identity)
    if geom and hardware and continuous and q.get('onset_independently_matched') and pair.get('paired'):s['verified'].add(identity)
 rows=[]
 for g in groups:
  s=sets[g];rows.append(dict(center=g,records=recordcounts[g],canonical_channels=len(s['all']),
    candidate_4s_geometric_channels=len(s['geometric']),diagnostically_paired_channels=len(s['paired']),
    sidecar_onset_corroborated_channels=len(s['onset']),known_LF_capable_acquisition_channels=len(s['hardware']),
    independently_verified_candidate_channels=len(s['verified']),verified_fraction=len(s['verified'])/len(s['all']),
    LF24_feature_bank_extracted=False,status='STAGE_A_GEOMETRY_AND_FIDELITY_NOT_LF_FEATURES'))
 assert len(sets['ALL']['all'])==7635
 pd.DataFrame(rows).to_csv(out/'LF_FEATURE_COVERAGE.csv',index=False)
 pd.DataFrame([dict(center=c,format=v,records=n) for (c,v),n in sorted(formats.items())]).to_csv(out/'EDF_HEADER_FORMAT_SUMMARY.csv',index=False)
 original_provenance=old['raw_exact_generation_provenance']=='VERIFIED'
 coverage=rows[0]['verified_fraction'];allcenters=[len(sets[g]['verified'])>0 for g in groups[1:]]
 # The EDF alternative is a distinct gate, not inherited old100%/27-mask logic.
 EDFroute=coverage>=.95 and all(allcenters)
 rawroute=EDFroute and original_provenance and old['paired_delta_theta_alpha_numerical_gate']
 immutable=old['A0_feature_export_unchanged'] and old['split_unchanged'] and old['raw_identity_complete']
 validmask=all(np.asarray(r['times']).size>0 and np.all(r['starts']>=r['valid_start']) and
   np.all(r['starts']+500<=r['valid_end']) for r in raw)
 passed=decide(coverage,allcenters,rawroute or EDFroute,immutable,validmask)
 gate=dict(stage='A',pass_gate=passed,terminal=None if passed else 'LF_DATA_FIDELITY_GATE_FAILED',
  signal_provenance='VERIFIED' if passed else 'LOW_FREQUENCY_PROVENANCE_UNVERIFIED',
  canonical_channels=7635,verified_candidate_channels=len(sets['ALL']['verified']),verified_fraction=coverage,
  required_fraction=.95,all_centers_have_verified_candidate_evidence=all(allcenters),
  raw_generation_provenance_verified=original_provenance,raw_route_pass=bool(rawroute),EDF_recovery_route_pass=bool(EDFroute),
  cohort_and_original_A0_preserved=bool(immutable),no_padding_in_measured_windows=bool(validmask),
  old100percent_or_historical27_gate_used_as_FCPA95_gate=False,
  input_hashes={'protocol':sha(lockpath),'stage_A_code':sha(Path(__file__)),
   'initial_signal_protocol':freeze['dependencies']['protocol'],'metadata_freeze':sha(root/'AUDIT_FREEZE_PRIVATE.json'),
   'measured_Raw_PSD':sha(root/'RAW_PSD_PRIVATE.pkl'),'private_EDF_plan':sha(root/'EDF_PLAN_PRIVATE.pkl'),
   'old_signal_gate':sha(root/'public/DATA_AVAILABILITY_GATE.json')},
  training_performed=False,LF24_extracted=False,baseline_development_replayed=False,new_outer_accessed=False,
  reasons=[] if passed else ['Raw cache exact filtering/build provenance remains unproven',
   'Independently corroborated time/acquisition evidence fails95percent canonical coverage and all-center requirement'])
 write(out/'DATA_GATE.json',gate)
 write(out/'SCIENTIFIC_TERMINAL.json',dict(terminal=gate['terminal'],stage_completed='A',
   later_stages='NOT_RUN_MANDATORY_DATA_GATE',method_efficacy='NOT_EVALUATED',new_outer_accessed=False))
 write(out/'VALIDATION_GATE.json',dict(status='NOT_RUN_STAGE_A_FAILED' if not passed else 'NOT_RUN',
   no_performance_criterion_relaxed=True))
 write(out/'RUN_STATUS.json',dict(status='COMPLETE_NEGATIVE_DATA_AUDIT' if not passed else 'STAGE_A_PASS',
   stage_A_executed=True,stages_B_to_F_executed=False,models_trained=0,new_outer_evaluations=0,
   private_waveforms_ids_paths_or_predictions_exported=False))
 print('FCPA_LF_STAGE_A_COMPLETE',gate['terminal'] or 'PASS',flush=True)


if __name__=='__main__':main()
