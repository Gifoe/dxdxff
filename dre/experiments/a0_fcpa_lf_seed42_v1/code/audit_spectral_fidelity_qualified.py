"""Qualify existing index-parity by EDF header timeline and acquisition evidence.

No new waveforms, feature extraction, clinical labels or predictions are read.
The earlier all256 index-based diagnostic remains unchanged and available;
EDF+D is never represented as independently verified physical-time parity.
"""
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from audit_spectral_fidelity import mid_power
from audit_source import sha


def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-runtime',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 with (a.audit_runtime/'EDF_PLAN_PRIVATE.pkl').open('rb') as f:plan=pickle.load(f)
 lock=json.loads((Path(__file__).parents[1]/'PROTOCOL_LOCK.json').read_text())
 hpmax=lock['stage_A']['verified_hardware_highpass_max_hz'];lpmin=lock['stage_A']['verified_hardware_lowpass_min_hz']
 cells=[]
 for i,q in enumerate(plan):
  with (a.audit_runtime/f'paired_private/record_{i:03d}.pkl').open('rb') as f:pair=pickle.load(f)
  if not pair['paired']:continue
  with Path(q['source']).open('rb') as f:reserved=f.read(256)[192:236].decode('latin1').strip()
  continuous=not reserved.startswith('EDF+D')
  scopes=['CONTINUOUS_HEADER_RECORD_CLOCK_DIAGNOSTIC'] if continuous else ['EDF_D_INDEX_ONLY_UNVERIFIED_TIMELINE']
  for scope in scopes:cells.append((scope,pair,np.ones(len(q['hp']),bool)))
  if continuous and q['onset_independently_matched']:
   valid=np.array([h is not None and h<=hpmax and l is not None and l>=lpmin for h,l in zip(q['hp'],q['lp'])])
   if valid.any():cells.append(('CONTINUOUS_KNOWN_ACQUISITION_DIAGNOSTIC',pair,valid))
 rows=[]
 for scope in sorted({c[0] for c in cells}):
  for center in ['ALL']+sorted({c[1]['center'] for c in cells}):
   subset=[c for c in cells if c[0]==scope and (center=='ALL' or c[1]['center']==center)]
   for source in ['EDF_original','EDF_low05','EDF_reader1','EDF_negative10']:
    for band in ['delta','theta','alpha','beta','mid13_40']:
     row=dict(scope=scope,center=center,reference=source,band=band,records=len(subset),
       status='MEASURED_QUALIFIED_DIAGNOSTIC' if subset else 'NO_COVERAGE_NOT_ESTIMABLE')
     if subset:
      def power(d):return mid_power(d) if band=='mid13_40' else d[band]
      x=np.concatenate([power(q['raw_powers'])[m].ravel() for _,q,m in subset]);y=np.concatenate([power(q['EDF_powers'][source])[m].ravel() for _,q,m in subset])
      ok=np.isfinite(x)&np.isfinite(y)&(y>1e-30);ratio=x[ok]/y[ok]
      row.update(channel_window_pairs=len(x),finite_pair_fraction=float(ok.mean()),
        raw_over_EDF_median=float(np.median(ratio)) if len(ratio) else None,
        raw_over_EDF_q25=float(np.percentile(ratio,25)) if len(ratio) else None,
        raw_over_EDF_q75=float(np.percentile(ratio,75)) if len(ratio) else None,
        spearman=float(spearmanr(x[ok],y[ok]).statistic) if ok.sum()>1 and np.ptp(x[ok]) and np.ptp(y[ok]) else None,
        onset_corroborated_records=sum(q['onset_independently_matched'] for _,q,_ in subset))
     rows.append(row)
 pd.DataFrame(rows).to_csv(a.output/'LF_PSD_COMPARISON_QUALIFIED.csv',index=False)
 binding=dict(code_sha256=sha(Path(__file__)),protocol_sha256=sha(Path(__file__).parents[1]/'PROTOCOL_LOCK.json'),
   private_plan_sha256=sha(a.audit_runtime/'EDF_PLAN_PRIVATE.pkl'),
   original_index_diagnostic_retained=True,EDF_D_physical_timeline_claim=False,
   new_signal_feature_or_label_extraction=False,model_or_gate_changed=False)
 (a.output/'SPECTRAL_QUALIFICATION_BINDING.json').write_text(json.dumps(binding,indent=2),encoding='utf-8')
 print('FCPA_LF_TIMELINE_QUALIFIED_SUMMARY_COMPLETE',flush=True)


if __name__=='__main__':main()
