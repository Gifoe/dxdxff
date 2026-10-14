"""Label-blind exact original per-channel/full44/1D-seizure arithmetic diagnosis."""
import argparse
from collections import defaultdict
import json
import pickle
from pathlib import Path
import sys
import numpy as np
from io_utils import read,load,sha,write,save
from prepare import parity
from raw_physiological_features import FEATURES

def main():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,required=True);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args();d=read(a.runtime/'RECONSTRUCTION_PRIVATE.pkl');meta=d['meta'];ids=set(meta['patient']);source=a.source/'task1_baselines/feature_aggregation.py';assert sha(source)==d['binding']['source_task1_baselines/feature_aggregation.py'];sys.path.insert(0,str(a.source));module=load('rd_exact_original_aggregation_diagnostic',source)
 path=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl');assert sha(path)==d['binding']['window']
 with path.open('rb') as f:cache=pickle.load(f)
 values=defaultdict(list);records=0
 for rec in cache['run_records']:
  pid=str(rec['subject_id'])
  if pid not in ids:continue
  s=rec['sample'];hv,names=module._p2_window_features(dict(features=s['window_features'],window_centers=s['window_relative_centers_sec'],feature_names=s['window_feature_names']));cols=[names.index(f'b0_delta__{f}') for f in FEATURES]
  for j,ch in enumerate(rec['channel_names_norm']):
   v=hv[:,j,:];valid=v[np.isfinite(v).all(1)];values[pid,str(ch)].append(np.nanmean(valid,axis=0))
  records+=1
 exact=np.zeros_like(d['E14'])
 for i,(pid,ch) in enumerate(zip(meta['patient'],meta['channel'])):
  seq=np.stack(values[pid,ch])
  for j,k in enumerate(cols):exact[i,2*j]=np.nanmean(seq[:,k]);exact[i,2*j+1]=np.nanstd(seq[:,k],ddof=0)
 q=parity(exact,d['archived_E14']);write(a.runtime/'results/E14_EXACT_AGGREGATION_DIAGNOSTIC.json',dict(status='PASS' if q['pass_gate'] else 'UNRESOLVED',parity=q,records=records,old_bulk_reconstruction_parity=parity(d['E14'],d['archived_E14']),changed_from_bulk_coordinates=int((exact!=d['E14']).sum()),max_change_from_bulk=float(np.max(np.abs(exact-d['E14']))),diagnostic_only=True,source_hash=sha(source),no_labels_or_new_Raw_read=True,explanation='Original per-channel mean of contiguous full44 evidence rows and per-coordinate1D cross-seizure mean/populationstd versus vectorized reductions over W,C,7 and S,C,7; same measurements/references/support, float32 accumulation/layout differences.'))
 save(a.runtime/'E14_EXACT_DIAGNOSTIC_PRIVATE.pkl',dict(E14=exact,source_binding=d['binding']));print('E14_EXACT_SOURCE_ARITHMETIC_DIAGNOSTIC',q,flush=True)

if __name__=='__main__':main()
