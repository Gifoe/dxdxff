"""Acquisition-unit diagnostic only; no numerical gate or clinical outcome."""
import argparse
from collections import Counter
import pickle
from pathlib import Path
import sys
import numpy as np
from io_utils import read,sha,write

def main(a):
 d=read(a.runtime/'RECONSTRUCTION_PRIVATE.pkl');path=Path(r'D:\nips-temp\neuroez_c_four_center_caches_task1_s5_8_v1\all_window_cache.pkl');assert sha(path)==d['binding']['window']
 sys.path.insert(0,str(a.source));from neuroez_c.dual_view_data import _record_key,_field
 with path.open('rb') as f:payload=pickle.load(f)
 lookup={r['compound_key']:r for p in d['ledgers'] for r in p['records']};rates=Counter();starts=[];ends=[];unknown=0
 for r in payload['run_records']:
  key=_record_key(r)
  if key not in lookup:continue
  q=lookup[key];sf=_field(r,'raw_temporal_sfreq',None)
  if sf is None or not np.isfinite(float(sf)) or float(sf)<=0:unknown+=1;continue
  sf=float(sf);rates[sf]+=1;starts.append(q['feature_valid'][0]/sf-q['raw_valid'][0]/250);ends.append(q['feature_valid'][1]/sf-q['raw_valid'][1]/250)
 def stats(v):return dict(cases=len(v),median_absolute_seconds=float(np.median(np.abs(v))) if v else None,max_absolute_seconds=float(np.max(np.abs(v))) if v else None,exact_seconds_agreement=int((np.asarray(v)==0).sum()))
 write(a.runtime/'results/VALID_RANGE_METADATA_UNITS.json',dict(status='DIAGNOSTIC_ONLY',records=len(lookup),feature_sampling_rate_metadata=dict(rates),raw_sfreq=250,unknown_feature_rate_records=unknown,valid_start_seconds_difference=stats(starts),valid_end_seconds_difference=stats(ends),integer_sample_bounds_not_comparable_until_rate_normalized=True,no_onset_clock_or_filter_provenance_claim=True,clinical_labels_used=False,source_sha256=sha(Path(__file__))))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
