"""Anonymous header sampling/calibration/filter inventory, not signal extraction."""
import argparse
import pickle
from collections import Counter
from pathlib import Path
import pandas as pd


def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-runtime',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 with (a.audit_runtime/'EDF_PLAN_PRIVATE.pkl').open('rb') as f:plan=pickle.load(f)
 rows=[]
 for center in ['ALL']+sorted({q['center'] for q in plan}):
  part=[q for q in plan if center=='ALL' or q['center']==center]
  for field,key in [('native_sampling_hz','native_channel_sfreq'),('physical_unit','units'),('hardware_HP_hz','hp'),('hardware_LP_hz','lp')]:
   values=Counter(str(v) if v is not None else 'UNKNOWN' for q in part for v in q.get(key,[]))
   for value,count in sorted(values.items()):rows.append(dict(center=center,field=field,value=value,selected_channel_occurrences=count,records=len(part)))
 a.output.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows).to_csv(a.output/'EDF_HEADER_SIGNAL_SPEC_SUMMARY.csv',index=False)
 print('FCPA_LF_HEADER_SIGNAL_SPEC_COMPLETE',flush=True)


if __name__=='__main__':main()
