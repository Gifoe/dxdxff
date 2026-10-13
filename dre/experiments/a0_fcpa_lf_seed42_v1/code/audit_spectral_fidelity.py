"""Summarize actually executed aligned PSD integrals; no feature bank or ML."""
import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def mid_power(p):
 # Residual diagnostic. It equals13..40 only if native bin edges tile exactly;
 # noninteger header rates may miss edges. Never call this a full PSD-shape proof.
 return p['total']-p['delta']-p['theta']-p['alpha']


def main():
 p=argparse.ArgumentParser();p.add_argument('--audit-runtime',required=True,type=Path)
 p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 rows=pd.read_csv(a.audit_runtime/'public/RAW_VS_EDF_PARITY_SUMMARY.csv').to_dict('records')
 pairs=[]
 for i in range(256):
  with (a.audit_runtime/f'paired_private/record_{i:03d}.pkl').open('rb') as f:q=pickle.load(f)
  if q['paired']:pairs.append(q)
 for center in ['ALL']+sorted({q['center'] for q in pairs}):
  subset=[q for q in pairs if center=='ALL' or q['center']==center]
  for source in ['EDF_original','EDF_low05','EDF_reader1','EDF_negative10']:
   x=np.concatenate([mid_power(q['raw_powers']).ravel() for q in subset])
   y=np.concatenate([mid_power(q['EDF_powers'][source]).ravel() for q in subset])
   ok=np.isfinite(x)&np.isfinite(y)&(y>1e-30);ratio=x[ok]/y[ok]
   rows.append(dict(center=center,reference=source,band='mid13_40',status='MEASURED_SIGNAL_ONLY',
    records=len(subset),pairs=len(x),finite_pair_fraction=float(ok.mean()),
    raw_over_EDF_q25=float(np.percentile(ratio,25)) if len(ratio) else None,
    raw_over_EDF_median=float(np.median(ratio)) if len(ratio) else None,
    raw_over_EDF_q75=float(np.percentile(ratio,75)) if len(ratio) else None,
    spearman=float(spearmanr(x[ok],y[ok]).statistic) if ok.sum()>1 and np.ptp(x[ok]) and np.ptp(y[ok]) else None,
    independent_onset_records=sum(q['onset_independently_matched'] for q in subset)))
 a.output.mkdir(parents=True,exist_ok=True)
 result=pd.DataFrame(rows);result['interpretation']='Same-window Welch integrals; originalEDF acquisition not universally verified; not dense PSD-shape proof'
 result.to_csv(a.output/'LF_PSD_COMPARISON.csv',index=False)
 print('FCPA_LF_ALIGNED_SPECTRAL_SUMMARY_COMPLETE',flush=True)


if __name__=='__main__':main()
