#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--root_dir',required=True);p.add_argument('--bootstrap_repeats',type=int,default=2000);p.add_argument('--seed',type=int,default=42);a=p.parse_args();root=Path(a.root_dir);rows=[]
 for name in ('R0_BASE','R1_RANK_COVERAGE','R2_HYBRID_CALIBRATION'):
  path=root/name/'formal_summary.csv';true=root/name/'truek_summary.csv'
  if path.is_file() and true.is_file(): rows.append({'profile':name,**pd.read_csv(path).iloc[0].to_dict(),**{'truek_patient_macro_f1':float(pd.read_csv(true).iloc[0].patient_macro_f1)}})
 pd.DataFrame(rows).to_csv(root/'ablation_summary.csv',index=False);(root/'V3_RCC_ABLATION_REPORT.md').write_text('# V3-RCC ablation\n\nProfiles are compared without automatic outer-test selection.\n')
if __name__=='__main__':main()
