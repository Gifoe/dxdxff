#!/usr/bin/env python3
"""Summarize adapter residuals and enforce the non-LZU invariance gate."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd
def main():
 p=argparse.ArgumentParser();p.add_argument('--shared_root',required=True);p.add_argument('--adapter_root',required=True);p.add_argument('--bootstrap_repeats',type=int,default=2000);p.add_argument('--seed',type=int,default=42);a=p.parse_args();root=Path(a.adapter_root);audit=json.loads((root/'non_lzu_invariance_audit.json').read_text())
 if not audit.get('passed',False):raise RuntimeError('Adapter result invalid: non-LZU invariance failed')
 residual=pd.read_csv(root/'adapter_residual_diagnostics.csv');residual.to_csv(root/'adapter_vs_shared_summary.csv',index=False);(root/'V3_RCC_LZU_ADAPTER_REPORT.md').write_text('# V3-RCC LZU adapter\n\nNon-LZU outputs passed exact invariance checks.\n')
if __name__=='__main__':main()
