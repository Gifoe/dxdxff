#!/usr/bin/env python3
"""Validation-only RCC selection. Never reads held-out summaries."""
from __future__ import annotations
import argparse,json
from pathlib import Path
import pandas as pd

def main():
 p=argparse.ArgumentParser();p.add_argument('--root_dir',required=True);p.add_argument('--output_manifest',required=True);a=p.parse_args();root=Path(a.root_dir)
 audit_path=root/'R0_BASE'/'r0_reproduction_audit.json'
 if not audit_path.is_file(): raise RuntimeError('R0 reproduction audit is required before adapter selection')
 r0=json.loads(audit_path.read_text())
 if r0.get('status') != 'passed': raise RuntimeError('R0 baseline reproduction is not admitted; adapter stage is blocked')
 chosen=None
 for name in ('R2_HYBRID_CALIBRATION','R1_RANK_COVERAGE'):
  path=root/name/'checkpoint_selection_by_epoch.csv'
  if path.is_file() and not pd.read_csv(path).empty: chosen=name;break
 if chosen is None: raise RuntimeError('No validation-complete R1/R2 RCC profile; adapter stage is blocked')
 thresholds=pd.read_csv(root/chosen/'fold_thresholds.csv') if (root/chosen/'fold_thresholds.csv').is_file() else pd.DataFrame()
 checkpoints={str(i):str(root/chosen/f'fold_{i}'/'best_b0_pruned_model.pth') for i in range(1,6)}
 payload={'selected_profile':chosen,'selection_source':'validation_only','per_fold_checkpoint_path':checkpoints,'per_fold_global_threshold':{str(r.outer_fold):float(r.threshold) for _,r in thresholds.iterrows()},'selection_reason':'R2 priority then R1, based only on validation checkpoint files'}
 Path(a.output_manifest).write_text(json.dumps(payload,indent=2),encoding='utf-8')
if __name__=='__main__':main()
