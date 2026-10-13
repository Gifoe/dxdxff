"""Descriptive incidence of the EXISTING Patient-Z SD>1e-8 rule; no new cutoff."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from runtime import args,setup

def main():
 a=args();o=setup(a);rows=[]
 for fold,d in enumerate(o.banks,1):
  b=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',weights_only=False)
  for phase,ix in [('FIT',d['train']),('VALIDATION',d['val'])]:
   ids=sorted(set(d['patient'][ix]))
   for q in b['mapping']['triples']:
    for v in ['abs','delta','ratio']:
     s=np.array([np.nanstd(d['raw_x'][ix[d['patient'][ix]==p],q[v]].astype(float)) for p in ids])
     rows.append(dict(fold=fold,phase=phase,group=q['group'],feature=q['feature'],aggregation=q['aggregation'],view=v,patients=len(ids),existing_patientZ_floor=1e-8,patient_fraction_using_denominator_one=float(np.mean(s<=1e-8)),median_patient_raw_SD=float(np.median(s)),min_patient_raw_SD=float(s.min()),max_patient_raw_SD=float(s.max()),normalization_not_changed=True))
 pd.DataFrame(rows).to_csv(o.pub/'PATIENT_Z_FLOOR_AUDIT.csv',index=False)
 o.c.write_json(o.pub/'PATIENT_Z_FLOOR_AUDIT.json',dict(status='DESCRIPTIVE_ONLY',rows=len(rows),cutoff_from_pinned_original_core=True,no_labels_or_outer_access=True,no_feature_selection_or_model_change=True,script_sha256=o.c.sha(Path(__file__))))
 print('EXISTING_PATIENT_Z_FLOOR_AUDITED',len(rows),flush=True)

if __name__=='__main__':main()
