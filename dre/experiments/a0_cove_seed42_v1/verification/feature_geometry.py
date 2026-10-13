"""Label-free cache magnitude and OOF/full-FIT drift; no feature selection or inference."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from runtime import args,setup
from conditional_ridge import residuals,weights,mean,sd

def main():
 a=args();o=setup(a);geometry=[];drift=[]
 for fold,d in enumerate(o.banks,1):
  b=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',weights_only=False)
  for phase,ix in [('FIT',d['train']),('VALIDATION',d['val'])]:
   w=weights(d['patient'][ix])
   for j,q in enumerate(b['mapping']['triples']):
    for v in ['abs','delta','ratio']:
     raw=d['raw_x'][ix,q[v]].astype(float);valid=np.isfinite(raw);wv=w[valid];z=raw[valid];xx=b['x'][ix,q[v]].astype(float)
     geometry.append(dict(fold=fold,phase=phase,group=j,feature=q['feature'],aggregation=q['aggregation'],view=v,finite_fraction=float(mean(valid.astype(float),w)),raw_weighted_mean=float(mean(z,wv)),raw_weighted_std=float(sd(z,wv)),raw_median_abs=float(np.median(np.abs(z))),raw_q95_abs=float(np.quantile(np.abs(z),.95)),raw_max_abs=float(np.max(np.abs(z))),post_preprocess_weighted_std=float(sd(xx,w)),original_scaler=float(d['pre']['scale'][q[v]])))
  tr=b['train'];va=b['val'];r=residuals(b['x'][tr],b['stats']['full'],b['mapping']);w=weights(b['patient'][tr]);vw=weights(b['patient'][va])
  for j,q in enumerate(b['mapping']['triples']):
   for v,name in [(0,'DELTA'),(1,'RATIO')]:
    full_sd=float(sd(r[:,j,v],w));oof_sd=float(b['stats']['raw_std'][j,v]);val_sd=float(sd(b['e'][va,j,v]*b['stats']['scale'][j,v],vw))
    drift.append(dict(fold=fold,group=j,feature=q['feature'],aggregation=q['aggregation'],residual=name,OOF_FIT_std=oof_sd,fullFIT_on_FIT_std=full_sd,fullFIT_on_validation_std=val_sd,fullFIT_to_OOF_FIT_ratio=full_sd/max(oof_sd,1e-6),validation_to_fullFIT_on_FIT_ratio=val_sd/max(full_sd,1e-6)))
 pd.DataFrame(geometry).to_csv(o.pub/'FEATURE_GEOMETRY.csv',index=False);pd.DataFrame(drift).to_csv(o.pub/'OOF_FULLFIT_DRIFT.csv',index=False)
 o.c.write_json(o.pub/'FEATURE_GEOMETRY_AUDIT.json',dict(status='COMPLETE_DESCRIPTIVE_ONLY',legal_FIT_VAL_rows_only=True,no_outer_access=True,no_labels_used=True,no_model_inference=True,no_new_selection=True,script_sha256=o.c.sha(Path(__file__)),geometry_rows=len(geometry),drift_rows=len(drift)))
 print('LABEL_FREE_FEATURE_GEOMETRY_COMPLETE',len(geometry),len(drift),flush=True)

if __name__=='__main__':main()
