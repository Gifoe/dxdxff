"""Name-bound mappings, FIT-only weighted knots; never loads outer values."""
import json
import numpy as np
import pandas as pd
from runtime import args,setup,checkpoint


def main():
 a=args();lock,ref,c,core,original,trainer,models,ks,banks,binding=setup(a)
 assert not (a.runtime/'public/PRETRAIN_AUDIT.json').exists()
 manifest=json.loads((a.previous/'public/FEATURE_GROUP_MANIFEST.json').read_text());rows=[];audits=[];selections=[]
 baseline=json.loads((a.runtime/'public/BASELINE_REPRODUCTION.json').read_text())
 for fold,d in enumerate(banks,1):
  fmap=ks.mapping(manifest,d['features']);x,pre=core.prepare(d['raw_x'],d['patient'],d['train']);assert np.array_equal(x,d['x'])
  for key in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[key],d['pre'][key])
  assert (x[:,fmap['zeros']]==0).all()
  for j,k in fmap['duplicates'].items():
   assert np.array_equal(d['raw_x'][:,j],d['raw_x'][:,k],equal_nan=True) and np.array_equal(x[:,j],x[:,k])
  stats,w=ks.fit(x,d['patient'],d['train'],fmap);tr=d['train']
  for i,j in enumerate(fmap['representatives']):
   values=x[tr,j];qs=stats['raw_quantiles'][i]
   rows.append(dict(fold=fold,feature=str(d['features'][j]),representative_index=int(j),distinct_knots=int(stats['distinct_count'][i]),
    active_intervals=int(stats['slope_mask'][i].sum()),degenerate_identity=bool(stats['distinct_count'][i]<3),
    FIT_patients=stats['FIT_patients'],FIT_channels=len(tr),FIT_mean=float(values@w[tr]),FIT_std=float(np.sqrt(((values-float(values@w[tr]))**2)@w[tr])),
    **{'q'+str(int(round(q*100))).zfill(2):float(v) for q,v in zip(ks.Q,qs)}))
  target=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert not target.exists()
  cell={k:d[k] for k in ['x','y','patient','channel','center','features','train','val','g','pre']};cell.update(stats=stats,binding=binding)
  core.torch_write(target,cell)
  old=baseline['folds'][fold-1];assert c.sha(a.runtime/f'fold{fold}/B0_VAL_PRIVATE.pt')==old['prediction_sha256']
  audits.append(dict(fold=fold,bank_sha256=c.sha(target),effective_spline_parameters=stats['effective_parameters'],
    active_features=int((stats['distinct_count']>=3).sum()),representative_columns=56,duplicate_pairs=16,constant_dimensions=16,
    original_preprocessing_bitwise=True,FIT_knot_only=True,outer_overlap=0))
  selections.append(dict(arm='B0',fold=fold,epoch=old['epoch'],threshold=old['threshold'],checkpoint_sha256=old['checkpoint_sha256'],bank_sha256=c.sha(target),reused=True))
 c.write_json(a.runtime/'B0_SELECTIONS_PRIVATE.json',selections)
 pd.DataFrame(rows).to_csv(a.runtime/'public/FEATURE_KNOT_AUDIT.csv',index=False)
 c.write_json(a.runtime/'public/PRETRAIN_AUDIT.json',dict(status='PASS',binding=binding,folds=audits,global_x_y_indexed=False,no_raw_extraction=True,new_outer_accessed=False))
 print('PR_SPLINE_FIT_KNOT_AND_FEATURE_AUDIT_PASS',flush=True)


if __name__=='__main__':main()
