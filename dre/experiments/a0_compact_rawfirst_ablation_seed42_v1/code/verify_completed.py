"""Independent sealed-score verifier, stage-specific; no model forward or tuning."""
import argparse
import json
import sys
import numpy as np
import pandas as pd
import torch
from audit_sources import arguments,context,historical,metadata
from training import path
from metrics import independent
from bootstrap import independent as independent_bootstrap


def verify(a,raw):
 o=context(a);lock,ref,c,core,original,trainer,helper,banks,manifest,bind=o;stage='RAW' if raw else 'FEATURE';pub=a.runtime/'public'
 status=json.loads((pub/f'{stage}_OUTER_STATUS.json').read_text());assert status['status']=='ONE_FROZEN_'+stage+'_OUTER_PASS_COMPLETE'
 fs=json.loads((a.runtime/f'{stage}_FROZEN_BEFORE_OUTER.json').read_text());assert fs['binding']==bind
 seal=json.loads((a.runtime/f'{stage}_OUTER_PREDICTION_SEAL.json').read_text());assert seal['binding']==c.sha(a.runtime/f'{stage}_FROZEN_BEFORE_OUTER.json')
 recs=fs['selections'];new=['R0'] if raw else ['F1','F2','F3','F4','F5'];refs=['F0'] if raw else ['F0','F6','F7']
 meta=metadata(a)
 with np.load(a.features,allow_pickle=True) as f:y=f['y']
 total=0;intervals=0;drift=0.
 for phase,clusters,appearances in [('DEVELOPMENT',47,65),('OUTER',80,80)]:
  rows=[]
  for arm in refs+new:
   visited=[]
   for fold in range(1,6):
    if arm in refs:
     bp,cp,pp,r=historical(a,arm,fold)
     admission=json.loads((pub/'BASELINE_REPRODUCTION.json').read_text());ad=next(v for v in admission['selections'] if v['arm']==arm and v['fold']==fold)
     assert c.sha(cp)==ad['checkpoint_sha256'] and c.sha(bp)==ad['bank_sha256'] and c.sha(pp)==ad['outer_prediction_sha256']
     if phase=='DEVELOPMENT':pp=a.runtime/f'historical/{arm}_fold{fold}_VAL_PRIVATE.pt';assert c.sha(pp)==ad['validation_replay_sha256']
     threshold=r['threshold']
    else:
     r=next(q for q in recs if q['arm']==arm and q['fold']==fold);cp=path(a,arm,fold)/'BEST_PRIVATE.pt'
     bp=a.original/f'fold{fold}/BANK_PRIVATE.pt' if raw else a.runtime/f'features/{arm}/fold{fold}/BANK_PRIVATE.pt'
     assert c.sha(cp)==r['checkpoint_sha256'] and c.sha(bp)==r['bank_sha256']
     ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold']
     assert sum(v.numel() for v in ck['model'].values())==r['parameters'] and ck['binding']['penalty']==0 and ck['binding']['seed']==42+1009*fold
     pp=a.runtime/f'{stage}_scores/{arm}_fold{fold}_{"VAL" if phase=="DEVELOPMENT" else "OUTER"}_PRIVATE.pt'
     assert c.sha(pp)==(fs['hashes'][f'{arm}_validation{fold}'] if phase=='DEVELOPMENT' else seal['files'][f'{arm}_{fold}']);threshold=r['threshold']
     if not raw:
      d=torch.load(bp,map_location='cpu',weights_only=False);spec=lock['matrix'][arm];mask=np.asarray([q['view'] in spec['views'] for q in manifest['columns']])
      assert np.array_equal(mask,d['mask']) and mask.sum()==spec['retained'] and np.all(d['x'][:,~mask]==0)
      if arm!='F1':assert np.array_equal(d['x'][:,mask],banks[fold-1]['x'][:,mask])
    cell=torch.load(pp,map_location='cpu',weights_only=False);ix=cell['indices'];visited.extend(ix.tolist());assert cell['threshold']==threshold
    d=banks[fold-1] if phase=='DEVELOPMENT' else dict(meta,y=y)
    if phase=='DEVELOPMENT':assert np.array_equal(ix,d['val'])
    for pid in sorted(set(d['patient'][ix])):
     k=np.flatnonzero(d['patient'][ix]==pid);k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')];ii=ix[k]
     im=independent(d['y'][ii],cell['scores'][k],threshold);cm=c.metric(d['y'][ii],cell['scores'][k],threshold)
     for name in im:
      assert np.isclose(im[name],cm[name],atol=1e-12,rtol=0,equal_nan=True)
      if np.isfinite(im[name]):drift=max(drift,abs(im[name]-cm[name]))
     rows.append(dict(arm=arm,patient=pid,fold=fold,center=str(d['center'][ii[0]]),**im));total+=1
   if phase=='OUTER':assert sorted(visited)==list(range(7635))
  df=pd.DataFrame(rows);metrics=lock['metrics']
  for arm,q in df.groupby('arm'):assert (len(q),q.patient.nunique())==(appearances,clusters)
  for filename,keys in [(f'{stage}_{phase}_SUMMARY.csv',['arm']),(f'{stage}_{phase}_BY_FOLD.csv',['arm','fold']),(f'{stage}_{phase}_BY_CENTER.csv',['arm','center'])]:
   actual=pd.read_csv(pub/filename).set_index(keys).sort_index();expect=df.groupby(keys)[metrics].mean().sort_index()
   assert actual.index.equals(expect.index) and np.allclose(actual[metrics],expect,atol=1e-12,rtol=0,equal_nan=True)
  if phase=='DEVELOPMENT':
   actual=pd.read_csv(pub/f'{stage}_DEVELOPMENT_UNIQUE_ID_SUMMARY.csv').set_index('arm').sort_index();expect=df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().sort_index();assert np.allclose(actual[metrics],expect,atol=1e-12,rtol=0,equal_nan=True)
  actual=pd.read_csv(pub/f'{stage}_{phase}_BOOTSTRAP.csv')
  for arm in refs+new:
   if arm=='F0':continue
   calc=independent_bootstrap(df[df.arm=='F0'],df[df.arm==arm],metrics)
   for m,vals in calc.items():
    r=actual[(actual.comparison=='F0 minus '+arm)&(actual.metric==m)].iloc[0];assert np.allclose(vals,[r.delta,r.ci_low,r.ci_high],atol=1e-12,rtol=0,equal_nan=True);intervals+=1
 assert context(a)[-1]==bind
 c.write_json(pub/f'{stage}_INDEPENDENT_VERIFICATION.json',dict(status='PASS',metric_cells=total,verified_intervals=intervals,max_metric_drift=drift,
  original_sources_banks_checkpoints_thresholds_score_hashes_verified=True,all_primary_fold_center_unique_ID_tables_verified=True,no_new_model_inference=True,bootstrap_draws=10000,
  numpy=np.__version__,torch=torch.__version__,stage=stage,scientific_status=lock['scientific_status']))
 aliases={'RAW_OUTER_SUMMARY.csv':'RAW_ONLY_OUTER_SUMMARY.csv','RAW_OUTER_BOOTSTRAP.csv':'RAW_ONLY_BOOTSTRAP.csv','RAW_OUTER_BY_FOLD.csv':'RAW_ONLY_BY_FOLD.csv','RAW_OUTER_BY_CENTER.csv':'RAW_ONLY_BY_CENTER.csv'} if raw else {
  'FEATURE_OUTER_SUMMARY.csv':'OUTER_ABLATION_SUMMARY.csv','FEATURE_OUTER_BOOTSTRAP.csv':'PAIRED_BOOTSTRAP.csv','FEATURE_OUTER_BY_FOLD.csv':'OUTER_BY_FOLD.csv','FEATURE_OUTER_BY_CENTER.csv':'OUTER_BY_CENTER.csv','FEATURE_TRAINING_HISTORY.csv':'TRAINING_HISTORY.csv','FEATURE_MODEL_SELECTION.csv':'MODEL_SELECTION.csv'}
 for src,dst in aliases.items():pd.read_csv(pub/src).to_csv(pub/dst,index=False)
 c.write_json(pub/'RUN_STATUS.json',dict(status='RAW_ONLY_EVALUATION_COMPLETE_ABLATIONS_PENDING' if raw else 'COMPLETE_A0_COMPACT_RAW_FIRST_ABLATION_OUTER_EXPLORATORY',raw_complete=True,feature_ablations_complete=not raw))
 print(stage+'_INDEPENDENT_VERIFICATION_PASS',flush=True)


if __name__=='__main__':
 stage=sys.argv.pop(1);assert stage in ['raw','feature'];verify(arguments(),raw=stage=='raw')
