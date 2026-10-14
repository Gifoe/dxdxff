"""Independent frozen Outer audit from private scores; no model/Raw inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from io_utils import sha,write,read
from evaluate_outer import outer_context,checked_cached
from evaluate import METRICS,independent
from bootstrap import compare

def main(a):
 lock,c,core,fac,records,rec,banks,hist,store,sels,binding,out,pub=outer_context(a);fp=out/'FROZEN_BEFORE_OUTER_PRIVATE.json';fr=json.loads(fp.read_text());status=json.loads((pub/'OUTER_STATUS.json').read_text());seal=out/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';ss=json.loads(seal.read_text());assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE' and status['freeze_sha256']==sha(fp) and status['seal_sha256']==sha(seal);assert fr['binding']==binding and fr['selections']==sels and ss['freeze_sha256']==sha(fp) and len(ss['files'])==25
 archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
 with np.load(archive,allow_pickle=True) as z:y=z['y']
 assert int((y==0).sum())==1743 and int((y==1).sum())==5892
 meta=rec['meta'];rows=[];fullscores={};decisions={};total_models=0
 for r in sels:
  arm,fold=r['arm'],r['fold'];p=out/f'scores_private/{arm}_fold{fold}.pkl';assert sha(Path(r['checkpoint']))==r['checkpoint_sha256'] and sha(p)==ss['files'][f'{arm}_{fold}'];q=read(p);ix=q['indices'];checked_cached(p,sha(fp),r['checkpoint_sha256'],ix);assert 'y' not in q;total_models+=1
  if r['bank']:assert sha(Path(r['bank']))==r['bank_sha256']
  per={arm:(q['logit'],r['threshold'])}
  if arm=='D2':
   for name,ll in q['interventions'].items():per['D2_'+name]=(ll,r['threshold'])
   per['D2_auxiliary_head_diagnostic']=(q['raw_logit'],.5)
  for tag,(logit,tau) in per.items():
   score=torch.sigmoid(torch.from_numpy(logit)).numpy()
   if tag not in fullscores:fullscores[tag]=np.full(7635,np.nan);decisions[tag]=np.zeros(7635,bool)
   assert np.isnan(fullscores[tag][ix]).all();fullscores[tag][ix]=score;decisions[tag][ix]=score>=tau
   for pid in sorted(set(meta['patient'][ix])):
    k=np.flatnonzero(meta['patient'][ix]==pid);k=k[np.argsort(meta['channel'][ix][k].astype(str),kind='stable')];ii=ix[k];rows.append(dict(arm=tag,fold=fold,patient=str(pid),center=str(meta['center'][ii[0]]),**independent(y[ii],score[k],tau)))
 df=pd.DataFrame(rows);saved=pd.DataFrame(read(out/'OUTER_METRICS_PRIVATE.pkl'));keys=['arm','fold','patient'];l=df.set_index(keys).sort_index();r=saved.set_index(keys).sort_index();assert l.index.equals(r.index) and np.allclose(l[METRICS],r[METRICS],atol=1e-12,rtol=0,equal_nan=True)
 for arm,g in df.groupby('arm'):assert len(g)==g.patient.nunique()==80 and int(g.channels.sum())==7635
 for name,keys in [('OUTER_SUMMARY.csv',['arm']),('OUTER_BY_FOLD.csv',['arm','fold']),('OUTER_BY_CENTER.csv',['arm','center'])]:
  v=pd.read_csv(pub/name).set_index(keys).sort_index();q=df.groupby(keys)[METRICS].mean().sort_index();assert v.index.equals(q.index) and np.allclose(v[METRICS],q[METRICS],atol=1e-12,rtol=0,equal_nan=True)
 boot=pd.read_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv')
 for left,right in lock['bootstrap']['comparisons']:
  actual=compare(c,df[df.arm==left],df[df.arm==right],METRICS)
  for value in actual:
   q=boot[(boot.comparison==f'{left} minus {right}')&(boot.metric==value['metric'])].iloc[0];assert np.allclose(q[['delta','ci_low','ci_high']].to_numpy(float),[value['delta'],value['ci_low'],value['ci_high']],atol=1e-12,rtol=0,equal_nan=True)
 absolute=pd.read_csv(pub/'OUTER_ABSOLUTE_INTERVALS.csv')
 for arm,g in df.groupby('arm'):
  b=c.cluster_bootstrap(g.patient.to_numpy(),g[METRICS].to_numpy(),10000)
  for j,m in enumerate(METRICS):
   q=absolute[(absolute.arm==arm)&(absolute.metric==m)].iloc[0];assert np.allclose(q[['estimate','ci_low','ci_high']].to_numpy(float),[g[m].mean(),*np.nanpercentile(b[:,j],[2.5,97.5])],atol=1e-12,rtol=0,equal_nan=True)
 pooled=pd.read_csv(pub/'OUTER_POOLED_CHANNELS.csv').set_index('arm')
 for arm in lock['models']:
  scores=fullscores[arm];p=decisions[arm];assert np.isfinite(scores).all();ez=y==0;tp=int((ez&~p).sum());fpv=int((~ez&~p).sum());tn=int((~ez&p).sum());fn=int((ez&p).sum());q=independent(y,scores,.5);ef=2*tp/(2*tp+fpv+fn);nf=2*tn/(2*tn+fpv+fn);se=tp/(tp+fn);sp=tn/(tn+fpv)
  q.update(macro_f1=(ef+nf)/2,ez_f1=ef,nez_f1=nf,sensitivity=se,specificity=sp,balanced_accuracy=(se+sp)/2,accuracy=(tp+tn)/7635,predicted_ez_fraction=float((~p).mean()),TP=tp,FP=fpv,TN=tn,FN=fn);q.pop('ez_mrr');q.pop('top1_is_ez')
  for key,val in q.items():assert np.isclose(pooled.loc[arm,key],val,atol=1e-12,rtol=0,equal_nan=True)
 assert total_models==25;write(pub/'OUTER_INDEPENDENT_VERIFICATION.json',dict(status='PASS',patients_per_arm=80,channels_per_arm=7635,patient_metric_cells=len(df),frozen_model_fold_cells=25,all_dependencies_unchanged=True,independent_sklearn_metrics=True,independent_paired_bootstrap_multiplicity=True,absolute_intervals_recomputed=True,pooled_confusions_and_fold_thresholds_verified=True,seed=42,draws=10000,no_model_or_Raw_rerun=True,no_selection_or_fitting=True,source_sha256=sha(Path(__file__)),aggregate_sha256={p.name:sha(p) for p in pub.glob('*.csv')}));print('FROZEN_OUTER_INDEPENDENT_VERIFICATION_PASS',25,len(df),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
