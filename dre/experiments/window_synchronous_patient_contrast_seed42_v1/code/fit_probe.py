"""Fixed LR and strict train/threshold/held-out roles. All predictions seal first."""
import json
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from common import sha,digest,read,save,write_json,patient_weights
from summarize_temporal_features import stage_indices

ARMS=['C0','C1','C2']

def plans(ids,fold):
 ids=np.array(sorted(set(ids)));seed=42+1009*fold;groups=np.array_split(np.random.default_rng(seed).permutation(ids),4);out=[]
 for h,held in enumerate(groups):
  remaining=np.array(sorted(set(ids)-set(held)));order=np.random.default_rng(seed+10000+h).permutation(remaining);n=int(np.ceil(.2*len(order)));select=order[:n];train=order[n:]
  assert set(train).isdisjoint(select) and set(train).isdisjoint(held) and set(select).isdisjoint(held)
  assert set(np.r_[train,select,held])==set(ids);out.append(dict(group=h,train=train.tolist(),select=select.tolist(),held=held.tolist()))
 return out

def weighted_median(x,w):
 order=np.argsort(x,kind='stable');s=np.cumsum(w[order]);return float(x[order[np.searchsorted(s,s[-1]/2,side='left')]])

def fit_preprocessor(x,pids):
 assert x.shape[1]==182;w=patient_weights(pids);median=[]
 for j in range(x.shape[1]):
  ok=np.isfinite(x[:,j]);assert ok.any(),'UNIDENTIFIABLE_TRAIN_IMPUTATION_COORDINATE';median.append(weighted_median(x[ok,j],w[ok]))
 median=np.array(median);filled=np.where(np.isfinite(x),x,median);scaler=StandardScaler().fit(filled,sample_weight=w)
 return dict(median=median,scaler=scaler,train_ids=sorted(set(pids)))

def transform(pre,x):
 assert x.shape[1]==182
 z=pre['scaler'].transform(np.where(np.isfinite(x),x,pre['median']));assert np.isfinite(z).all();return z

def threshold(y,prob,pid):
 grid=np.round(np.arange(0,1.0001,.005),3);pred=prob[:,None]>=grid;mm=[];ef=[];ba=[]
 for p in sorted(set(pid)):
  take=pid==p;z=(y[take]==0)[:,None];q=pred[take]
  tp=(z&q).sum(0);fp=(~z&q).sum(0);tn=(~z&~q).sum(0);fn=(z&~q).sum(0)
  e=np.divide(2.*tp,2*tp+fp+fn,out=np.zeros(201),where=2*tp+fp+fn>0);n=np.divide(2.*tn,2*tn+fp+fn,out=np.zeros(201),where=2*tn+fp+fn>0)
  se=np.divide(tp,tp+fn,out=np.full(201,np.nan),where=tp+fn>0);sp=np.divide(tn,tn+fp,out=np.full(201,np.nan),where=tn+fp>0)
  mm.append((e+n)/2);ef.append(e);ba.append(np.nanmean([se,sp],axis=0))
 means=np.mean(mm,axis=0);ee=np.mean(ef,axis=0);bb=np.nanmean(ba,axis=0)
 key=max((float(means[i]),float(ee[i]),float(bb[i]),-abs(float(t)-.5),-float(t),float(t)) for i,t in enumerate(grid))
 return dict(threshold=key[-1],macro_f1=key[0],ez_f1=key[1],balanced_accuracy=key[2])

def fit(a,lock,binding):
 import torch  # Only FIT-only label artifact is deserialized; no full FIT/VAL bank.
 bank=read(a.runtime/'CONTEXT_TABLES_PRIVATE.pkl');assert bank['binding']==binding and bank['control']['status']=='PASS'
 frozen=a.runtime/'PROBE_FREEZE_PRIVATE.json'
 if frozen.exists():assert json.loads(frozen.read_text())['binding']==binding;return
 selections=[];labels_by_fold={};allfiles={};preaudit=[];allplans={};labelhash={};resources=[]
 for fold in range(1,6):
  roles=bank['folds'][fold];path=Path(r'C:\pr_uncertainty_aware_supervision_seed42_runtime')/f'fold{fold}/OOF_FROZEN_PRIVATE.pt';labelhash[fold]=sha(path);q=torch.load(path,map_location='cpu',weights_only=False)
  pid=np.asarray(q['patient']).astype(str);channel=np.asarray(q['channel']).astype(str);assert set(pid)==roles['fit'] and not set(pid)&(roles['validation']|roles['test'])
  for p in set(pid):assert set(channel[pid==p])==set(bank['tables'][p]['channel'])
  y=np.asarray(q['y_nez']);assert set(y)<={0,1}
  assert len(set(zip(pid,channel)))==len(pid)
  index=np.lexsort((channel,pid));pid=pid[index];channel=channel[index];y=y[index];tables={arm:[] for arm in ARMS};centers=[];record_count=[];channel_count=[]
  for p,ch in zip(pid,channel):
   b=bank['tables'][p];matches=np.flatnonzero(b['channel']==ch);assert len(matches)==1;j=int(matches[0]);centers.append(b['center']);record_count.append(b['records']);channel_count.append(len(b['channel']))
   for arm in ARMS:tables[arm].append(b[arm][j])
  tables={arm:np.stack(v) for arm,v in tables.items()};assert all(x.shape==(len(pid),182) for x in tables.values())
  assert np.array_equal(tables['C0'][:,:98],tables['C1'][:,:98],equal_nan=True) and np.array_equal(tables['C1'][:,:98],tables['C2'][:,:98],equal_nan=True);assert np.array_equal(np.isfinite(tables['C1']),np.isfinite(tables['C2']))
  labels_by_fold[fold]=dict(patient=pid,channel=channel,y=y,center=np.array(centers),records=np.array(record_count),channel_count=np.array(channel_count));allplans[fold]=plans(roles['fit'],fold);seen=[]
  for plan in allplans[fold]:
   ti=np.flatnonzero(np.isin(pid,plan['train']));vi=np.flatnonzero(np.isin(pid,plan['select']));hi=np.flatnonzero(np.isin(pid,plan['held']));seen.extend(plan['held']);selfpre=None
   assert not (set(plan['train']+plan['select']+plan['held'])&(roles['validation']|roles['test']))
   for arm in ARMS:
    cell=a.runtime/f'fold{fold}/inner{plan["group"]}/{arm}_PRIVATE.pkl'
    if cell.exists():result=read(cell);assert result['binding']==binding and result['plan']==plan
    else:
     t0=__import__('time').perf_counter();pre=fit_preprocessor(tables[arm][ti],pid[ti]);x=transform(pre,tables[arm][ti]);m=LogisticRegression(penalty='l2',C=1.,solver='lbfgs',max_iter=1000,random_state=42,tol=1e-4,class_weight=None)
     with warnings.catch_warnings(),threadpool_limits(limits=2):
      warnings.simplefilter('error',ConvergenceWarning);m.fit(x,(y[ti]==0).astype(int),sample_weight=patient_weights(pid[ti]))
     assert m.n_iter_.max()<1000 and m.n_features_in_==182 and m.classes_.tolist()==[0,1]
     ps=m.predict_proba(transform(pre,tables[arm][vi]))[:,1];sel=threshold(y[vi],ps,pid[vi]);tau=sel['threshold']
     # Freeze numeric tau before any held-out clinical label is scored.
     pe=m.predict_proba(transform(pre,tables[arm][hi]))[:,1];interventions={};extra={}
     if arm=='C1':
      for stage in ['early','later']:
       changed=tables[arm][hi].copy();jj=stage_indices(stage);changed[:,jj]=tables['C2'][hi][:,jj];interventions[stage+'_shift']=m.predict_proba(transform(pre,changed))[:,1]
      z=transform(pre,tables[arm][hi]);extra['context_logit']=z[:,98:]@m.coef_[0,98:];neutral=z.copy();neutral[:,98:]=0;interventions['context_neutral']=m.predict_proba(neutral)[:,1]
     result=dict(binding=binding,plan=plan,indices=hi,scores=pe,threshold=tau,selection=sel,pre=pre,model=m,interventions=interventions,extra=extra,own_selection_indices=vi,own_selection_scores=ps,training_ids_hash=digest(plan['train']),selection_ids_hash=digest(plan['select']),evaluation_ids_hash=digest(plan['held']),wall_seconds=__import__('time').perf_counter()-t0)
     save(cell,result)
    pre=result['pre'];assert set(pre['train_ids'])==set(plan['train'])
    vals=np.r_[pre['median'][:98],pre['scaler'].mean_[:98],pre['scaler'].scale_[:98]]
    if selfpre is None:selfpre=vals
    else:assert np.array_equal(selfpre,vals)
    if arm=='C0':assert np.all(pre['scaler'].transform(np.where(np.isfinite(tables[arm][ti]),tables[arm][ti],pre['median']))[:,98:]==0) and np.all(result['model'].coef_[0,98:]==0)
    selections.append(dict(fold=fold,inner=plan['group'],arm=arm,train_patients=len(plan['train']),threshold_patients=len(plan['select']),evaluation_patients=len(plan['held']),threshold=result['threshold'],iterations=int(result['model'].n_iter_.max()),input_width=182,model_file_sha256=sha(cell)));allfiles[f'{fold}_{plan["group"]}_{arm}']=sha(cell)
    preaudit.append(dict(fold=fold,inner=plan['group'],arm=arm,TRAIN_only=True,train_selection_overlap=0,train_evaluation_overlap=0,selection_evaluation_overlap=0,original_VAL_TEST_overlap=0,input_width=182,fit_weight_per_patient=1.,center_or_id_feature=False))
    resources.append(dict(fold=fold,inner=plan['group'],arm=arm,seconds=result['wall_seconds'],iterations=int(result['model'].n_iter_.max())));print('FIXED_PROBE_COMPLETE',fold,plan['group'],arm,len(selections),'of60',flush=True)
   write_json(a.pub/'RUN_STATUS.json',dict(status='FIT_ONLY_PROBES',probe_models_fitted=len(selections),total=60,no_Development_or_outer_evaluation=True))
  assert len(seen)==len(set(seen))==len(roles['fit']) and set(seen)==roles['fit']
 save(a.runtime/'FIT_LABELS_PRIVATE.pkl',dict(binding=binding,by_fold=labels_by_fold,label_source_hashes=labelhash))
 write_json(frozen,dict(status='PASS',binding=binding,files=allfiles,plans=allplans,label_source_hashes=labelhash,all_thresholds_before_inner_evaluation=True,no_Development_or_outer_evaluation=True))
 pd.DataFrame(selections).to_csv(a.pub/'PROBE_SELECTION.csv',index=False);pd.DataFrame(preaudit).to_csv(a.pub/'LEAKAGE_AND_PREPROCESSING_AUDIT.csv',index=False);pd.DataFrame(resources).to_csv(a.pub/'RESOURCE_AUDIT.csv',index=False)
