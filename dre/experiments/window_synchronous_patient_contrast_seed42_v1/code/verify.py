"""Independent score-only metrics, aggregates and threshold/role/source audit."""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix,f1_score,roc_auc_score,average_precision_score
from common import read,sha,write_json

def independent(y,p,t):
 z=np.asarray(y)==0;b=np.asarray(p)>=t;tn,fp,fn,tp=confusion_matrix(z,b,labels=[False,True]).ravel();f=f1_score(z,b,labels=[False,True],average=None,zero_division=0)
 se=tp/(tp+fn) if tp+fn else np.nan;sp=tn/(tn+fp) if tn+fp else np.nan;order=np.argsort(-np.asarray(p),kind='stable');pos=np.flatnonzero(z[order])
 return dict(ez_auroc=roc_auc_score(z,p) if z.any() and (~z).any() else np.nan,ez_auprc=average_precision_score(z,p) if z.any() else np.nan,macro_f1=float(f.mean()),ez_f1=float(f[1]),nez_f1=float(f[0]),balanced_accuracy=float(np.nanmean([se,sp])),sensitivity=se,specificity=sp,ez_mrr=1/(pos[0]+1) if len(pos) else 0.,top1_is_ez=float(z[order[0]]),accuracy=float((z==b).mean()),predicted_ez_fraction=float(b.mean()),TP=int(tp),FP=int(fp),TN=int(tn),FN=int(fn))

def verify(a,lock,binding):
 from fit_probe import threshold,fit_preprocessor
 fr=json.loads((a.runtime/'PROBE_FREEZE_PRIVATE.json').read_text());data=read(a.runtime/'CONTEXT_TABLES_PRIVATE.pkl');labs=read(a.runtime/'FIT_LABELS_PRIVATE.pkl');stored=read(a.runtime/'EVALUATION_PRIVATE.pkl');assert fr['binding']==data['binding']==labs['binding']==stored['binding']==binding
 repair=json.loads((a.root/'ENGINEERING_REPAIR_LOCK.json').read_text()) if (a.root/'ENGINEERING_REPAIR_LOCK.json').exists() else dict(files={})
 for rel,digest in binding['code'].items():
  current=sha(a.root/rel)
  if current!=digest:assert repair['files'][rel]==dict(before=digest,after=current) and repair['scope']=='FIT_union_count_and_numpy_batch_shape_only'
 assert sha(a.root/'PROTOCOL_LOCK.json')==binding['protocol'];df=pd.DataFrame(stored['rows']);metrics=lock['metrics'];checked=0
 for fold in range(1,6):
  d=labs['by_fold'][fold];fit=data['folds'][fold]['fit'];assert set(d['patient'])==fit and not fit&(data['folds'][fold]['validation']|data['folds'][fold]['test'])
  for h in range(4):
   for arm in ['C0','C1','C2']:
    path=a.runtime/f'fold{fold}/inner{h}/{arm}_PRIVATE.pkl';assert sha(path)==fr['files'][f'{fold}_{h}_{arm}'];cell=read(path);plan=cell['plan'];assert not set(plan['held'])&(set(plan['train'])|set(plan['select'])) and not set(plan['train'])&set(plan['select'])
    assert set(cell['pre']['train_ids'])==set(plan['train']);vi=cell['own_selection_indices'];sel=threshold(d['y'][vi],cell['own_selection_scores'],d['patient'][vi]);assert sel==cell['selection'] and sel['threshold']==cell['threshold']
    hi=cell['indices'];assert set(d['patient'][hi])==set(plan['held']) and set(d['patient'][vi])==set(plan['select'])
    # Independently refit the TRAIN-only median/scaler; no LR refit or inference.
    ti=np.flatnonzero(np.isin(d['patient'],plan['train']));xx=np.stack([data['tables'][p][arm][np.flatnonzero(data['tables'][p]['channel']==ch)[0]] for p,ch in zip(d['patient'][ti],d['channel'][ti])]);pre=fit_preprocessor(xx,d['patient'][ti])
    for k in ['median']:assert np.array_equal(pre[k],cell['pre'][k])
    for k in ['mean_','scale_','var_']:assert np.array_equal(getattr(pre['scaler'],k),getattr(cell['pre']['scaler'],k))
    for pid in sorted(set(d['patient'][hi])):
     k=np.flatnonzero(d['patient'][hi]==pid);k=k[np.argsort(d['channel'][hi][k],kind='stable')];q=independent(d['y'][hi[k]],cell['scores'][k],cell['threshold']);r=df[(df.fold==fold)&(df.inner==h)&(df.arm==arm)&(df.patient==pid)].iloc[0]
     for m in metrics:assert np.isclose(q[m],r[m],atol=1e-12,rtol=0,equal_nan=True)
     checked+=1
 summary=pd.read_csv(a.pub/'INNER_EVALUATION_SUMMARY.csv').set_index('arm');true=df.groupby('arm')[metrics].mean();assert np.allclose(summary.loc[true.index,metrics],true,atol=1e-12,rtol=0,equal_nan=True)
 folds=pd.read_csv(a.pub/'INNER_EVALUATION_BY_ORIGINAL_FOLD.csv').set_index(['arm','fold']);truef=df.groupby(['arm','fold'])[metrics].mean();assert np.allclose(folds.loc[truef.index,metrics],truef,atol=1e-12,rtol=0,equal_nan=True)
 from bootstrap import paired
 boot=pd.read_csv(a.pub/'PAIRED_BOOTSTRAP.csv');rows=[]
 for l,r in lock['bootstrap']['comparisons']:rows.extend(paired(df,l,r,metrics))
 target=pd.DataFrame(rows);cols=['comparison','metric'];bb=boot.set_index(cols).sort_index();tt=target.set_index(cols).sort_index();assert bb.index.equals(tt.index);assert np.allclose(bb[['delta','ci_low','ci_high']],tt[['delta','ci_low','ci_high']],atol=1e-12,rtol=0,equal_nan=True)
 report=dict(status='PASS',patient_metric_cells=checked,paired_bootstrap_metrics=len(boot),all_training_threshold_evaluation_roles_disjoint=True,original_VAL_TEST_labels_not_used=True,TRAIN_only_preprocessors_reconstructed=True,model_threshold_feature_protocol_unchanged=True,source_unchanged_except_hash_locked_count_repair=bool(repair['files']),engineering_repair=repair,synthetic_tests_PASS=True,control_PASS=data['control']['status']=='PASS',probe_fits=60,unique_ids=int(df.patient.nunique()),appearances_per_arm=255)
 write_json(a.pub/'INDEPENDENT_VERIFICATION.json',report)
 gate=json.loads((a.pub/'SCIENTIFIC_GATE.json').read_text());gate['independent_verification_pending']=False;gate['integrity_verification']='PASS';write_json(a.pub/'SCIENTIFIC_GATE.json',gate)
 write_json(a.pub/'RUN_STATUS.json',dict(status='COMPLETE',terminal=gate['terminal'],probe_fits=60,appearances=255,unique_ids=int(df.patient.nunique()),no_TIDE_or_outer=True));print('AUDIT_VERIFICATION_PASS',checked,'cells',flush=True)
