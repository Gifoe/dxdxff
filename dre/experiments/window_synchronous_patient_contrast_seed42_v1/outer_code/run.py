"""Explicit user-authorized final FIT split, fifteen LR freezes, one Outer pass."""
import argparse
import json
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/'code'))
from common import sha,read,save,write_json,patient_weights,metric
from fit_probe import fit_preprocessor,transform,threshold
from bootstrap import paired
from verify import independent

ARMS=['C0','C1','C2']

def plan(ids,fold):
 ids=np.array(sorted(ids));order=np.random.default_rng(42+1009*fold+90000).permutation(ids);n=int(np.ceil(.2*len(ids)))
 return dict(train=order[n:].tolist(),select=order[:n].tolist())

def context(a):
 lock=json.loads((ROOT/'OUTER_PROTOCOL_LOCK.json').read_text(encoding='utf-8'));inner=read(a.inner/'CONTEXT_TABLES_PRIVATE.pkl');labs=read(a.inner/'FIT_LABELS_PRIVATE.pkl');original=inner['binding']
 assert sha(ROOT/'PROTOCOL_LOCK.json')==lock['original_protocol_sha256']==original['protocol']
 assert json.loads((a.inner/'results/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((a.inner/'results/SCIENTIFIC_GATE.json').read_text())['pass_all']
 repair=json.loads((ROOT/'ENGINEERING_REPAIR_LOCK.json').read_text());deps={}
 for rel,old in original['code'].items():
  path=ROOT/rel;now=sha(path)
  if now!=old:assert repair['files'][rel]==dict(before=old,after=now)
  deps['parent_'+rel]=now
 for path in sorted((ROOT/'outer_code').glob('*.py')):deps['outer_code_'+path.name]=sha(path)
 for path in sorted((ROOT/'outer_tests').glob('*.py')):deps['outer_test_'+path.name]=sha(path)
 for name in ['OUTER_PROTOCOL_LOCK.json','PROTOCOL_LOCK.json','ENGINEERING_REPAIR_LOCK.json']:deps[name]=sha(ROOT/name)
 for name in ['CONTEXT_TABLES_PRIVATE.pkl','FIT_LABELS_PRIVATE.pkl','PROBE_FREEZE_PRIVATE.json','INPUT_FREEZE_PRIVATE.json']:deps['private_parent_'+name]=sha(a.inner/name)
 with np.load(a.archive,allow_pickle=True) as f:meta={k:f[k].astype(str) for k in ['patient','channel','center']}
 assert sha(a.archive)==lock['feature_archive_sha256']==original['inputs']['archive'];deps['feature_archive']=sha(a.archive)
 assert len(meta['patient'])==7635 and len(set(zip(meta['patient'],meta['channel'])))==7635 and len(set(meta['patient']))==80
 seen=[]
 for fold,roles in inner['folds'].items():
  assert set(labs['by_fold'][fold]['patient'])==roles['fit']
  assert all(roles[u].isdisjoint(roles[v]) for u,v in [('fit','validation'),('fit','test'),('validation','test')]);seen.extend(roles['test'])
 assert len(seen)==len(set(seen))==80
 a.pub=a.runtime/'results';a.pub.mkdir(parents=True,exist_ok=True)
 return lock,inner,labs,meta,deps

def matrix(inner,pid,channel,arm):
 rows=[]
 for p,ch in zip(pid,channel):
  q=inner['tables'][p];ix=np.flatnonzero(q['channel']==ch);assert len(ix)==1;rows.append(q[arm][ix[0]])
 return np.stack(rows)

def train(a,c):
 lock,inner,labs,meta,deps=c;freeze=a.runtime/'FROZEN_BEFORE_OUTER.json'
 if freeze.exists():assert json.loads(freeze.read_text())['dependencies']==deps;return
 files={};selections=[]
 for fold in range(1,6):
  d=labs['by_fold'][fold];p=plan(inner['folds'][fold]['fit'],fold);assert set(p['train']).isdisjoint(p['select']);assert set(p['train']+p['select'])==inner['folds'][fold]['fit'];assert not set(p['train']+p['select'])&(inner['folds'][fold]['validation']|inner['folds'][fold]['test'])
  ti=np.flatnonzero(np.isin(d['patient'],p['train']));vi=np.flatnonzero(np.isin(d['patient'],p['select']));shared=None
  for arm in ARMS:
   path=a.runtime/f'fold{fold}/{arm}_FINAL_PRIVATE.pkl'
   if path.exists():cell=read(path);assert cell['dependencies']==deps and cell['plan']==p
   else:
    x=matrix(inner,d['patient'],d['channel'],arm);pre=fit_preprocessor(x[ti],d['patient'][ti]);model=LogisticRegression(penalty='l2',C=1.,solver='lbfgs',max_iter=1000,tol=1e-4,random_state=42,class_weight=None)
    with warnings.catch_warnings():warnings.simplefilter('error',ConvergenceWarning);model.fit(transform(pre,x[ti]),(d['y'][ti]==0).astype(int),sample_weight=patient_weights(d['patient'][ti]))
    ss=model.predict_proba(transform(pre,x[vi]))[:,1];choice=threshold(d['y'][vi],ss,d['patient'][vi]);cell=dict(dependencies=deps,plan=p,model=model,pre=pre,selection=choice,threshold=choice['threshold'],selection_indices=vi,selection_scores=ss)
    assert model.n_iter_.max()<1000;save(path,cell)
   values=np.r_[cell['pre']['median'][:98],cell['pre']['scaler'].mean_[:98],cell['pre']['scaler'].scale_[:98]]
   if shared is None:shared=values
   else:assert np.array_equal(shared,values)
   if arm=='C0':assert np.all(cell['model'].coef_[0,98:]==0)
   files[f'{fold}_{arm}']=sha(path);selections.append(dict(fold=fold,arm=arm,train_patients=len(p['train']),threshold_patients=len(p['select']),threshold=cell['threshold'],iterations=int(cell['model'].n_iter_.max()),width=182,model_sha256=sha(path)));print('OUTER_FINAL_PROBE_FROZEN',fold,arm,len(files),'of15',flush=True)
 freeze_data=dict(status='PASS',dependencies=deps,files=files,selections=selections,all15_frozen_before_outer_inference=True,outer_clinical_labels_unread=True,original_Development_labels_unused=True)
 write_json(freeze,freeze_data);pd.DataFrame(selections).to_csv(a.pub/'FINAL_PROBE_SELECTION.csv',index=False);write_json(a.pub/'FREEZE_AUDIT.json',freeze_data)

def evaluate(a,c):
 lock,inner,labs,meta,deps=c;done=a.pub/'OUTER_STATUS.json'
 if done.exists():assert json.loads(done.read_text())['status']=='COMPLETE';return
 freeze=a.runtime/'FROZEN_BEFORE_OUTER.json';fr=json.loads(freeze.read_text());assert fr['dependencies']==deps and len(fr['files'])==15;binding=sha(freeze);files={};covered=[]
 for fold in range(1,6):
  ix=np.flatnonzero(np.isin(meta['patient'],list(inner['folds'][fold]['test'])));assert len(ix)==lock['test_channels_by_fold'][fold-1] and len(set(meta['patient'][ix]))==lock['test_patients_by_fold'][fold-1];covered.extend(ix.tolist())
  for arm in ARMS:
   modelpath=a.runtime/f'fold{fold}/{arm}_FINAL_PRIVATE.pkl';assert sha(modelpath)==fr['files'][f'{fold}_{arm}'];cell=read(modelpath);out=a.runtime/f'fold{fold}/{arm}_OUTER_PREDICTIONS_PRIVATE.pkl'
   if out.exists():q=read(out);assert q['freeze']==binding and np.array_equal(q['indices'],ix)
   else:
    x=matrix(inner,meta['patient'][ix],meta['channel'][ix],arm);score=cell['model'].predict_proba(transform(cell['pre'],x))[:,1];q=dict(freeze=binding,indices=ix,scores=score,threshold=cell['threshold']);save(out,q)
   assert 'y' not in q and np.isfinite(q['scores']).all();files[f'{fold}_{arm}']=sha(out)
 assert sorted(covered)==list(range(7635));seal=a.runtime/'OUTER_PREDICTIONS_SEAL.json';seal_data=dict(freeze=binding,files=files,all15_scores_frozen_before_clinical_labels=True)
 if seal.exists():assert json.loads(seal.read_text())==seal_data
 else:write_json(seal,seal_data)
 # Newly authorized Outer union is all80 IDs. This member is first accessed
 # only after15-model and15-score seals, exclusively for corresponding TEST.
 with np.load(a.archive,allow_pickle=True) as f:y=f['y']
 assert y.shape==(7635,) and (y==0).sum()==1743 and (y==1).sum()==5892
 rows=[];all_decisions={arm:np.zeros(7635,bool) for arm in ARMS};all_scores={arm:np.zeros(7635) for arm in ARMS}
 for fold in range(1,6):
  for arm in ARMS:
   path=a.runtime/f'fold{fold}/{arm}_OUTER_PREDICTIONS_PRIVATE.pkl';assert sha(path)==files[f'{fold}_{arm}'];q=read(path);ix=q['indices'];all_scores[arm][ix]=q['scores'];all_decisions[arm][ix]=q['scores']>=q['threshold']
   for pid in sorted(set(meta['patient'][ix])):
    k=np.flatnonzero(meta['patient'][ix]==pid);k=k[np.argsort(meta['channel'][ix][k],kind='stable')];ii=ix[k];v=metric(y[ii],q['scores'][k],q['threshold']);vv=independent(y[ii],q['scores'][k],q['threshold']);assert all(np.isclose(v[m],vv[m],atol=1e-12,rtol=0,equal_nan=True) for m in lock['metrics']);rows.append(dict(arm=arm,fold=fold,inner=0,patient=pid,center=meta['center'][ii[0]],channels=len(ii),**v))
 df=pd.DataFrame(rows);assert len(df)==240 and df.patient.nunique()==80 and all(df[df.arm==arm].channels.sum()==7635 for arm in ARMS)
 metrics=lock['metrics'];summary=df.groupby('arm')[metrics].mean().reset_index();summary['patients']=80;summary['channels']=7635;summary.to_csv(a.pub/'OUTER_PATIENT_EQUAL_METRICS.csv',index=False)
 for key in ['fold','center']:df.groupby(['arm',key])[metrics].mean().reset_index().to_csv(a.pub/f'OUTER_BY_{key.upper()}.csv',index=False)
 boot=[]
 for left,right in lock['bootstrap']['comparisons']:boot.extend(paired(df,left,right,metrics))
 pd.DataFrame(boot).to_csv(a.pub/'OUTER_PAIRED_BOOTSTRAP.csv',index=False);df.groupby('arm')[['TP','FP','TN','FN']].sum().reset_index().to_csv(a.pub/'OUTER_CONFUSION_COUNTS.csv',index=False)
 rank=[]
 from scipy.stats import spearmanr
 for ref in ['C0','C2']:
  for pid in sorted(set(meta['patient'])):
   ii=np.flatnonzero(meta['patient']==pid);ii=ii[np.argsort(meta['channel'][ii],kind='stable')];s0=all_scores[ref][ii];s1=all_scores['C1'][ii];p0=all_decisions[ref][ii];p1=all_decisions['C1'][ii];z=y[ii]==0;i,j=np.triu_indices(len(ii),1);reverse=((s0[i]-s0[j])*(s1[i]-s1[j])<0)
   rank.append(dict(reference=ref,patient=pid,spearman=float(spearmanr(s0,s1).statistic),pair_reversal_fraction=float(reverse.mean()),top1_changed=bool(np.argmax(s0)!=np.argmax(s1)),TP_recovered=int((z&~p0&p1).sum()),TP_spoiled=int((z&p0&~p1).sum()),FP_added=int((~z&~p0&p1).sum()),FP_removed=int((~z&p0&~p1).sum())))
 rk=pd.DataFrame(rank);rk.groupby('reference').mean(numeric_only=True).reset_index().to_csv(a.pub/'OUTER_RANKING_CHANGES.csv',index=False);rk.groupby('reference')[['TP_recovered','TP_spoiled','FP_added','FP_removed']].sum().reset_index().to_csv(a.pub/'OUTER_ERROR_RECOVERY.csv',index=False)
 s=summary.set_index('arm');fm=df.groupby(['arm','fold']).ez_auroc.mean();positive=int((fm.loc['C1']>fm.loc['C0']).sum());ci=next(q for q in boot if q['comparison']=='C1 minus C2' and q['metric']=='ez_auroc')
 conditions=dict(AUROC_C1_C0=s.loc['C1','ez_auroc']-s.loc['C0','ez_auroc']>=.010,AUROC_C1_C2=s.loc['C1','ez_auroc']-s.loc['C2','ez_auroc']>=.005,AP_C1_ge_C0=s.loc['C1','ez_auprc']>=s.loc['C0','ez_auprc'],AP_C1_ge_C2=s.loc['C1','ez_auprc']>=s.loc['C2','ez_auprc'],MacroF1_preserved=s.loc['C1','macro_f1']>=s.loc['C0','macro_f1']-.005,EZF1_preserved=s.loc['C1','ez_f1']>=s.loc['C0','ez_f1']-.005,specificity_preserved=s.loc['C1','specificity']>=s.loc['C0','specificity']-.010,positive_original_folds=positive>=4,C1_C2_AUROC_CI_positive=ci['ci_low']>0,synchronization_control_PASS=inner['control']['status']=='PASS',integrity_PASS=True)
 # Score-only independent verification and TRAIN-only reconstruction; no refit.
 for fold in range(1,6):
  d=labs['by_fold'][fold]
  for arm in ARMS:
   path=a.runtime/f'fold{fold}/{arm}_FINAL_PRIVATE.pkl';assert sha(path)==fr['files'][f'{fold}_{arm}'];q=read(path);p=q['plan'];ti=np.flatnonzero(np.isin(d['patient'],p['train']));vi=q['selection_indices'];pre=fit_preprocessor(matrix(inner,d['patient'][ti],d['channel'][ti],arm),d['patient'][ti]);assert np.array_equal(pre['median'],q['pre']['median'])
   for k in ['mean_','var_','scale_']:assert np.array_equal(getattr(pre['scaler'],k),getattr(q['pre']['scaler'],k))
   assert threshold(d['y'][vi],q['selection_scores'],d['patient'][vi])==q['selection'];assert not set(p['train']+p['select'])&(inner['folds'][fold]['validation']|inner['folds'][fold]['test'])
 assert context(a)[-1]==deps;save(a.runtime/'OUTER_METRICS_PRIVATE.pkl',dict(rows=rows,rank=rank,freeze=binding));write_json(a.pub/'OUTER_GATE.json',dict(all_pass=all(conditions.values()),conditions=conditions,positive_folds=positive,scientific_status=lock['scientific_status'],no_independent_confirmation=True))
 write_json(a.pub/'OUTER_VERIFICATION.json',dict(status='PASS',patient_metric_cells=240,patient_id_once_per_arm=80,channels_once_per_arm=7635,EZ=1743,NEZ=5892,independent_sklearn_metric_parity=True,TRAIN_only_preprocessing_reconstructed=True,all15_model_hashes_unchanged=True,all15_prediction_hashes_unchanged=True,threshold_replay_exact=True,all_dependencies_unchanged=True,freeze_sha256=binding,prediction_seal_sha256=sha(seal),no_Development_evaluation=True,no_raw_or_EDF=True))
 write_json(done,dict(status='COMPLETE',scientific_status=lock['scientific_status'],patients=80,channels=7635,final_probe_fits=15,outer_gate_pass=all(conditions.values()),freeze_sha256=binding,one_fixed_prediction_pass=True,no_further_iteration=True));print(summary[['arm','ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr']].to_string(index=False),flush=True);print('OUTER_COMPLETE_VERIFIED',flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--inner',type=Path,required=True);p.add_argument('--runtime',type=Path,required=True);p.add_argument('--archive',type=Path,default=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'));p.add_argument('--phase',choices=['train','evaluate','all'],default='all');a=p.parse_args();a.runtime.mkdir(parents=True,exist_ok=True);c=context(a)
 with threadpool_limits(limits=2):
  if a.phase in ['train','all']:train(a,c)
  if a.phase in ['evaluate','all']:evaluate(a,c)

if __name__=='__main__':main()
