"""Score-only primary evaluation and prespecified frozen mechanism diagnostics."""
import json
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from common import sha,read,save,write_json,metric
from bootstrap import paired

def subgroup(df,key,metrics):
 out=[]
 for (arm,group),q in df.groupby(['arm',key]):
  suppress=q.patient.nunique()<3
  out.append(dict(arm=arm,**{key:group},appearances=len(q),unique_patients=q.patient.nunique(),small_cell_suppressed=suppress,**{m:np.nan if suppress else float(q[m].mean()) for m in metrics}))
 return out

def adjusted(df,left,right):
 ll=df[df.arm==left].set_index(['fold','inner','patient']).sort_index();rr=df[df.arm==right].set_index(['fold','inner','patient']).sort_index();assert ll.index.equals(rr.index)
 d=(ll.ez_auroc-rr.ez_auroc).to_numpy();ok=np.isfinite(d);q=ll.iloc[np.flatnonzero(ok)];p=q.index.get_level_values('patient').to_numpy().astype(str);f=q.index.get_level_values('fold').to_numpy()
 x=np.column_stack([np.ones(len(q)),np.log(q.channel_count.to_numpy()/64),q.records.to_numpy()-2,*[(f==j).astype(float)-.2 for j in range(1,5)]]);d=d[ok]
 if np.linalg.matrix_rank(x)<x.shape[1]:return dict(comparison=left+' minus '+right,status='NOT_ESTIMABLE_RANK_DEFICIENT')
 coefficient=np.linalg.lstsq(x,d,rcond=None)[0];ids=np.unique(p);ix=np.random.default_rng(42).integers(0,len(ids),(10000,len(ids)));mult=np.stack([(ix==j).sum(1) for j in range(len(ids))],1)
 xtx=np.stack([x[p==pid].T@x[p==pid] for pid in ids]);xty=np.stack([x[p==pid].T@d[p==pid] for pid in ids]);aa=(mult@xtx.reshape(len(ids),-1)).reshape(10000,x.shape[1],x.shape[1]);bb=mult@xty
 ranks=np.linalg.matrix_rank(aa);valid=ranks==x.shape[1];sol=np.linalg.solve(aa[valid],bb[valid,:,None])[...,0];boot=sol[:,0]
 return dict(comparison=left+' minus '+right,status='ESTIMABLE',adjusted_delta_at64_channels2_seizures_equal_folds=float(coefficient[0]),ci_low=float(np.percentile(boot,2.5)),ci_high=float(np.percentile(boot,97.5)),log_channel_count_slope=float(coefficient[1]),seizure_count_slope=float(coefficient[2]),bootstrap_estimable_draws=int(valid.sum()),draws=10000,unique_patients=len(ids),causal_claim=False)

def evaluate(a,lock,binding):
 bank=read(a.runtime/'CONTEXT_TABLES_PRIVATE.pkl');label=read(a.runtime/'FIT_LABELS_PRIVATE.pkl');fr=json.loads((a.runtime/'PROBE_FREEZE_PRIVATE.json').read_text());assert fr['binding']==label['binding']==bank['binding']==binding
 rows=[];inter=[];rank=[];sealed={}
 for fold in range(1,6):
  d=label['by_fold'][fold]
  for h in range(4):
   cells={arm:read(a.runtime/f'fold{fold}/inner{h}/{arm}_PRIVATE.pkl') for arm in ['C0','C1','C2']};hi=cells['C0']['indices'];assert all(np.array_equal(c['indices'],hi) for c in cells.values())
   for arm,c in cells.items():assert sha(a.runtime/f'fold{fold}/inner{h}/{arm}_PRIVATE.pkl')==fr['files'][f'{fold}_{h}_{arm}'];sealed[f'{fold}_{h}_{arm}']=fr['files'][f'{fold}_{h}_{arm}']
   for pid in sorted(set(d['patient'][hi])):
    k=np.flatnonzero(d['patient'][hi]==pid);k=k[np.argsort(d['channel'][hi][k],kind='stable')];ii=hi[k];y=d['y'][ii];center=d['center'][ii[0]];n=int(d['channel_count'][ii[0]]);nr=int(d['records'][ii[0]])
    info=dict(fold=fold,inner=h,patient=pid,center=center,channel_count=n,records=nr,channel_stratum='<=64' if n<=64 else ('65-128' if n<=128 else '>128'),seizure_stratum=str(nr) if nr<3 else '3+')
    for arm,c in cells.items():rows.append(dict(info,arm=arm,**metric(y,c['scores'][k],c['threshold'])))
    c=cells['C1'];s1=c['scores'][k];p1=s1>=c['threshold'];z=y==0;logit=c['extra']['context_logit'][k]
    for tag,s in c['interventions'].items():inter.append(dict(info,arm=tag,**metric(y,s[k],c['threshold'])))
    for reference in ['C0','C2']:
     base=cells[reference];s0=base['scores'][k];p0=s0>=base['threshold'];i,j=np.triu_indices(len(k),1);aa=np.sign(s0[i]-s0[j]);bb=np.sign(s1[i]-s1[j]);reverse=int((aa*bb<0).sum());changes=int((aa!=bb).sum());rho=spearmanr(s0,s1).statistic if np.std(s0)>0 and np.std(s1)>0 else np.nan
     with warnings.catch_warnings():warnings.simplefilter('ignore',RuntimeWarning);vv=float(np.nanmean(np.nanvar(bank['tables'][pid]['C1'][:,98:],axis=0)))
     rank.append(dict(info,reference=reference,ranking_spearman=rho,strict_pair_reversals=reverse,pair_order_changes_including_ties=changes,total_pairs=len(i),strict_pair_reversal_fraction=reverse/max(len(i),1),top1_changed=bool(np.argmax(s0)!=np.argmax(s1)),TP_recovered=int((z&~p0&p1).sum()),TP_spoiled=int((z&p0&~p1).sum()),FP_added=int((~z&~p0&p1).sum()),FP_removed=int((~z&p0&~p1).sum()),all_errors_recovered=int(((p0!=z)&(p1==z)).sum()),correct_decisions_spoiled=int(((p0==z)&(p1!=z)).sum()),context_logit_mean_abs=float(np.abs(logit).mean()),context_logit_within_patient_std=float(logit.std()),raw_context_feature_variance_mean=vv))
 df=pd.DataFrame(rows);expected=set().union(*(v['fit'] for v in bank['folds'].values()));assert len(df)==765 and all(df[df.arm==arm].shape[0]==255 for arm in ['C0','C1','C2']);assert set(df.patient)==expected
 metrics=lock['metrics'];summary=df.groupby('arm')[metrics].mean().reset_index();summary['appearances']=255;summary['unique_patients']=len(expected);summary['input_width']=182;summary.to_csv(a.pub/'INNER_EVALUATION_SUMMARY.csv',index=False)
 df.groupby(['arm','fold'])[metrics].mean().reset_index().to_csv(a.pub/'INNER_EVALUATION_BY_ORIGINAL_FOLD.csv',index=False)
 pd.DataFrame(subgroup(df,'center',metrics)).to_csv(a.pub/'INNER_EVALUATION_BY_CENTER.csv',index=False)
 df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().reset_index().to_csv(a.pub/'UNIQUE_PATIENT_SUMMARY.csv',index=False)
 for key in ['channel_stratum','seizure_stratum']:pd.DataFrame(subgroup(df,key,metrics)).to_csv(a.pub/f'INNER_EVALUATION_BY_{key.upper()}.csv',index=False)
 df.groupby('arm')[['sensitivity','specificity','TP','FP','TN','FN','accuracy','predicted_ez_fraction']].mean().reset_index().to_csv(a.pub/'SENSITIVITY_SPECIFICITY_AUDIT.csv',index=False)
 boot=[]
 for left,right in lock['bootstrap']['comparisons']:boot.extend(paired(df,left,right,metrics))
 pd.DataFrame(boot).to_csv(a.pub/'PAIRED_BOOTSTRAP.csv',index=False)
 mi=pd.DataFrame(inter);mi.groupby('arm')[metrics].mean().reset_index().to_csv(a.pub/'STAGE_AND_CONTEXT_INTERVENTIONS.csv',index=False)
 ib=[]
 for tag in ['early_shift','later_shift','context_neutral']:ib.extend(paired(pd.concat([df[df.arm=='C1'],mi[mi.arm==tag]]),'C1',tag,metrics))
 pd.DataFrame(ib).to_csv(a.pub/'INTERVENTION_BOOTSTRAP.csv',index=False)
 rk=pd.DataFrame(rank);fields=['ranking_spearman','strict_pair_reversal_fraction','top1_changed','TP_recovered','TP_spoiled','FP_added','FP_removed','all_errors_recovered','correct_decisions_spoiled','context_logit_mean_abs','context_logit_within_patient_std','raw_context_feature_variance_mean']
 rr=[]
 for ref,q in rk.groupby('reference'):rr.append(dict(reference=ref,appearances=len(q),unique_patients=q.patient.nunique(),strict_pair_reversals=int(q.strict_pair_reversals.sum()),total_pairs=int(q.total_pairs.sum()),pooled_strict_pair_fraction=float(q.strict_pair_reversals.sum()/q.total_pairs.sum()),**q[fields].mean().to_dict()))
 pd.DataFrame(rr).to_csv(a.pub/'RANKING_CHANGES.csv',index=False)
 errors=['TP_recovered','TP_spoiled','FP_added','FP_removed','all_errors_recovered','correct_decisions_spoiled']
 rk.groupby('reference')[errors].sum().reset_index().to_csv(a.pub/'ERROR_COUNTS_REPEATED_APPEARANCES.csv',index=False)
 for key in ['channel_stratum','seizure_stratum']:
  out=[]
  for (ref,v),q in rk.groupby(['reference',key]):out.append(dict(reference=ref,**{key:v},appearances=len(q),unique_patients=q.patient.nunique(),**{f:float(q[f].mean()) if q.patient.nunique()>=3 else np.nan for f in fields}))
  pd.DataFrame(out).to_csv(a.pub/f'RANKING_BY_{key.upper()}.csv',index=False)
 pd.DataFrame([adjusted(df,'C1','C0'),adjusted(df,'C1','C2')]).to_csv(a.pub/'CHANNEL_COUNT_ADJUSTED_AUDIT.csv',index=False)
 s=summary.set_index('arm');foldmeans=df.groupby(['arm','fold'])[metrics].mean();positive=int((foldmeans.loc['C1','ez_auroc']>foldmeans.loc['C0','ez_auroc']).sum());ci=next(q for q in boot if q['comparison']=='C1 minus C2' and q['metric']=='ez_auroc');g=lock['gates']
 conditions=dict(AUROC_C1_C0=s.loc['C1','ez_auroc']-s.loc['C0','ez_auroc']>=.010,AUROC_C1_C2=s.loc['C1','ez_auroc']-s.loc['C2','ez_auroc']>=.005,AP_C1_ge_C0=s.loc['C1','ez_auprc']>=s.loc['C0','ez_auprc'],AP_C1_ge_C2=s.loc['C1','ez_auprc']>=s.loc['C2','ez_auprc'],MacroF1_preserved=s.loc['C1','macro_f1']>=s.loc['C0','macro_f1']-.005,EZF1_preserved=s.loc['C1','ez_f1']>=s.loc['C0','ez_f1']-.005,specificity_preserved=s.loc['C1','specificity']>=s.loc['C0','specificity']-.010,positive_original_folds=positive>=4,C1_C2_AUROC_CI_positive=ci['ci_low']>0,synchronization_control_PASS=bank['control']['status']=='PASS',alignment_and_leakage_PASS=True)
 if all(conditions.values()):terminal='SYNCHRONIZATION_MECHANISM_PROMISING_NOT_INDEPENDENTLY_CONFIRMED'
 elif conditions['AUROC_C1_C0'] and not (conditions['AUROC_C1_C2'] and conditions['C1_C2_AUROC_CI_positive']):terminal='PATIENT_CONTEXT_POSSIBLY_USEFUL_SYNCHRONIZATION_NOT_IDENTIFIED'
 elif not conditions['AUROC_C1_C0'] and not conditions['AUROC_C1_C2']:terminal='NO_DEMONSTRATED_SYNCHRONOUS_CONTEXT_UTILITY'
 else:terminal='CONTINUATION_GATE_FAILED'
 actual=dict(C1_C0_AUROC=float(s.loc['C1','ez_auroc']-s.loc['C0','ez_auroc']),C1_C2_AUROC=float(s.loc['C1','ez_auroc']-s.loc['C2','ez_auroc']),positive_original_folds=positive,C1_C2_AUROC_ci_low=ci['ci_low'])
 write_json(a.pub/'SCIENTIFIC_GATE.json',dict(terminal=terminal,pass_all=all(conditions.values()),conditions=conditions,actual=actual,independent_verification_pending=True,no_TIDE_training=True,no_Development_or_outer_evaluation=True))
 save(a.runtime/'EVALUATION_PRIVATE.pkl',dict(binding=binding,rows=rows,interventions=inter,ranking=rank));write_json(a.runtime/'PREDICTION_SEAL_PRIVATE.json',dict(binding=binding,files=sealed,probe_freeze_sha256=sha(a.runtime/'PROBE_FREEZE_PRIVATE.json')))
 write_json(a.pub/'RUN_STATUS.json',dict(status='EVALUATED_PENDING_VERIFICATION',terminal=terminal,probe_models_fitted=60,appearances=255,unique_patients=len(expected),no_TIDE_or_outer=True));print(summary[['arm','ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr']].to_string(index=False),flush=True)
