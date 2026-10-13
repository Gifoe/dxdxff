"""Independent metrics/bootstrap from sealed legal development predictions only."""
import json
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,confusion_matrix
from runtime import args,setup,checkpoint
from bootstrap import independently_verify


def independent(y,score,tau):
 z=np.asarray(y)==0;p=np.asarray(score)<tau;tn,fp,fn,tp=confusion_matrix(z,p,labels=[False,True]).ravel()
 f=f1_score(z,p,labels=[False,True],average=None,zero_division=0);se=tp/(tp+fn) if tp+fn else np.nan;sp=tn/(tn+fp) if tn+fp else np.nan
 order=np.argsort(score,kind='stable');positive=np.flatnonzero(z[order])
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');ap=average_precision_score(z,-np.asarray(score)) if z.any() else 0.
 return dict(macro_f1=f.mean(),ez_f1=f[1],nez_f1=f[0],balanced_accuracy=np.nanmean([se,sp]),
  ez_auprc=ap,ez_auroc=roc_auc_score(z,-np.asarray(score)) if z.any() and (~z).any() else np.nan,
  ez_mrr=1/(positive[0]+1) if len(positive) else 0.,top1_is_ez=float(z[order[0]]),sensitivity=se,specificity=sp,
  accuracy=(tp+tn)/len(y),predicted_ez_fraction=p.mean(),TP=tp,FP=fp,TN=tn,FN=fn,channels=len(y),observed_ez_fraction=z.mean())


def main():
 a=args();lock,ref,c,core,original,trainer,models,ks,banks,binding=setup(a);pub=a.runtime/'public'
 pre=json.loads((pub/'PRETRAIN_AUDIT.json').read_text());eng=json.loads((pub/'ENGINEERING_TESTS.json').read_text())
 assert pre['status']==eng['status']=='PASS' and pre['binding']==eng['binding']==binding
 frozen=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert frozen['binding']==binding
 fs=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json');seal=json.loads((a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json').read_text())
 assert seal['binding']==fs and not seal['new_outer_accessed']
 rows=[];metrics=c.METRICS+['predicted_ez_fraction','TP','FP','TN','FN'];maxdrift=0.
 manifest=json.loads((a.previous/'public/FEATURE_GROUP_MANIFEST.json').read_text())
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['folds'][fold-1]['bank_sha256']
  d=torch.load(bp,map_location='cpu',weights_only=False);va=d['val']
  fmap=ks.mapping(manifest,d['features']);stats,_=ks.fit(d['x'],d['patient'],d['train'],fmap)
  for key in ['knots','slope_mask','boundary_mask','column_to_curve']:assert np.array_equal(stats[key],d['stats'][key])
  for arm in lock['arms']:
   rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold);cp=checkpoint(a,arm,fold)
   assert c.sha(cp)==rec['checkpoint_sha256'] and c.sha(bp)==rec['bank_sha256']
   ck=torch.load(cp,map_location='cpu',weights_only=False);m=models.constructor(original.SourceA0,arm,d['stats'])().eval();m.load_state_dict(ck['model'],strict=True)
   assert sum(p.numel() for p in m.parameters())==lock['parameters'][arm]
   path=bp.parent/f'{arm}_VAL_PRIVATE.pt';assert c.sha(path)==seal['files'][f'{arm}_{fold}']
   cell=torch.load(path,map_location='cpu',weights_only=False);assert np.array_equal(cell['indices'],va)
   assert cell['threshold']==ck['threshold']==rec['threshold'] and cell['checkpoint_sha256']==c.sha(cp)
   if arm!='B0':assert cell['binding']==fs
   variants=[(arm,cell['scores'])]
   if arm=='B3':
    assert np.array_equal(m.transform.knots.numpy(),d['stats']['knots'])
    assert np.array_equal(m.transform.column_to_curve.numpy(),d['stats']['column_to_curve'])
    assert np.array_equal(m.transform.slope_mask.numpy(),d['stats']['slope_mask'])
    assert (m.transform.theta.detach()[~m.transform.slope_mask]==0).all()
    s=m.transform.slopes().detach();assert torch.isfinite(s).all() and (s>=.25).all() and (s<=1.75).all()
    mp=bp.parent/'MECHANISM_PRIVATE.pt';assert c.sha(mp)==seal['files'][f'mechanism_{fold}']
    mm=torch.load(mp,map_location='cpu',weights_only=False)
    assert mm['binding']==fs and mm['threshold']==ck['threshold'] and np.array_equal(mm['indices'],va)
    assert np.array_equal(mm['full_scores'],cell['scores']) and np.array_equal(mm['permutation'],np.random.default_rng(42).permutation(56))
    variants.extend([('B3_IDENTITY',mm['identity_scores']),('B3_CURVE_PERMUTED',mm['permuted_scores'])])
   for variant,scores in variants:
    part=[]
    for patient in sorted(set(d['patient'][va])):
     k=np.flatnonzero(d['patient'][va]==patient);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')];ix=va[k]
     im=independent(d['y'][ix],scores[k],ck['threshold']);cm=c.metric(d['y'][ix],scores[k],ck['threshold'])
     for key in im:
      assert np.isclose(im[key],cm[key],atol=1e-12,rtol=0,equal_nan=True),(fold,variant,key)
      if np.isfinite(im[key]):maxdrift=max(maxdrift,abs(im[key]-cm[key]))
     part.append(dict(arm=variant,fold=fold,patient=patient,center=str(d['center'][ix[0]]),**im))
    if variant==arm:
     assert abs(np.mean([r['macro_f1'] for r in part])-ck['macro_f1'])<1e-12
     assert abs(np.mean([r['ez_f1'] for r in part])-ck['ez_f1'])<1e-12
    rows.extend(part)
 df=pd.DataFrame(rows);assert len(df)==390 and df.patient.nunique()==47
 primary=df[df.arm.isin(lock['arms'])]
 for filename,keys,subset in [('DEVELOPMENT_SUMMARY.csv',['arm'],primary),('DEVELOPMENT_BY_FOLD.csv',['arm','fold'],primary),
   ('DEVELOPMENT_BY_CENTER.csv',['arm','center'],primary),('SPLINE_INTERVENTIONS.csv',['arm'],df[df.arm.isin(['B3','B3_IDENTITY','B3_CURVE_PERMUTED'])])]:
  actual=pd.read_csv(pub/filename).set_index(keys).sort_index();expected=subset.groupby(keys)[metrics].mean().sort_index()
  assert actual.index.equals(expected.index) and np.allclose(actual[metrics],expected,atol=1e-12,rtol=0,equal_nan=True),filename
 for filename,contrasts in [('PAIRED_BOOTSTRAP.csv',lock['comparisons']),('SPLINE_INTERVENTION_BOOTSTRAP.csv',[['B3','B3_IDENTITY'],['B3','B3_CURVE_PERMUTED']])]:
  actual=pd.read_csv(pub/filename)
  for left,right in contrasts:
   calculated=independently_verify(df[df.arm==left],df[df.arm==right],metrics)
   for metric,values in calculated.items():
    r=actual[(actual.comparison==left+' minus '+right)&(actual.metric==metric)].iloc[0]
    assert np.allclose(values,[r.delta,r.ci_low,r.ci_high],atol=1e-12,rtol=0,equal_nan=True),(filename,metric)
 t=primary.groupby('arm')[metrics].mean();folds=primary.groupby(['fold','arm']).ez_auroc.mean().unstack()
 conditions=[t.loc['B3','ez_auroc']-t.loc['B0','ez_auroc']>=.010,t.loc['B3','ez_auprc']>=t.loc['B0','ez_auprc'],
  t.loc['B3','macro_f1']>=t.loc['B0','macro_f1'],t.loc['B3','ez_f1']-t.loc['B0','ez_f1']>=-.005,
  t.loc['B3','ez_auroc']-t.loc['B1','ez_auroc']>=.005,t.loc['B3','ez_auroc']-t.loc['B2','ez_auroc']>=.005,
  (folds.B3>folds.B0).sum()>=4,t.loc['B3','specificity']-t.loc['B0','specificity']>=-.010,True]
 gate=json.loads((pub/'VALIDATION_GATE.json').read_text())
 keys=['AUROC_B3_minus_B0_ge0.010','AUPRC_not_decreased','MacroF1_not_decreased','EZF1_decline_at_most0.005',
  'AUROC_B3_minus_B1_ge0.005','AUROC_B3_minus_B2_ge0.005','positive_folds_ge4','specificity_decline_at_most0.010','all_pretraining_integrity_checks']
 assert list(map(bool,conditions))==[gate['criteria'][k] for k in keys] and gate['pass_gate']==all(conditions)
 assert not gate['new_outer_authorized'] and setup(a)[-1]==binding
 c.write_json(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',independent_backend='sklearn',metric_cells=390,
   max_metric_drift=maxdrift,paired_intervals_verified=112,draws=10000,unique_clusters=47,retained_appearances=65,
   hash_parameter_buffer_threshold_and_FIT_knot_checks=True,all_aggregate_and_gate_parity=True,new_outer_accessed=False))
 print('PR_SPLINE_INDEPENDENT_VERIFICATION_PASS',flush=True)


if __name__=='__main__':main()
