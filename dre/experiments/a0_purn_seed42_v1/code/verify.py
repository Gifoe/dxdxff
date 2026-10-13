"""Independent sealed-score metric/bootstrap and source/buffer/statistic verification."""
import json
import hashlib
import warnings
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,confusion_matrix
from runtime import args,setup,checkpoint
from bootstrap import independently_verify
from evaluate import interaction_tables
from mechanism import aggregate_distributions


def independent(y,score,tau):
 z=np.asarray(y)==0;p=np.asarray(score)<tau;tn,fp,fn,tp=confusion_matrix(z,p,labels=[False,True]).ravel()
 f=f1_score(z,p,labels=[False,True],average=None,zero_division=0);se=tp/(tp+fn) if tp+fn else np.nan;sp=tn/(tn+fp) if tn+fp else np.nan
 order=np.argsort(score,kind='stable');pos=np.flatnonzero(z[order])
 with warnings.catch_warnings():
  warnings.simplefilter('ignore');ap=average_precision_score(z,-np.asarray(score)) if z.any() else 0.
 return dict(macro_f1=f.mean(),ez_f1=f[1],nez_f1=f[0],balanced_accuracy=np.nanmean([se,sp]),
  ez_auprc=ap,ez_auroc=roc_auc_score(z,-np.asarray(score)) if z.any() and (~z).any() else np.nan,
  ez_mrr=1/(pos[0]+1) if len(pos) else 0.,top1_is_ez=float(z[order[0]]),sensitivity=se,specificity=sp,
  accuracy=(tp+tn)/len(y),predicted_ez_fraction=p.mean(),TP=tp,FP=fp,TN=tn,FN=fn,channels=len(y),observed_ez_fraction=z.mean())


def main():
 a=args();lock,ref,c,core,original,trainer,models,fm,banks,binding=setup(a);pub=a.runtime/'public'
 pre=json.loads((pub/'PRETRAIN_AUDIT.json').read_text());eng=json.loads((pub/'ENGINEERING_TESTS.json').read_text())
 assert pre['status']==eng['status']=='PASS' and pre['binding']==eng['binding']==binding
 frozen=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert frozen['binding']==binding
 fs=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json');seal=json.loads((a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json').read_text());assert seal['binding']==fs
 metrics=c.METRICS+['predicted_ez_fraction','TP','FP','TN','FN'];rows=[];maxdrift=0.
 manifest=json.loads((a.previous/'public/FEATURE_GROUP_MANIFEST.json').read_text())
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==pre['folds'][fold-1]['bank_sha256']
  d=torch.load(bp,map_location='cpu',weights_only=False);va=d['val'];original_bank=banks[fold-1]
  assert np.array_equal(d['x'],original_bank['x']) and hashlib.sha256(d['x'].tobytes()).hexdigest()==pre['folds'][fold-1]['original_x_sha256']
  assert d['mapping']==fm.mapping(manifest,d['features']) and fm.validate_values(original_bank,d['mapping'])
  for arm in lock['arms']:
   rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold);cp=checkpoint(a,arm,fold)
   assert c.sha(cp)==rec['checkpoint_sha256'] and c.sha(bp)==rec['bank_sha256']
   ck=torch.load(cp,map_location='cpu',weights_only=False);m=models.constructor(original.SourceA0,arm,d['mapping'])().eval();m.load_state_dict(ck['model'],strict=True)
   assert sum(p.numel() for p in m.parameters())==8817 and sum(p.numel() for p in m.network[0].parameters())==176
   if arm=='B0':assert isinstance(m.network[0],torch.nn.LayerNorm)
   else:assert np.array_equal(m.network[0].stat_indices.numpy(),np.arange(88) if arm=='B1' else d['mapping']['stat_indices'])
   path=bp.parent/f'{arm}_VAL_PRIVATE.pt';assert c.sha(path)==seal['files'][f'{arm}_{fold}'];cell=torch.load(path,map_location='cpu',weights_only=False)
   assert np.array_equal(cell['indices'],va) and cell['threshold']==rec['threshold']==ck['threshold'] and cell['checkpoint_sha256']==c.sha(cp)
   if arm!='B0':assert cell['binding']==fs
   for patient in sorted(set(d['patient'][va])):
    k=np.flatnonzero(d['patient'][va]==patient);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')];ix=va[k]
    im=independent(d['y'][ix],cell['scores'][k],ck['threshold']);cm=c.metric(d['y'][ix],cell['scores'][k],ck['threshold'])
    for key in im:
     assert np.isclose(im[key],cm[key],atol=1e-12,rtol=0,equal_nan=True),(fold,arm,key)
     if np.isfinite(im[key]):maxdrift=max(maxdrift,abs(im[key]-cm[key]))
    im['channel_stratum']='<64' if len(ix)<64 else '64to127' if len(ix)<128 else '>=128'
    rows.append(dict(arm=arm,fold=fold,patient=patient,center=str(d['center'][ix[0]]),**im))
 df=pd.DataFrame(rows);assert len(df)==260 and df.patient.nunique()==47
 for filename,keys in [('DEVELOPMENT_SUMMARY.csv',['arm']),('DEVELOPMENT_BY_FOLD.csv',['arm','fold']),
  ('DEVELOPMENT_BY_CENTER.csv',['arm','center']),('DEVELOPMENT_BY_CHANNEL_STRATA.csv',['arm','channel_stratum'])]:
  actual=pd.read_csv(pub/filename).set_index(keys).sort_index();expected=df.groupby(keys)[metrics].mean().sort_index()
  assert actual.index.equals(expected.index) and np.allclose(actual[metrics],expected,atol=1e-12,rtol=0,equal_nan=True)
 uactual=pd.read_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv').set_index('arm').sort_index()
 uexpected=df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().sort_index();assert np.allclose(uactual[metrics],uexpected,atol=1e-12,rtol=0)
 for filename in ['PAIRED_BOOTSTRAP.csv','FACTORIAL_COMPARISONS.csv']:
  actual=pd.read_csv(pub/filename)
  for left,right in lock['comparisons']:
   calc=independently_verify(df[df.arm==left],df[df.arm==right],metrics)
   for metric,v in calc.items():
    r=actual[(actual.comparison==left+' minus '+right)&(actual.metric==metric)].iloc[0];assert np.allclose(v,[r.delta,r.ci_low,r.ci_high],atol=1e-12,rtol=0,equal_nan=True)
 ll,rr=interaction_tables(df,metrics);calc=independently_verify(ll,rr,metrics);actual=pd.read_csv(pub/'FACTORIAL_COMPARISONS.csv')
 for metric,v in calc.items():
  r=actual[(actual.comparison=='(B3-B2)-(B1-B0)')&(actual.metric==metric)].iloc[0];assert np.allclose(v,[r.delta,r.ci_low,r.ci_high],atol=1e-12,rtol=0,equal_nan=True)
 # Separate independently expressed gate, not reuse evaluator decision code.
 t=df.groupby('arm')[metrics].mean();f=df.groupby(['fold','arm']).ez_auroc.mean().unstack()
 conditions=[t.loc['B3','ez_auroc']-t.loc['B0','ez_auroc']>=.010,t.loc['B3','ez_auroc']-t.loc['B1','ez_auroc']>=.005,
  t.loc['B3','ez_auroc']-t.loc['B2','ez_auroc']>=.005,t.loc['B3','ez_auprc']>=t.loc['B0','ez_auprc'],
  t.loc['B3','macro_f1']>=t.loc['B0','macro_f1'],t.loc['B3','ez_f1']-t.loc['B0','ez_f1']>=-.005,
  t.loc['B3','specificity']-t.loc['B0','specificity']>=-.010,(f.B3>f.B0).sum()>=4,True,True]
 gate=json.loads((pub/'VALIDATION_GATE.json').read_text())
 keys=['AUROC_B3_minus_B0_ge0_010','AUROC_B3_minus_B1_ge0_005','AUROC_B3_minus_B2_ge0_005','AUPRC_not_decreased','MacroF1_not_decreased',
  'EZF1_decline_at_most0_005','specificity_decline_at_most0_010','positive_AUROC_folds_ge4','no_fold_preprocessing_or_feature_mismatch','all_identity_parameter_numerical_leakage_checks']
 assert [gate['criteria'][k] for k in keys]==list(map(bool,conditions)) and gate['pass_gate']==all(conditions)
 ms=json.loads((a.runtime/'MECHANISM_SEAL.json').read_text());assert ms['binding']==fs and ms['sha256']==c.sha(a.runtime/'NORMALIZATION_MECHANISM_PRIVATE.pt')
 mech=torch.load(a.runtime/'NORMALIZATION_MECHANISM_PRIVATE.pt',map_location='cpu',weights_only=False);assert mech['binding']==fs
 # Independently recompute core normalization/duplicate statistics from original inputs.
 for fold in range(1,6):
  d=banks[fold-1];u=fm.mapping(manifest,d['features'])['stat_indices'];xx=d['x'].astype(np.float64)
  for role,ix in [('FIT',d['train']),('VAL',d['val'])]:
   q=mech['statistics'];q=q[(q.fold==fold)&(q.role==role)]
   assert np.array_equal(q.patient.to_numpy(),d['patient'][ix])
   a88=xx[ix];a56=a88[:,u];s88=(a88*a88).sum(1)/88;s56=(a56*a56).sum(1)/56
   vals={'mean88':a88.sum(1)/88,'mean56':a56.sum(1)/56,'RMS56_over_RMS88_eps':np.sqrt((s56+1e-5)/(s88+1e-5)),
    'duplicate_second_moment_contribution88':(a88[:,72:88]**2).sum(1)/88}
   for name,v in vals.items():assert np.allclose(q[name],v,atol=1e-12,rtol=1e-12)
 stat=mech['statistics'];fields=[k for k in stat if k not in ['patient','center','fold','unit','role']]
 check=aggregate_distributions(stat,fields);cols=['role','fold','center','statistic']
 actual=pd.concat([pd.read_csv(pub/'NORMALIZATION_STATISTICS.csv'),pd.read_csv(pub/'DUPLICATE_INFLUENCE.csv')],ignore_index=True)
 for frame in [check,actual]:
  for col in cols:
   if col not in frame:frame[col]=np.nan
  frame[cols]=frame[cols].fillna('ALL').astype(str)
 numeric=['finite_fraction','mean','std','minimum','q05','q25','median','q75','q95','maximum']
 aa=actual.set_index(cols).sort_index();bb=check.set_index(cols).sort_index()
 assert aa.index.equals(bb.index) and np.allclose(aa[numeric],bb[numeric],atol=1e-10,rtol=1e-10,equal_nan=True)
 assert setup(a)[-1]==binding and not gate['new_outer_authorized']
 c.write_json(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',metric_cells=260,max_metric_drift=maxdrift,
  verified_contrast_intervals=96,draws=10000,unique_clusters=47,retained_appearances=65,source_bank_buffer_affine_threshold_checks=True,
  independent_numpy_statistic_checks=True,all_aggregate_gate_parity=True,new_outer_accessed=False))
 print('PURN_INDEPENDENT_VERIFICATION_PASS',flush=True)


if __name__=='__main__':main()
