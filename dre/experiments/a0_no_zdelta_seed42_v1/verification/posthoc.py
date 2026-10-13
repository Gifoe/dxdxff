"""Additional sealed-score checks; no training, inference, or threshold selection."""
import json
import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score,average_precision_score,f1_score,confusion_matrix
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from baseline_admission import args,dependencies


def main():
 a=args();ref,c,core,original,trainer,helper,banks,binding=dependencies(a)
 pub=a.runtime/'public';ad=json.loads((pub/'PRETRAIN_AUDIT.json').read_text());freeze=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text())
 assert json.loads((pub/'OUTER_STATUS.json').read_text())['status']=='ONE_FROZEN_A0_NO_ZDELTA_OUTER_PASS_COMPLETE'
 assert json.loads((pub/'INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
 seal=json.loads((a.runtime/'OUTER_PREDICTION_SEAL.json').read_text());assert seal['binding']==c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json')
 for fold,old in enumerate(banks,1):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert c.sha(bp)==ad['folds'][fold-1]['bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False)
  views=[re.fullmatch(r'(b0|physics)_(abs|delta|zdelta|ratio)__(.+)__seizure_(mean|std)',str(x)).group(2) for x in d['features']]
  mask=np.asarray([v!='zdelta' for v in views]);assert np.array_equal(mask,d['mask']) and int((~mask).sum())==26
  assert np.array_equal(d['x'][:,mask],old['x'][:,mask]) and np.all(d['x'][:,~mask]==0)
  for k in ['patient','channel','center','features','y','train','val','raw_x','g']:
   assert np.array_equal(d[k],old[k],equal_nan=True) if np.asarray(d[k]).dtype.kind in 'fc' else np.array_equal(d[k],old[k])
  assert set(d['pre'])==set(old['pre'])
  for k in d['pre']:assert np.array_equal(d['pre'][k],old['pre'][k])
 with np.load(a.features,allow_pickle=True) as f:y=f['y']
 assert len(y)==7635 and int((y==0).sum())==1743
 actual=pd.read_csv(pub/'OUTER_POOLED_CHANNEL_METRICS.csv').set_index('arm');drift=0.
 for arm in ['A0','A0_NO_ZDELTA']:
  scores=np.empty(len(y));pred=np.zeros(len(y),bool);visited=[]
  for fold in range(1,6):
   path=a.previous/f'no_source/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt' if arm=='A0' else a.runtime/f'{arm}_fold{fold}_OUTER_PRIVATE.pt'
   assert c.sha(path)==seal['files'][f'{arm}_{fold}'];cell=torch.load(path,map_location='cpu',weights_only=False);ix=cell['indices'];visited.extend(ix.tolist())
   r=next(r for r in freeze['selections'] if r['arm']==arm and r['fold']==fold);assert cell['threshold']==r['threshold']
   scores[ix]=cell['scores'];pred[ix]=cell['scores']<r['threshold']
  assert sorted(visited)==list(range(7635));z=y==0;tn,fp,fn,tp=confusion_matrix(z,pred,labels=[False,True]).ravel()
  f=f1_score(z,pred,labels=[False,True],average=None,zero_division=0);se=tp/(tp+fn);sp=tn/(tn+fp)
  expected=dict(macro_f1=float(f.mean()),ez_f1=f[1],nez_f1=f[0],balanced_accuracy=(se+sp)/2,ez_auprc=average_precision_score(z,-scores),ez_auroc=roc_auc_score(z,-scores),
   sensitivity=se,specificity=sp,accuracy=(tp+tn)/len(y),TP=tp,FP=fp,TN=tn,FN=fn,channels=len(y),predicted_ez_fraction=pred.mean(),observed_ez_fraction=z.mean())
  for k,v in expected.items():assert abs(v-actual.loc[arm,k])<1e-12;drift=max(drift,abs(v-actual.loc[arm,k]))
 c.write_json(pub/'POSTHOC_VERIFICATION.json',dict(status='PASS',source_free_original_banks_unchanged=True,all_five_masks_independently_name_verified=True,masked_columns=26,
  all_retained_values_and_preprocessor_statistics_exact=True,independent_pooled_metric_drift=drift,no_training_or_new_model_inference=True,no_threshold_reselection=True,
  torch=torch.__version__,numpy=np.__version__,script_sha256=c.sha(Path(__file__)),scientific_code_unchanged_after_outcomes=True))
 print('POSTHOC_MASK_INPUT_POOLED_PARITY_PASS',flush=True)


if __name__=='__main__':main()
