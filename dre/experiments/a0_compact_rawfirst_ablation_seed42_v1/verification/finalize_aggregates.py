"""Reporting-only union and pooled confusion verification; no model inference."""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from audit_sources import arguments,context


def main():
 a=arguments();o=context(a);lock,c=o[0],o[2];pub=a.runtime/'public';audits={}
 for stage in ['RAW','FEATURE']:
  audits[stage]=json.loads((pub/f'{stage}_INDEPENDENT_VERIFICATION.json').read_text());assert audits[stage]['status']=='PASS'
  assert json.loads((pub/f'{stage}_OUTER_STATUS.json').read_text())['status']=='ONE_FROZEN_'+stage+'_OUTER_PASS_COMPLETE'
 assert json.loads((a.runtime/'RAW_PUBLICATION_PRIVATE.json').read_text())['github_commit']=='22425ef38e34cf65d64e5dc761bdf375e3c8d20c'
 metrics=lock['metrics'];combined={}
 for phase,count,clusters in [('OUTER',80,80),('DEVELOPMENT',65,47)]:
  frames=[pd.DataFrame(torch.load(a.runtime/f'{stage}_{phase}_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False)) for stage in ['RAW','FEATURE']]
  overlap=[d[d.arm=='F0'].set_index(['fold','patient']).sort_index() for d in frames];assert overlap[0].index.equals(overlap[1].index)
  assert np.allclose(overlap[0][metrics],overlap[1][metrics],atol=1e-12,rtol=0,equal_nan=True)
  df=pd.concat([frames[0][frames[0].arm=='R0'],frames[1]],ignore_index=True)
  assert df.arm.nunique()==9 and all((len(q),q.patient.nunique())==(count,clusters) for _,q in df.groupby('arm'))
  result=df.groupby('arm')[metrics].mean().reset_index();result['parameters']=result.arm.map(lambda v:18153 if v=='R0' else lock['matrix'][v]['parameters'])
  result['status']=result.arm.map(lambda v:'historical_hash_verified_reuse' if v in ['F0','F6','F7'] else 'new_five_fold_train_and_frozen_outer')
  result.to_csv(pub/f'{phase}_ALL_MODELS_SUMMARY.csv',index=False);combined[phase]=df
  if phase=='DEVELOPMENT':df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_ALL_UNIQUE_ID_SUMMARY.csv',index=False)
 pooled=[]
 for arm,q in combined['OUTER'].groupby('arm'):
  counts={k:int(q[k].sum()) for k in ['TP','FP','TN','FN']}
  assert q.channels.sum()==7635 and counts['TP']+counts['FN']==1743 and counts['FP']+counts['TN']==5892
  pooled.append(dict(arm=arm,unit='secondary_pooled_confusion_counts',**counts,channels=7635))
 pd.DataFrame(pooled).to_csv(pub/'POOLED_CONFUSION_COUNTS.csv',index=False)
 for stage in ['RAW','FEATURE']:
  pp=pd.read_csv(pub/f'{stage}_OUTER_POOLED_CHANNELS.csv').set_index('arm');total=pd.DataFrame(pooled).set_index('arm')
  for arm in pp.index:
   assert np.array_equal(pp.loc[arm,['TP','FP','TN','FN']].astype(int),total.loc[arm,['TP','FP','TN','FN']].astype(int))
 histories=pd.concat([pd.read_csv(pub/f'{stage}_TRAINING_HISTORY.csv') for stage in ['RAW','FEATURE']],ignore_index=True)
 histories.to_csv(pub/'ALL_NEW_TRAINING_HISTORY.csv',index=False)
 selections=pd.concat([pd.read_csv(pub/f'{stage}_MODEL_SELECTION.csv') for stage in ['RAW','FEATURE']],ignore_index=True);assert len(selections)==30
 selections.to_csv(pub/'ALL_NEW_MODEL_SELECTION.csv',index=False)
 total_epochs=histories.groupby(['arm','fold']).epoch.max().sum();assert total_epochs==len(histories)
 c.write_json(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',raw_status='PASS',feature_status='PASS',
  stage_audit_sha256={k:c.sha(pub/f'{k}_INDEPENDENT_VERIFICATION.json') for k in ['RAW','FEATURE']},
  checked_patient_model_cells=sum(v['metric_cells'] for v in audits.values()),unique_arm_patient_appearance_cells=1305,
  checked_bootstrap_intervals=sum(v['verified_intervals'] for v in audits.values()),max_metric_drift=max(v['max_metric_drift'] for v in audits.values()),
  raw_F0_feature_F0_exact_union_parity=True,all9_pooled_confusion_counts_verified=True,new_model_fold_cells=30,historical_model_fold_cells_reused=15,new_training_epochs=int(total_epochs),
  raw_result_published_before_feature_training=True,raw_publication_commit='22425ef38e34cf65d64e5dc761bdf375e3c8d20c',no_new_inference_or_tuning=True,script_sha256=c.sha(Path(__file__))))
 c.write_json(pub/'RUN_STATUS.json',dict(status='COMPLETE_A0_COMPACT_RAW_FIRST_ABLATION_OUTER_EXPLORATORY',raw_complete=True,feature_ablations_complete=True,
  new_raw_cells=5,new_feature_cells=25,historical_reused_cells=15,all9_configurations_complete=True,no_new_model_or_test_driven_iteration=True))
 print('ALL_NINE_CONFIGURATIONS_UNION_AND_CONFUSION_VERIFIED',flush=True)


if __name__=='__main__':main()
