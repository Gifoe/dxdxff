"""Aggregate-only independent delivery arithmetic; no private data or inference."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
ROOT=Path(__file__).parents[1]

def test_summary_matches_fold_and_center_aggregation():
 metrics=json.loads((ROOT/'PROTOCOL_LOCK.json').read_text())['metrics']
 for folder,summaryfile,foldfile,centerfile,weights,cweights in [
  ('results','INNER_EVALUATION_SUMMARY.csv','INNER_EVALUATION_BY_ORIGINAL_FOLD.csv','INNER_EVALUATION_BY_CENTER.csv',[51,51,50,52,51],{'hup':116,'lzu':70,'multicenter':49,'pediatric':20}),
  ('outer_results','OUTER_PATIENT_EQUAL_METRICS.csv','OUTER_BY_FOLD.csv','OUTER_BY_CENTER.csv',[16,16,17,15,16],{'hup':36,'lzu':21,'multicenter':15,'pediatric':8})]:
  p=ROOT/folder;s=pd.read_csv(p/summaryfile).set_index('arm');f=pd.read_csv(p/foldfile);c=pd.read_csv(p/centerfile)
  for arm in ['C0','C1','C2']:
   q=f[f.arm==arm].sort_values('fold');actual=np.average(q[metrics],axis=0,weights=weights);assert np.allclose(actual,s.loc[arm,metrics].astype(float),atol=1e-12,rtol=0)
   q=c[c.arm==arm];actual=np.average(q[metrics],axis=0,weights=[cweights[k] for k in q.center]);assert np.allclose(actual,s.loc[arm,metrics].astype(float),atol=1e-12,rtol=0)

def test_frozen_bootstrap_points_and_counts():
 for folder,filename,total in [('results','INNER_EVALUATION_SUMMARY.csv',79),('outer_results','OUTER_PATIENT_EQUAL_METRICS.csv',80)]:
  p=ROOT/folder;s=pd.read_csv(p/filename).set_index('arm');b=pd.read_csv(p/('PAIRED_BOOTSTRAP.csv' if total==79 else 'OUTER_PAIRED_BOOTSTRAP.csv'))
  for row in b.itertuples():
   left,right=row.comparison.split(' minus ');assert np.isclose(row.delta,s.loc[left,row.metric]-s.loc[right,row.metric],atol=1e-12,rtol=0)
   assert row.unique_patient_clusters==total and row.draws==10000 and not row.threshold_reoptimized
 c=pd.read_csv(ROOT/'outer_results/OUTER_CONFUSION_COUNTS.csv');assert ((c.TP+c.FP+c.TN+c.FN)==7635).all();assert ((c.TP+c.FN)==1743).all();assert ((c.FP+c.TN)==5892).all()

def test_gate_and_completed_status():
 inner=json.loads((ROOT/'results/SCIENTIFIC_GATE.json').read_text());outer=json.loads((ROOT/'outer_results/OUTER_GATE.json').read_text())
 assert inner['pass_all'] and not inner['independent_verification_pending']
 assert not outer['all_pass'] and [k for k,v in outer['conditions'].items() if not v]==['C1_C2_AUROC_CI_positive']
 assert json.loads((ROOT/'results/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((ROOT/'outer_results/OUTER_VERIFICATION.json').read_text())['status']=='PASS'
 assert json.loads((ROOT/'outer_results/OUTER_STATUS.json').read_text())['status']=='COMPLETE'
