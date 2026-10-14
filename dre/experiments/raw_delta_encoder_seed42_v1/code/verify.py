"""Independent aggregate/checkpoint verification. No Raw/model/Outer inference."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from io_utils import sha,write,read
from dependencies import context
from evaluate import independent,METRICS
from bootstrap import compare

def main(a):
 runtime=a.runtime;pub=runtime/'results';root=Path(__file__).parents[1];c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,runtime)
 aa=json.loads((pub/'STAGE_A_GATE.json').read_text());bb=json.loads((pub/'STAGE_B_GATE.json').read_text());assert aa['pass_gate'] and bb['pass_gate']
 for f,h in aa['binding']['code'].items():assert sha(root/'code'/f)==h
 for f,h in bb['binding']['code'].items():assert sha(root/'code'/f)==h
 assert sha(root/'tests/test_stage_a.py')==aa['binding']['tests'] and sha(root/'tests/test_stage_b.py')==bb['binding']['test']
 assert np.array_equal(rec['E14'],rec['archived_E14']) and np.isfinite(rec['R14']).all();stageB=json.loads((runtime/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text());computed=[];cells=0
 stageC=json.loads((pub/'STAGE_C_GATE.json').read_text());assert stageC['binding']['Stage_B_freeze']==sha(runtime/'STAGE_B_MODEL_FREEZE_PRIVATE.json')
 for f,h in stageC['binding']['code'].items():assert sha(root/'code'/f)==h
 assert sha(root/'tests/test_stage_c.py')==stageC['binding']['tests'];selections=json.loads((runtime/'STAGE_C_MODEL_FREEZE_PRIVATE.json').read_text())['selections']
 for fold,d in enumerate(banks,1):
  va=d['val'];predictions={}
  for arm in ['D0','E1','D1']:
   q=read(runtime/f'scores/{arm}_fold{fold}_PRIVATE.pkl');predictions[arm]=(q['score'],q['threshold'])
   if arm!='D0':
    r=next(r for r in stageB['selections'] if r['arm']==arm and r['fold']==fold);cp=runtime/f'{arm}/fold{fold}/TRAIN/BEST_PRIVATE.pt';bp=cp.parent.parent/'BANK_PRIVATE.pt';assert sha(cp)==r['checkpoint_sha256'] and sha(bp)==r['bank_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt');assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold'];cells+=1
  for arm in ['D2','D3']:
   r=next(r for r in selections if r['arm']==arm and r['fold']==fold);cp=runtime/f'{arm}/fold{fold}/TRAIN/BEST_PRIVATE.pt';q=read(runtime/f'scores/{arm}_fold{fold}_DYNAMIC_PRIVATE.pkl');assert sha(cp)==r['checkpoint_sha256']==q['checkpoint_sha256'];assert r['D1_checkpoint_sha256']==sha(runtime/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt');ck=torch.load(cp,map_location='cpu',weights_only=False);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt');assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold'];cells+=1
   predictions[arm]=(torch.sigmoid(torch.from_numpy(q['logit'])).numpy(),r['threshold'])
   if arm=='D2':
    for key,value in q['interventions'].items():predictions[f'D2_{key}']=(torch.sigmoid(torch.from_numpy(value)).numpy(),r['threshold'])
    predictions['D2_auxiliary_head_diagnostic']=(torch.sigmoid(torch.from_numpy(q['raw_logit'])).numpy(),.5)
   if arm=='D3':assert r['initial_state_sha256']==next(v['initial_state_sha256'] for v in selections if v['arm']=='D2' and v['fold']==fold)
  for arm,(score,tau) in predictions.items():
   assert np.isfinite(score).all()
   for pid in sorted(set(d['patient'][va])):
    k=np.flatnonzero(d['patient'][va]==pid);k=k[np.argsort(d['channel'][va][k].astype(str),kind='stable')];ii=va[k];computed.append(dict(arm=arm,fold=fold,patient=str(pid),center=str(d['center'][ii[0]]),**independent(d['y'][ii],score[k],tau)))
 df=pd.DataFrame(computed);saved=pd.DataFrame(read(runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl'));key=['arm','fold','patient'];l=df.set_index(key).sort_index();r=saved.set_index(key).sort_index();assert l.index.equals(r.index);assert np.allclose(l[METRICS],r[METRICS],atol=1e-12,rtol=0,equal_nan=True)
 for name,keys in [('DEVELOPMENT_SUMMARY.csv',['arm']),('DEVELOPMENT_BY_FOLD.csv',['arm','fold']),('DEVELOPMENT_BY_CENTER.csv',['arm','center'])]:
  s=pd.read_csv(pub/name).set_index(keys).sort_index();v=df.groupby(keys)[METRICS].mean().sort_index();assert s.index.equals(v.index);assert np.allclose(s[METRICS],v[METRICS],atol=1e-12,rtol=0,equal_nan=True)
 s=pd.read_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv').set_index('arm').sort_index();v=df.groupby(['arm','patient'])[METRICS].mean().groupby('arm').mean();assert np.allclose(s[METRICS],v,atol=1e-12,rtol=0,equal_nan=True)
 boot=pd.read_csv(pub/'PAIRED_BOOTSTRAP.csv')
 for left,right in [('D1','E1'),('E1','D0'),('D1','D0'),('D2','D1'),('D2','D3')]:
  bb=compare(c,df[df.arm==left],df[df.arm==right],METRICS)
  for v in bb:
   row=boot[(boot.comparison==f'{left} minus {right}')&(boot.metric==v['metric'])].iloc[0];assert np.allclose(row[['delta','ci_low','ci_high']].to_numpy(float),[v['delta'],v['ci_low'],v['ci_high']],atol=1e-12,rtol=0,equal_nan=True)
 assert cells==20 and all((len(q),q.patient.nunique())==(65,47) for _,q in df.groupby('arm'))
 write(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',new_training_cells=cells,patient_metric_cells=len(df),per_arm_appearances=65,per_arm_unique_IDs=47,independent_sklearn_metrics=True,independent_bootstrap_multiplicity=True,bootstrap_draws=10000,seed=42,no_model_or_Raw_rerun=True,no_Outer_access=True,protocol_sha256=sha(root/'PROTOCOL_LOCK.json'),source_sha256=sha(Path(__file__)),public_aggregate_sha256={p.name:sha(p) for p in pub.glob('*.csv')}));print('RAW_DELTA_INDEPENDENT_VERIFICATION_PASS',cells,len(df),flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
