"""Explicit user-authorized frozen outer extension; original development lock unchanged."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
sys.path.insert(0,str(Path(__file__).parents[1]/'code'))
from runtime import setup,checkpoint,load
from evaluate import interaction_tables
from verify import independent
from bootstrap import independently_verify


def context():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','evaluate','verify']);p.add_argument('--runtime',type=Path,default=Path(r'C:\a0_purn_seed42_outer_runtime'))
 p.add_argument('--development',type=Path,default=Path(r'C:\a0_purn_seed42_v1_runtime'));p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
 a=p.parse_args();root=Path(__file__).parents[1]
 da=SimpleNamespace(runtime=a.development,source=a.source,original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'),
  previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'),features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),protocol=root/'PROTOCOL_LOCK.json')
 o=setup(da);lock,ref,c,core,original,trainer,models,fm,banks,binding=o
 ampath=root/'OUTER_PROTOCOL_AMENDMENT.json';am=json.loads(ampath.read_text())
 dp=da.runtime/'DEVELOPMENT_FREEZE.json';assert c.sha(da.protocol)==am['development_protocol_sha256'] and c.sha(dp)==am['development_freeze_sha256']
 assert c.sha(da.runtime/'public/INDEPENDENT_VERIFICATION.json')==am['development_verification_sha256']
 assert json.loads((da.runtime/'public/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
 frozen=json.loads(dp.read_text());assert frozen['binding']==binding
 for name in ['PRETRAIN_AUDIT','ENGINEERING_TESTS']:
  ad=json.loads((da.runtime/f'public/{name}.json').read_text());assert ad['status']=='PASS' and ad['binding']==binding
 assert not json.loads((da.runtime/'public/VALIDATION_GATE.json').read_text())['pass_gate'] and am['development_gate_remains_failed']
 helper=da.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py';assert c.sha(helper)==am['frozen_transform_source_sha256']
 hist=load('purn_outer_historical_transform',helper)
 oldseal=da.previous/'OUTER_PREDICTIONS_SEAL.json';assert c.sha(oldseal)==am['B0_outer_seal_sha256']
 oldfiles=json.loads(oldseal.read_text())['files']
 paths={'amendment':ampath,'development_freeze':dp,'development_verification':da.runtime/'public/INDEPENDENT_VERIFICATION.json',
  'development_gate':da.runtime/'public/VALIDATION_GATE.json','transform':helper,'B0_outer_seal':oldseal,'code':Path(__file__),
  'contract':root/'outer_tests/contract.py'}
 for rec in frozen['selections']:
  arm,fold=rec['arm'],rec['fold'];cp=checkpoint(da,arm,fold);bp=da.runtime/f'fold{fold}/BANK_PRIVATE.pt';vp=bp.parent/f'{arm}_VAL_PRIVATE.pt'
  assert c.sha(cp)==rec['checkpoint_sha256'] and c.sha(bp)==rec['bank_sha256'];paths.update({f'{arm}_ck{fold}':cp,f'bank{fold}':bp,f'{arm}_val{fold}':vp})
  ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']==rec['epoch'] and ck['threshold']==rec['threshold']
 for fold in range(1,6):
  pp=da.previous/f'no_source/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt';assert c.sha(pp)==oldfiles[f'no_source_{fold}'];paths[f'B0_outer{fold}']=pp
 a.runtime.mkdir(parents=True,exist_ok=True);(a.runtime/'public').mkdir(exist_ok=True)
 return a,da,o,am,paths,frozen,hist


def prepare(ctx):
 a,da,o,am,paths,frozen,hist=ctx;lock,ref,c,core,original,trainer,models,fm,banks,binding=o
 assert not (a.runtime/'FROZEN_BEFORE_OUTER.json').exists()
 before={k:c.sha(v) for k,v in paths.items()};raw=hist.read_raw(da.features);rows=[];visited=[]
 assert raw['x'].shape==(7635,88) and len(set(raw['patient']))==80 and len(set(zip(raw['patient'],raw['channel'])))==7635
 for fold in range(1,6):
  d=torch.load(paths[f'bank{fold}'],map_location='cpu',weights_only=False);ids=hist.role_ids(raw,fold)
  di=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']|ids['validation'])));ti=np.flatnonzero(np.isin(raw['patient'],list(ids['test'])))
  for key in ['patient','channel','center']:assert np.array_equal(raw[key][di],d[key])
  assert np.array_equal(raw['features'],d['features']) and set(d['pre']['fit_patient_ids'])==ids['fit'] and not set(d['patient'])&ids['test']
  assert len(ids['test'])==am['test_patients_by_fold'][fold-1] and len(ti)==am['test_channels_by_fold'][fold-1];visited.extend(ti.tolist())
  x=hist.frozen_transform(raw['x'][di],d['patient'],d['pre'],core);assert np.array_equal(x,d['x'])
  for arm in lock['arms']:
   rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold);ck=torch.load(paths[f'{arm}_ck{fold}'],map_location='cpu',weights_only=False)
   m=models.constructor(original.SourceA0,arm,d['mapping'])().eval();m.load_state_dict(ck['model'],strict=True)
   assert sum(p.numel() for p in m.parameters())==8817
   val=torch.load(paths[f'{arm}_val{fold}'],map_location='cpu',weights_only=False)
   with torch.inference_mode():ll=m(torch.from_numpy(x[d['val']]));ss=torch.sigmoid(ll).numpy()
   ld=float(np.abs(ll.numpy()-val['logits']).max());sd=float(np.abs(ss-val['scores']).max())
   assert ld<=am['logit_replay_absolute_tolerance'] and sd<=am['probability_replay_absolute_tolerance']
   assert np.array_equal(ss>=rec['threshold'],val['scores']>=rec['threshold'])
   rows.append(dict(arm=arm,fold=fold,epoch=rec['epoch'],threshold=rec['threshold'],checkpoint_sha256=c.sha(paths[f'{arm}_ck{fold}']),
    bank_sha256=c.sha(paths[f'bank{fold}']),preprocessor_bitwise=True,logit_replay_drift=ld,probability_replay_drift=sd,outer_patients=len(ids['test']),outer_channels=len(ti),outer_overlap=0))
 assert sorted(visited)==list(range(7635)) and before=={k:c.sha(v) for k,v in paths.items()}
 c.write_json(a.runtime/'FROZEN_BEFORE_OUTER.json',dict(status='PASS',hashes=before,development_binding=binding,rows=rows,full_y_accessed=False,all20_selections_frozen=True))
 c.write_json(a.runtime/'public/FROZEN_DEPENDENCY_AUDIT.json',dict(status='PASS',hashes=before,rows=rows,full_y_accessed=False,all20_selections_frozen=True))
 print('PURN_ALL_OUTER_SELECTIONS_FROZEN_LABELS_UNREAD',flush=True)


def compare(c,left,right,metrics):
 ll=left.set_index(['fold','patient']).sort_index();rr=right.set_index(['fold','patient']).sort_index();assert ll.index.equals(rr.index)
 ids=ll.index.get_level_values('patient').to_numpy();assert len(ids)==len(set(ids))==80
 delta=ll[metrics].to_numpy()-rr[metrics].to_numpy();boot=c.cluster_bootstrap(ids,delta,10000)
 return [dict(metric=m,delta=float(np.nanmean(delta[:,i])),ci_low=float(np.nanpercentile(boot[:,i],2.5)),ci_high=float(np.nanpercentile(boot[:,i],97.5)),
  draws=10000,seed=42,unique_clusters=80,thresholds_reoptimized=False) for i,m in enumerate(metrics)]


def evaluate(ctx):
 a,da,o,am,paths,frozen,hist=ctx;lock,ref,c,core,original,trainer,models,fm,banks,binding=o;pub=a.runtime/'public'
 assert not (pub/'OUTER_STATUS.json').exists(),'Completed scientific evaluation cannot rerun'
 fr=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text());assert fr['development_binding']==binding and fr['hashes']=={k:c.sha(v) for k,v in paths.items()}
 fs=c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json');raw=hist.read_raw(da.features);cells={};files={}
 for fold in range(1,6):
  d=torch.load(paths[f'bank{fold}'],map_location='cpu',weights_only=False);ids=hist.role_ids(raw,fold);ix=np.flatnonzero(np.isin(raw['patient'],list(ids['test'])))
  x=hist.frozen_transform(raw['x'][ix],raw['patient'][ix],d['pre'],core)
  for arm in lock['arms']:
   rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold)
   if arm=='B0':path=paths[f'B0_outer{fold}'];cell=torch.load(path,map_location='cpu',weights_only=False)
   else:
    path=a.runtime/f'{arm}_fold{fold}_OUTER_PRIVATE.pt'
    if path.exists():
     assert json.loads(path.with_suffix('.sha256.json').read_text())==dict(binding=fs,sha256=c.sha(path));cell=torch.load(path,map_location='cpu',weights_only=False)
     assert cell['binding']==fs and cell['checkpoint_sha256']==rec['checkpoint_sha256']
    else:
     ck=torch.load(paths[f'{arm}_ck{fold}'],map_location='cpu',weights_only=False);m=models.constructor(original.SourceA0,arm,d['mapping'])().eval();m.load_state_dict(ck['model'],strict=True)
     with torch.inference_mode():ll=m(torch.from_numpy(x));score=torch.sigmoid(ll).numpy()
     cell=dict(binding=fs,indices=ix,logits=ll.numpy(),scores=score,threshold=rec['threshold'],checkpoint_sha256=rec['checkpoint_sha256'])
     core.torch_write(path,cell);c.write_json(path.with_suffix('.sha256.json'),dict(binding=fs,sha256=c.sha(path)))
   assert np.array_equal(cell['indices'],ix) and cell['threshold']==rec['threshold'] and np.isfinite(cell['scores']).all() and 'y' not in cell
   cells[arm,fold]=cell;files[f'{arm}_{fold}']=c.sha(path)
  print('PURN_OUTER_FOLD_SCORES_SEALED',fold,flush=True)
 c.write_json(a.runtime/'OUTER_PREDICTION_SEAL.json',dict(binding=fs,files=files,all15_new_predictions_before_full_y=True))
 with np.load(da.features,allow_pickle=True) as f:y=f['y']
 assert y.shape==(7635,) and int((y==0).sum())==1743 and int((y==1).sum())==5892
 rows=[];metrics=c.METRICS+['predicted_ez_fraction','TP','FP','TN','FN'];pooled=[]
 for arm in lock['arms']:
  all_scores=np.empty(7635);all_pred=np.zeros(7635,bool);visited=[]
  for fold in range(1,6):
   cell=cells[arm,fold];ix=cell['indices'];visited.extend(ix.tolist());all_scores[ix]=cell['scores'];all_pred[ix]=cell['scores']>=cell['threshold']
   d=banks[fold-1];di=np.flatnonzero(np.isin(raw['patient'],d['patient']));assert np.array_equal(y[di],d['y'])
   for p in sorted(set(raw['patient'][ix])):
    k=np.flatnonzero(raw['patient'][ix]==p);k=k[np.argsort(raw['channel'][ix][k].astype(str),kind='stable')];si=ix[k]
    row=dict(arm=arm,patient=p,fold=fold,center=str(raw['center'][si[0]]),**c.metric(y[si],cell['scores'][k],cell['threshold']))
    row['channel_stratum']='<64' if len(si)<64 else '64to127' if len(si)<128 else '>=128';rows.append(row)
  assert sorted(visited)==list(range(7635))
  order=np.lexsort((raw['channel'].astype(str),raw['patient'].astype(str)));q=c.metric(y[order],all_scores[order],0.,pred=all_pred[order]);q.pop('ez_mrr');q.pop('top1_is_ez');pooled.append(dict(arm=arm,unit='secondary_pooled_channels',**q))
 df=pd.DataFrame(rows);assert len(df)==320 and df.patient.nunique()==80
 assert all(len(q)==q.patient.nunique()==80 and int(q.channels.sum())==7635 for _,q in df.groupby('arm'))
 core.torch_write(a.runtime/'OUTER_PATIENT_METRICS_PRIVATE.pt',rows)
 summary=df.groupby('arm')[metrics].mean().reset_index();summary['parameters']=8817;summary.to_csv(pub/'OUTER_SUMMARY.csv',index=False)
 for metric,value in {'ez_auroc':.739768,'ez_auprc':.537887,'macro_f1':.624919,'ez_f1':.419405}.items():assert abs(summary.set_index('arm').loc['B0',metric]-value)<5e-7
 for key,label in [('fold','FOLD'),('center','CENTER'),('channel_stratum','CHANNEL_STRATA')]:
  group=[dict(arm=arm,**{key:v},patients=len(part),channels=int(part.channels.sum()),**part[metrics].mean().to_dict()) for (arm,v),part in df.groupby(['arm',key])]
  pd.DataFrame(group).to_csv(pub/('OUTER_BY_'+label+'.csv'),index=False)
 pd.DataFrame(pooled).to_csv(pub/'OUTER_POOLED_CHANNEL_METRICS.csv',index=False)
 effects=[]
 for left,right in am['comparisons']:effects.extend(dict(comparison=left+' minus '+right,**r) for r in compare(c,df[df.arm==left],df[df.arm==right],metrics))
 pd.DataFrame(effects).to_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv',index=False)
 ll,rr=interaction_tables(df,metrics);inter=[dict(comparison='(B3-B2)-(B1-B0)',**r) for r in compare(c,ll,rr,metrics)]
 pd.DataFrame(effects+inter).to_csv(pub/'OUTER_FACTORIAL_COMPARISONS.csv',index=False)
 assert fr['hashes']=={k:c.sha(v) for k,v in paths.items()} and setup(da)[-1]==binding
 c.write_json(pub/'IMMUTABILITY_AND_COVERAGE_AUDIT.json',dict(status='PASS',no_retraining=True,no_checkpoint_change=True,no_threshold_reselection=True,
  original_development_gate_unchanged_failed=True,all15_new_predictions_before_full_y=True,patients_per_arm=80,channels_per_arm=7635,EZ=1743,NEZ=5892,
  FIT_VAL_export_label_parity=True,all_dependencies_unchanged=True,freeze_sha256=fs,seal_sha256=c.sha(a.runtime/'OUTER_PREDICTION_SEAL.json')))
 c.write_json(pub/'OUTER_STATUS.json',dict(status='ONE_FROZEN_PURN_FOUR_ARM_OUTER_PASS_COMPLETE',scientific_status=am['scientific_status'],new_inference_cells=15,reused_B0_cells=5,patients=80,channels=7635,development_gate_unchanged_failed=True))
 print('ONE_FROZEN_PURN_FOUR_ARM_OUTER_PASS_COMPLETE',flush=True)


def verify(ctx):
 a,da,o,am,paths,frozen,hist=ctx;lock,ref,c,core,original,trainer,models,fm,banks,binding=o;pub=a.runtime/'public'
 fr=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text());assert fr['hashes']=={k:c.sha(v) for k,v in paths.items()}
 seal=json.loads((a.runtime/'OUTER_PREDICTION_SEAL.json').read_text());assert seal['binding']==c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json')
 with np.load(da.features,allow_pickle=True) as f:raw={k:f[k] for k in ['y','patient','channel','center']}
 rows=[];metrics=c.METRICS+['predicted_ez_fraction','TP','FP','TN','FN'];drift=0.
 for arm in lock['arms']:
  visited=[]
  for fold in range(1,6):
   path=paths[f'B0_outer{fold}'] if arm=='B0' else a.runtime/f'{arm}_fold{fold}_OUTER_PRIVATE.pt';assert c.sha(path)==seal['files'][f'{arm}_{fold}']
   cell=torch.load(path,map_location='cpu',weights_only=False);ix=cell['indices'];visited.extend(ix.tolist())
   rec=next(r for r in frozen['selections'] if r['arm']==arm and r['fold']==fold);assert cell['threshold']==rec['threshold']
   for p in sorted(set(raw['patient'][ix])):
    k=np.flatnonzero(raw['patient'][ix]==p);k=k[np.argsort(raw['channel'][ix][k].astype(str),kind='stable')];si=ix[k]
    im=independent(raw['y'][si],cell['scores'][k],cell['threshold']);cm=c.metric(raw['y'][si],cell['scores'][k],cell['threshold'])
    for name in im:
     assert np.isclose(im[name],cm[name],atol=1e-12,rtol=0,equal_nan=True)
     if np.isfinite(im[name]):drift=max(drift,abs(im[name]-cm[name]))
    row=dict(arm=arm,fold=fold,patient=p,center=str(raw['center'][si[0]]),**im);row['channel_stratum']='<64' if len(si)<64 else '64to127' if len(si)<128 else '>=128';rows.append(row)
  assert sorted(visited)==list(range(7635))
 df=pd.DataFrame(rows);assert len(df)==320 and df.patient.nunique()==80
 for filename,keys in [('OUTER_SUMMARY.csv',['arm']),('OUTER_BY_FOLD.csv',['arm','fold']),('OUTER_BY_CENTER.csv',['arm','center']),('OUTER_BY_CHANNEL_STRATA.csv',['arm','channel_stratum'])]:
  actual=pd.read_csv(pub/filename).set_index(keys).sort_index();expected=df.groupby(keys)[metrics].mean().sort_index()
  assert actual.index.equals(expected.index) and np.allclose(actual[metrics],expected,atol=1e-12,rtol=0,equal_nan=True)
 for filename in ['OUTER_PAIRED_BOOTSTRAP.csv','OUTER_FACTORIAL_COMPARISONS.csv']:
  actual=pd.read_csv(pub/filename)
  for left,right in am['comparisons']:
   calc=independently_verify(df[df.arm==left],df[df.arm==right],metrics)
   for metric,v in calc.items():
    rr=actual[(actual.comparison==left+' minus '+right)&(actual.metric==metric)].iloc[0];assert np.allclose(v,[rr.delta,rr.ci_low,rr.ci_high],atol=1e-12,rtol=0,equal_nan=True)
 ll,rr=interaction_tables(df,metrics);calc=independently_verify(ll,rr,metrics);actual=pd.read_csv(pub/'OUTER_FACTORIAL_COMPARISONS.csv')
 for metric,v in calc.items():
  rr=actual[(actual.comparison=='(B3-B2)-(B1-B0)')&(actual.metric==metric)].iloc[0];assert np.allclose(v,[rr.delta,rr.ci_low,rr.ci_high],atol=1e-12,rtol=0,equal_nan=True)
 assert setup(da)[-1]==binding
 c.write_json(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',metric_cells=320,max_metric_drift=drift,verified_contrast_intervals=96,
  draws=10000,unique_clusters=80,checkpoint_bank_threshold_prediction_source_checks=True,all_aggregate_parity=True,development_gate_unchanged_failed=True,no_new_model_inference=True))
 print('PURN_OUTER_INDEPENDENT_VERIFICATION_PASS',flush=True)


if __name__=='__main__':
 ctx=context();{'prepare':prepare,'evaluate':evaluate,'verify':verify}[ctx[0].phase](ctx)
