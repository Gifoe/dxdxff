"""Two sole-view arms. Original trainer; legal banks; freeze then score seal then y."""
import argparse
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from dependencies import dependencies,load
from metrics import independent
from bootstrap import compare,independent as independent_bootstrap

NEW=['DELTA_ONLY','RATIO_ONLY']
REFS=['F0','ABS_ONLY']

def arguments():
 p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','train','evaluate','verify'])
 p.add_argument('--runtime',type=Path,default=Path(r'D:\a0_compact_delta_ratio_only_seed42_v1_runtime'))
 p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
 a=p.parse_args();root=Path(__file__).parents[1]
 return SimpleNamespace(**vars(a),root=root,protocol=root/'PROTOCOL_LOCK.json',original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'),
  previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'),compact=Path(r'C:\d2_no_source_no_zdelta_outer_seed42_v1_runtime'),
  absroot=Path(r'D:\a0_compact_rawfirst_ablation_seed42_v1_runtime'),features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'))

def context(a):
 ref,c,core,original,trainer,helper,banks,base=dependencies(a);lock=json.loads(a.protocol.read_text())
 assert list(lock['matrix'])==REFS+NEW and lock['new_cells']==10
 paths={'protocol':a.protocol,'manifest':a.previous/'public/FEATURE_GROUP_MANIFEST.json','outer_transform':a.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py'}
 assert c.sha(paths['manifest'])==lock['feature_manifest_sha256'] and c.sha(paths['outer_transform'])==lock['outer_transform_sha256']
 for arm,root,names in [('F0',a.compact,{'freeze':'FROZEN_BEFORE_OUTER.json','seal':'OUTER_PREDICTIONS_SEAL.json'}),
   ('ABS_ONLY',a.absroot,{'freeze':'FEATURE_FROZEN_BEFORE_OUTER.json','seal':'FEATURE_OUTER_PREDICTION_SEAL.json','independent':'public/INDEPENDENT_VERIFICATION.json'})]:
  for key,file in names.items():
   tag=arm+'_'+key;paths[tag]=root/file;assert c.sha(paths[tag])==lock['historical_hashes'][tag]
 assert json.loads(paths['ABS_ONLY_independent'].read_text())['status']=='PASS'
 for folder in ['code','tests']:
  for p in sorted((a.root/folder).glob('*.py')):paths[folder+'/'+p.name]=p
 bind=dict(original=base,new={k:c.sha(p) for k,p in paths.items()})
 return lock,ref,c,core,original,trainer,helper,banks,json.loads(paths['manifest'].read_text()),bind

def mask_for(lock,manifest,arm):
 cols=manifest['columns'];assert len(cols)==88 and [q['index'] for q in cols]==list(range(88))
 mask=np.array([q['view'] in lock['matrix'][arm]['views'] for q in cols]);assert mask.sum()==lock['matrix'][arm]['retained']
 return mask

def historical(a,arm,fold):
 if arm=='F0':
  fr=json.loads((a.compact/'FROZEN_BEFORE_OUTER.json').read_text());r=fr['selections'][fold-1];root=a.compact/f'fold{fold}'
  pp=root/'OUTER_PREDICTIONS_PRIVATE.pt';seal=json.loads((a.compact/'OUTER_PREDICTIONS_SEAL.json').read_text())['files'][f'w/o Source + w/o ZDELTA_{fold}']
  return root/'BANK_PRIVATE.pt',root/'TRAIN/BEST_PRIVATE.pt',pp,dict(r,threshold=r['nez_threshold']),seal
 fr=json.loads((a.absroot/'FEATURE_FROZEN_BEFORE_OUTER.json').read_text());r=next(r for r in fr['selections'] if r['arm']=='F2' and r['fold']==fold)
 root=a.absroot/f'features/F2/fold{fold}';pp=a.absroot/f'FEATURE_scores/F2_fold{fold}_OUTER_PRIVATE.pt'
 return root/'BANK_PRIVATE.pt',root/'TRAIN/BEST_PRIVATE.pt',pp,r,json.loads((a.absroot/'FEATURE_OUTER_PREDICTION_SEAL.json').read_text())['files'][f'F2_{fold}']

def metadata(a):
 with np.load(a.features,allow_pickle=True) as f:return {k:f[k] for k in ['patient','channel','center','features','split_patient','split_role','split_fold']}

def rows(c,d,ix,scores,tau,arm,fold):
 out=[]
 for pid in sorted(set(d['patient'][ix])):
  k=np.flatnonzero(d['patient'][ix]==pid);k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')];ii=ix[k]
  out.append(dict(arm=arm,patient=pid,fold=fold,center=str(d['center'][ii[0]]),**c.metric(d['y'][ii],scores[k],tau)))
 return out

def check_metrics(c,d,ix,score,tau):
 for pid in sorted(set(d['patient'][ix])):
  k=np.flatnonzero(d['patient'][ix]==pid);k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')]
  left=c.metric(d['y'][ix[k]],score[k],tau);right=independent(d['y'][ix[k]],score[k],tau)
  for name in right:assert np.isclose(left[name],right[name],atol=1e-12,rtol=0,equal_nan=True),(name,left[name],right[name])

def paths(a,arm,fold):
 root=a.runtime/f'features/{arm}/fold{fold}';return root/'BANK_PRIVATE.pt',root/'TRAIN/BEST_PRIVATE.pt'

def prepare(a):
 o=context(a);lock,ref,c,core,original,trainer,helper,banks,manifest,bind=o;pub=a.runtime/'public'
 assert not (a.runtime/'EXPERIMENT_FREEZE.json').exists()
 tests=[];admission=[];dev=[];historical_files={};fac=helper.source_free_factory(original.SourceA0)
 for arm in REFS+NEW:
  mask=mask_for(lock,manifest,arm)
  for fold,d in enumerate(banks,1):
   assert helper.parse_names(d['features'])==[{k:q[k] for k in ['index','name','block','view','physiological_feature','aggregation']} for q in manifest['columns']]
   x=helper.masked(d['x'],mask);assert np.array_equal(x[:,mask],d['x'][:,mask]) and np.all(x[:,~mask]==0)
   if arm in NEW:
    bp,_=paths(a,arm,fold);assert not bp.exists();core.torch_write(bp,dict(d,x=x,mask=mask,arm=arm))
    admission.append(dict(arm=arm,fold=fold,retained=int(mask.sum()),bank_sha256=c.sha(bp),parameters=8817))
   else:
    bp,cp,pp,r,ps=historical(a,arm,fold)
    assert c.sha(bp)==r['bank_sha256'] and c.sha(cp)==r['checkpoint_sha256'] and c.sha(pp)==ps
    b=torch.load(bp,map_location='cpu',weights_only=False);assert np.array_equal(b['x'],x)
    for k in ['patient','channel','features','y','g','train','val']:assert np.array_equal(b[k],d[k])
    for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(b['pre'][k],d['pre'][k])
    ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['threshold']==r['threshold'] and ck['epoch']==r['epoch']
    seed=42+1009*fold;core.seed_all(seed);m=fac(seed).cuda().eval();assert core.state_hash(m.state_dict())==ck['binding']['initial_hash']
    assert sum(p.numel() for p in m.parameters())==8817 and isinstance(m.network[0],torch.nn.LayerNorm)
    m.load_state_dict(ck['model'],strict=True);va=d['val']
    with torch.inference_mode():ss=torch.sigmoid(m(torch.as_tensor(x[va],device='cuda'),torch.as_tensor(d['g'][va],device='cuda'))).cpu().numpy()
    sel=core.select_threshold(d['y'][va],ss,d['patient'][va])
    for k in ['threshold','macro_f1','ez_f1']:assert abs(sel[k]-ck[k])<1e-12
    vp=a.runtime/f'scores/{arm}_fold{fold}_VAL_PRIVATE.pt';core.torch_write(vp,dict(indices=va,scores=ss,threshold=r['threshold'],checkpoint_sha256=c.sha(cp)))
    dev.extend(rows(c,d,va,ss,r['threshold'],arm,fold))
    for key,p in [('bank',bp),('checkpoint',cp),('outer',pp),('validation',vp)]:historical_files[f'{arm}_{key}{fold}']=c.sha(p)
    del m
 tests.extend(['manifest_name_and_index_order','soleview_bitwise_retained_post_preprocess_mask','original_fit_val_roles_and_fit_only_scaler','historical_F0_ABS_checkpoint_mask_selection_and_outer_seal'])
 # Tiny deterministic synthetic training tests; no clinical selection/outcome.
 sx=np.random.default_rng(42).normal(size=(16,88)).astype(np.float32);sy=np.tile([0,1],8);pid=np.repeat(['s1','s2','s3','s4'],4);g=np.zeros(16,dtype=int)
 old=trainer.SourceA0;trainer.SourceA0=fac;root=a.runtime/'engineering/resume'
 try:
  trainer.train(sx,sy,pid,g,np.arange(8),np.arange(8,16),root/'continuous',1051,'synthetic',device='cpu',max_epochs=3,penalty=0.)
  trainer.train(sx,sy,pid,g,np.arange(8),np.arange(8,16),root/'resume',1051,'synthetic',device='cpu',max_epochs=3,stop_after=1,penalty=0.)
  trainer.train(sx,sy,pid,g,np.arange(8),np.arange(8,16),root/'resume',1051,'synthetic',device='cpu',max_epochs=3,penalty=0.)
 finally:trainer.SourceA0=old
 p,q=[torch.load(root/s/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False) for s in ['continuous','resume']]
 assert core.state_hash(p['model'])==core.state_hash(q['model']) and p['history']==q['history'] and p['best_key']==q['best_key']
 assert torch.equal(p['rng']['torch'],q['rng']['torch']) and all(torch.equal(v,w) for v,w in zip(p['rng']['cuda'],q['rng']['cuda']))
 for k,slot in p['optimizer']['state'].items():
  for n,v in slot.items():assert torch.equal(v,q['optimizer']['state'][k][n])
 tests.append('exact_synthetic_CPU_model_optimizer_history_RNG_resume')
 core.seed_all(1051);m=fac(1051).eval();assert sum(p.numel() for p in m.parameters())==8817
 with torch.inference_mode():assert torch.equal(m(torch.from_numpy(sx),torch.zeros(16,dtype=torch.long)),m(torch.from_numpy(sx),torch.full((16,),3,dtype=torch.long)))
 tests.append('8817_shared_initialization_and_source_code_ignored')
 core.torch_write(a.runtime/'HISTORICAL_DEVELOPMENT_PRIVATE.pt',dev)
 c.write_json(pub/'PRETRAIN_AUDIT.json',dict(status='PASS',binding=bind,tests=tests,test_count=len(tests),admission=admission,historical_files=historical_files,no_raw_access=True,no_new_outer_metrics=True))
 c.write_json(a.runtime/'EXPERIMENT_FREEZE.json',dict(status='PASS',binding=bind,matrix=lock['matrix'],historical_files=historical_files))
 print('PRETRAIN_PASS',flush=True)

def train(a):
 lock,ref,c,core,original,trainer,helper,banks,manifest,bind=context(a);pub=a.runtime/'public'
 assert json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text())['binding']==bind
 audit=json.loads((pub/'PRETRAIN_AUDIT.json').read_text());assert audit['status']=='PASS' and audit['binding']==bind
 assert not (a.runtime/'DEVELOPMENT_FREEZE.json').exists();sels=[];history=[];resource=[];fac=helper.source_free_factory(original.SourceA0)
 for fold in range(1,6):
  for arm in NEW:
   bp,cp=paths(a,arm,fold);assert c.sha(bp)==next(r['bank_sha256'] for r in audit['admission'] if r['arm']==arm and r['fold']==fold)
   d=torch.load(bp,map_location='cpu',weights_only=False);seed=42+1009*fold;core.seed_all(seed);initial=core.state_hash(fac(seed).state_dict())
   old=trainer.SourceA0;trainer.SourceA0=fac;t=time.perf_counter();torch.cuda.reset_peak_memory_stats()
   try:trainer.train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],cp.parent,seed,c.digest(dict(binding=bind,bank=c.sha(bp),arm=arm)),device='cuda',penalty=0.)
   finally:trainer.SourceA0=old
   ck=torch.load(cp,map_location='cpu',weights_only=False);last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text())
   assert ck['binding']['initial_hash']==initial and c.sha(cp)==done['best_sha256'] and c.sha(cp.parent/'LAST_PRIVATE.pt')==done['last_sha256']
   sels.append(dict(arm=arm,fold=fold,epoch=ck['epoch'],threshold=ck['threshold'],bank_sha256=c.sha(bp),checkpoint_sha256=c.sha(cp),parameters=8817))
   history.extend(dict(arm=arm,fold=fold,**r) for r in last['history']);resource.append(dict(arm=arm,fold=fold,epochs=last['epoch'],wall_seconds=time.perf_counter()-t,gpu_peak_bytes=torch.cuda.max_memory_allocated()))
   c.write_json(pub/'TRAIN_PROGRESS.json',dict(status='RUNNING',cells_complete=len(sels),total=10));print('SELECTED',arm,fold,ck['epoch'],ck['threshold'],flush=True)
 assert context(a)[-1]==bind
 c.write_json(a.runtime/'DEVELOPMENT_FREEZE.json',dict(status='PASS',binding=bind,selections=sels))
 pd.DataFrame(sels).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.DataFrame(history).to_csv(pub/'TRAINING_HISTORY.csv',index=False);pd.DataFrame(resource).to_csv(pub/'RESOURCE_AUDIT.csv',index=False)
 c.write_json(pub/'TRAIN_PROGRESS.json',dict(status='ALL_TRAIN_VALIDATION_COMPLETE',cells_complete=10,total=10))

def aggregate(a,c,core,lock,rs,phase):
 df=pd.DataFrame(rs);metrics=lock['metrics'];pub=a.runtime/'public'
 for arm,q in df.groupby('arm'):
  assert (len(q),q.patient.nunique())==((65,47) if phase=='DEVELOPMENT' else (80,80))
  if phase=='OUTER':assert int(q.channels.sum())==7635
 df.groupby('arm')[metrics].mean().reset_index().to_csv(pub/f'{phase}_SUMMARY.csv',index=False)
 for key in ['fold','center']:
  pd.DataFrame([dict(arm=arm,**{key:v},patients_or_appearances=len(q),channels=int(q.channels.sum()),**q[metrics].mean().to_dict()) for (arm,v),q in df.groupby(['arm',key])]).to_csv(pub/f'{phase}_BY_{key.upper()}.csv',index=False)
 if phase=='DEVELOPMENT':df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
 boot=[]
 for spec in lock['bootstrap']['comparisons']:
  l,r=spec.split(' minus ');boot.extend(dict(comparison=spec,**v) for v in compare(c,df[df.arm==l],df[df.arm==r],metrics))
 pd.DataFrame(boot).to_csv(pub/f'{phase}_BOOTSTRAP.csv',index=False);core.torch_write(a.runtime/f'{phase}_METRICS_PRIVATE.pt',rs)

def evaluate(a):
 lock,ref,c,core,original,trainer,helper,banks,manifest,bind=context(a);pub=a.runtime/'public'
 assert not (pub/'OUTER_STATUS.json').exists(),'Completed evaluation cannot rerun'
 assert json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text())['binding']==bind
 fr=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert fr['binding']==bind and len(fr['selections'])==10
 dev=torch.load(a.runtime/'HISTORICAL_DEVELOPMENT_PRIVATE.pt',weights_only=False);hashes={};replay=[];fac=helper.source_free_factory(original.SourceA0)
 for r in fr['selections']:
  arm,fold=r['arm'],r['fold'];bp,cp=paths(a,arm,fold);assert c.sha(bp)==r['bank_sha256'] and c.sha(cp)==r['checkpoint_sha256']
  d=torch.load(bp,map_location='cpu',weights_only=False);ck=torch.load(cp,map_location='cpu',weights_only=False);va=d['val'];vp=a.runtime/f'scores/{arm}_fold{fold}_VAL_PRIVATE.pt'
  if vp.exists():
   cell=torch.load(vp,map_location='cpu',weights_only=False);assert json.loads(vp.with_suffix('.sha256.json').read_text())==dict(binding=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json'),sha256=c.sha(vp))
  else:
   m=fac().eval();m.load_state_dict(ck['model'],strict=True)
   with torch.inference_mode():ll=m(torch.from_numpy(d['x'][va]),torch.as_tensor(d['g'][va]));ss=torch.sigmoid(ll).numpy()
   m=m.cuda()
   with torch.inference_mode():vl=m(torch.as_tensor(d['x'][va],device='cuda'),torch.as_tensor(d['g'][va],device='cuda'));vs=torch.sigmoid(vl).cpu().numpy()
   ld=float(np.abs(ll.numpy()-vl.cpu().numpy()).max());sd=float(np.abs(ss-vs).max())
   assert ld<=lock['parity']['logit_cpu_cuda_atol'] and sd<=lock['parity']['score_cpu_cuda_atol'] and np.array_equal(ss>=ck['threshold'],vs>=ck['threshold'])
   sel=core.select_threshold(d['y'][va],vs,d['patient'][va])
   for k in ['threshold','macro_f1','ez_f1']:assert abs(sel[k]-ck[k])<1e-12
   cell=dict(indices=va,scores=ss,threshold=ck['threshold'],checkpoint_sha256=c.sha(cp),parity=dict(arm=arm,fold=fold,logit_drift=ld,score_drift=sd,selection_exact=True))
   core.torch_write(vp,cell);c.write_json(vp.with_suffix('.sha256.json'),dict(binding=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json'),sha256=c.sha(vp)));del m
  assert cell['checkpoint_sha256']==c.sha(cp) and np.array_equal(cell['indices'],va) and cell['threshold']==r['threshold']
  dev.extend(rows(c,d,va,cell['scores'],r['threshold'],arm,fold));replay.append(cell['parity'])
  for key,p in [('bank',bp),('checkpoint',cp),('validation',vp)]:hashes[f'{arm}_{key}{fold}']=c.sha(p)
 aggregate(a,c,core,lock,dev,'DEVELOPMENT')
 ff=a.runtime/'FROZEN_BEFORE_OUTER.json';frozen=dict(status='PASS',binding=bind,selections=fr['selections'],hashes=hashes,development_freeze_sha256=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json'),parity=replay,new_outer_unread=True)
 if ff.exists():assert json.loads(ff.read_text())==frozen
 else:c.write_json(ff,frozen)
 fs=c.sha(ff);c.write_json(pub/'FREEZE_AUDIT.json',frozen);meta=metadata(a);hist=load('soleview_outer',a.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py');cells={};files={}
 for fold in range(1,6):
  ids=set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']=='test')]);ix=np.flatnonzero(np.isin(meta['patient'],list(ids)))
  assert len(ids)==lock['outer_fold_patients'][fold-1] and len(ix)==lock['outer_fold_channels'][fold-1]
  for arm in NEW:
   r=next(r for r in fr['selections'] if r['arm']==arm and r['fold']==fold);bp,cp=paths(a,arm,fold);assert c.sha(cp)==r['checkpoint_sha256'];pp=a.runtime/f'scores/{arm}_fold{fold}_OUTER_PRIVATE.pt'
   if pp.exists():
    assert json.loads(pp.with_suffix('.sha256.json').read_text())==dict(binding=fs,sha256=c.sha(pp));cell=torch.load(pp,map_location='cpu',weights_only=False)
   else:
    d=torch.load(bp,map_location='cpu',weights_only=False);ck=torch.load(cp,map_location='cpu',weights_only=False)
    with np.load(a.features,allow_pickle=True) as f:xx=f['x'][ix]
    xx=helper.masked(hist.frozen_transform(xx,meta['patient'][ix],d['pre'],core),d['mask']);m=fac().eval();m.load_state_dict(ck['model'],strict=True)
    with torch.inference_mode():ss=torch.sigmoid(m(torch.from_numpy(xx),torch.zeros(len(ix),dtype=torch.long))).numpy()
    cell=dict(indices=ix,scores=ss,threshold=r['threshold'],binding=fs,checkpoint_sha256=c.sha(cp));core.torch_write(pp,cell);c.write_json(pp.with_suffix('.sha256.json'),dict(binding=fs,sha256=c.sha(pp)));del m
   assert cell['binding']==fs and cell['checkpoint_sha256']==c.sha(cp) and np.array_equal(cell['indices'],ix) and cell['threshold']==r['threshold'] and 'y' not in cell
   cells[arm,fold]=cell;files[f'{arm}_{fold}']=c.sha(pp)
  print('OUTER_FOLD_SEALED',fold,flush=True)
 c.write_json(a.runtime/'OUTER_PREDICTION_SEAL.json',dict(binding=fs,files=files,all10_new_scores_before_full_y=True))
 # First full archive y read in this process AFTER every new score is frozen.
 with np.load(a.features,allow_pickle=True) as f:y=f['y']
 assert len(y)==7635 and int((y==0).sum())==1743 and int((y==1).sum())==5892
 rs=[];pooled=[]
 for arm in REFS+NEW:
  visited=[];scores=np.empty(7635);pred=np.zeros(7635,bool)
  for fold in range(1,6):
   if arm in REFS:
    _,_,pp,r,ps=historical(a,arm,fold);assert c.sha(pp)==ps;cell=torch.load(pp,map_location='cpu',weights_only=False);tau=r['threshold']
   else:cell=cells[arm,fold];tau=cell['threshold']
   ix=cell['indices'];visited.extend(ix.tolist());ids=set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']=='test')]);assert set(meta['patient'][ix])==ids
   check_metrics(c,dict(meta,y=y),ix,cell['scores'],tau);rs.extend(rows(c,dict(meta,y=y),ix,cell['scores'],tau,arm,fold));scores[ix]=cell['scores'];pred[ix]=cell['scores']>=tau
  assert sorted(visited)==list(range(7635));order=np.lexsort((meta['channel'].astype(str),meta['patient'].astype(str)));q=c.metric(y[order],scores[order],0.,pred=pred[order]);q.pop('ez_mrr');q.pop('top1_is_ez');pooled.append(dict(arm=arm,unit='secondary_pooled_channels',**q))
 df=pd.DataFrame(rs);means=df.groupby('arm')[lock['metrics']].mean()
 assert np.allclose(means.loc['F0',['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr']],[.7369771785860808,.5306525381402717,.6242355736555875,.4146706708280264,.6942666410497293],atol=1e-12,rtol=0)
 assert np.allclose(means.loc['ABS_ONLY',['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr']],[.6990111964374394,.4828189044775727,.6053067386792244,.390291336268963,.6529692991960256],atol=1e-12,rtol=0)
 aggregate(a,c,core,lock,rs,'OUTER');pd.DataFrame(pooled).to_csv(pub/'OUTER_POOLED_CHANNELS.csv',index=False)
 assert context(a)[-1]==bind
 c.write_json(pub/'OUTER_STATUS.json',dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',new_cells=10,patients_per_arm=80,channels_per_arm=7635,freeze_sha256=fs,seal_sha256=c.sha(a.runtime/'OUTER_PREDICTION_SEAL.json'),historical_refs_reused=True,all_new_scores_before_full_y=True,scientific_status=lock['scientific_status']))
 print('ONE_FROZEN_OUTER_PASS_COMPLETE',flush=True)

def verify(a):
 lock,ref,c,core,original,trainer,helper,banks,manifest,bind=context(a);pub=a.runtime/'public'
 status=json.loads((pub/'OUTER_STATUS.json').read_text());assert status['status']=='ONE_FROZEN_OUTER_PASS_COMPLETE'
 ff=a.runtime/'FROZEN_BEFORE_OUTER.json';assert c.sha(ff)==status['freeze_sha256'];fr=json.loads(ff.read_text());assert fr['binding']==bind
 seal=json.loads((a.runtime/'OUTER_PREDICTION_SEAL.json').read_text());assert c.sha(a.runtime/'OUTER_PREDICTION_SEAL.json')==status['seal_sha256'] and seal['binding']==c.sha(ff)
 for r in fr['selections']:
  bp,cp=paths(a,r['arm'],r['fold']);d=torch.load(bp,map_location='cpu',weights_only=False);ck=torch.load(cp,map_location='cpu',weights_only=False)
  assert c.sha(bp)==r['bank_sha256'] and c.sha(cp)==r['checkpoint_sha256'] and ck['threshold']==r['threshold'] and ck['epoch']==r['epoch']
  assert np.array_equal(d['mask'],mask_for(lock,manifest,r['arm'])) and np.array_equal(d['x'],helper.masked(banks[r['fold']-1]['x'],d['mask']))
  done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==c.sha(cp) and done['last_sha256']==c.sha(cp.parent/'LAST_PRIVATE.pt')
  core.seed_all(42+1009*r['fold']);assert core.state_hash(helper.source_free_factory(original.SourceA0)(42+1009*r['fold']).state_dict())==ck['binding']['initial_hash']
  assert ck['binding']['device']=='cuda' and ck['binding']['penalty']==0 and ck['binding']['max_epochs']==30
  for k in ['patient','channel','y','g','train','val']:assert np.array_equal(d[k],banks[r['fold']-1][k])
  for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(d['pre'][k],banks[r['fold']-1]['pre'][k])
  p=a.runtime/f"scores/{r['arm']}_fold{r['fold']}_OUTER_PRIVATE.pt";assert c.sha(p)==seal['files'][f"{r['arm']}_{r['fold']}"]
 meta=metadata(a)
 historical_hashes=json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text())['historical_files']
 for arm in REFS:
  for fold in range(1,6):
   bp,cp,pp,r,ps=historical(a,arm,fold)
   for key,p in [('bank',bp),('checkpoint',cp),('outer',pp),('validation',a.runtime/f'scores/{arm}_fold{fold}_VAL_PRIVATE.pt')]:assert c.sha(p)==historical_hashes[f'{arm}_{key}{fold}']
 with np.load(a.features,allow_pickle=True) as f:y=f['y']
 checked=0
 for phase in ['DEVELOPMENT','OUTER']:
  rs=[]
  for arm in REFS+NEW:
   for fold in range(1,6):
    if phase=='DEVELOPMENT':
     d=banks[fold-1];p=a.runtime/f'scores/{arm}_fold{fold}_VAL_PRIVATE.pt'
    else:
     d=dict(meta,y=y);p=historical(a,arm,fold)[2] if arm in REFS else a.runtime/f'scores/{arm}_fold{fold}_OUTER_PRIVATE.pt'
    cell=torch.load(p,map_location='cpu',weights_only=False);ix=cell['indices'];tau=cell['threshold']
    for pid in sorted(set(d['patient'][ix])):
     k=np.flatnonzero(d['patient'][ix]==pid);k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')];ii=ix[k]
     rs.append(dict(arm=arm,patient=pid,fold=fold,center=str(d['center'][ii[0]]),**independent(d['y'][ii],cell['scores'][k],tau)));checked+=1
  df=pd.DataFrame(rs);saved=pd.DataFrame(torch.load(a.runtime/f'{phase}_METRICS_PRIVATE.pt',weights_only=False)).set_index(['arm','fold','patient']).sort_index();computed=df.set_index(['arm','fold','patient']).sort_index();assert saved.index.equals(computed.index)
  assert np.allclose(saved[lock['metrics']],computed[lock['metrics']],atol=1e-12,rtol=0,equal_nan=True)
  summary=pd.read_csv(pub/f'{phase}_SUMMARY.csv').set_index('arm');assert np.allclose(summary[lock['metrics']],df.groupby('arm')[lock['metrics']].mean(),atol=1e-12,rtol=0,equal_nan=True)
  for key in ['fold','center']:
   s=pd.read_csv(pub/f'{phase}_BY_{key.upper()}.csv').set_index(['arm',key]).sort_index();v=df.groupby(['arm',key])[lock['metrics']].mean();assert np.allclose(s[lock['metrics']],v,atol=1e-12,rtol=0,equal_nan=True)
  if phase=='DEVELOPMENT':
   s=pd.read_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv').set_index('arm');v=df.groupby(['arm','patient'])[lock['metrics']].mean().groupby('arm').mean();assert np.allclose(s[lock['metrics']],v,atol=1e-12,rtol=0,equal_nan=True)
  else:
   pooled=pd.read_csv(pub/'OUTER_POOLED_CHANNELS.csv').set_index('arm')
   for arm in REFS+NEW:
    score=np.empty(7635);decision=np.zeros(7635,bool)
    for fold in range(1,6):
     pp=historical(a,arm,fold)[2] if arm in REFS else a.runtime/f'scores/{arm}_fold{fold}_OUTER_PRIVATE.pt'
     cell=torch.load(pp,map_location='cpu',weights_only=False);ix=cell['indices'];score[ix]=cell['scores'];decision[ix]=cell['scores']>=cell['threshold']
    q=independent(y,score,.5);tp=int(((y==0)&~decision).sum());fp=int(((y==1)&~decision).sum());tn=int(((y==1)&decision).sum());fn=int(((y==0)&decision).sum())
    ef=2*tp/(2*tp+fp+fn);nf=2*tn/(2*tn+fp+fn);se=tp/(tp+fn);sp=tn/(tn+fp)
    q.update(macro_f1=(ef+nf)/2,ez_f1=ef,nez_f1=nf,TP=tp,FP=fp,TN=tn,FN=fn,sensitivity=se,specificity=sp,balanced_accuracy=(se+sp)/2,accuracy=(tp+tn)/len(y),predicted_ez_fraction=float(np.mean(~decision)))
    q.pop('ez_mrr');q.pop('top1_is_ez')
    for k,v in q.items():assert np.isclose(pooled.loc[arm,k],v,atol=1e-12,rtol=0,equal_nan=True)
  boot=pd.read_csv(pub/f'{phase}_BOOTSTRAP.csv')
  for spec in lock['bootstrap']['comparisons']:
   l,r=spec.split(' minus ');bb=independent_bootstrap(df[df.arm==l],df[df.arm==r],lock['metrics'])
   for m,v in bb.items():row=boot[(boot.comparison==spec)&(boot.metric==m)].iloc[0];assert np.allclose(row[['delta','ci_low','ci_high']].to_numpy(dtype=float),v,atol=1e-12,rtol=0,equal_nan=True)
 c.write_json(pub/'INDEPENDENT_VERIFICATION.json',dict(status='PASS',binding=bind,patient_metric_cells=checked,new_training_cells=10,metrics_independent_sklearn=True,bootstrap_independent_multiplicity=True,draws=10000,seed=42,no_model_rerun=True,aggregate_hashes={p.name:c.sha(p) for p in pub.glob('*.csv')}))
 print('INDEPENDENT_VERIFICATION_PASS',checked,flush=True)

if __name__=='__main__':
 a=arguments();globals()[a.stage](a)
