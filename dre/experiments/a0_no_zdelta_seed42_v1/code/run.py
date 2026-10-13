"""Five source-free A0 cells; only ZDELTA masked; private sealed prediction audit.

No EEG I/O. Development uses original FIT/VAL-only banks. All five selected
models and preprocessors freeze before new outer feature/label access.
"""
import argparse
import copy
import json
import time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from baseline_admission import dependencies,load,patient_rows
from independent import independent


def context():
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['prepare','train','development','freeze_outer','outer','verify'])
 p.add_argument('--runtime',type=Path,default=Path(r'C:\a0_no_zdelta_seed42_v1_runtime'))
 p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'))
 opt=p.parse_args();root=Path(__file__).parents[1]
 a=SimpleNamespace(**vars(opt),original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'),
  previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'),features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),protocol=root/'PROTOCOL_LOCK.json')
 ref,c,core,original,trainer,helper,banks,oldbind=dependencies(a)
 baseline=json.loads((a.runtime/'public/BASELINE_REPRODUCTION.json').read_text());assert baseline['status']=='PASS' and baseline['binding']==oldbind
 lock=json.loads(a.protocol.read_text());assert lock['arms']==['A0','A0_NO_ZDELTA'] and lock['parameters']==8817
 manifest=a.previous/'public/FEATURE_GROUP_MANIFEST.json';assert c.sha(manifest)==lock['feature_manifest_sha256']
 histpath=a.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py';assert c.sha(histpath)==lock['historical_transform_source_sha256']
 hist=load('nozdelta_historical_transform',histpath)
 paths={'protocol':a.protocol,'baseline_replay':a.runtime/'public/BASELINE_REPRODUCTION.json','feature_manifest':manifest,'historical_transform':histpath}
 paths.update({f'code_{q.name}':q for q in sorted((root/'code').glob('*.py'))})
 binding={'original':oldbind,'new':{k:c.sha(v) for k,v in paths.items()}}
 return a,lock,ref,c,core,original,trainer,helper,banks,binding,hist


def checkpoint(a,arm,fold):
 return a.previous/f'no_source/fold{fold}/TRAIN/BEST_PRIVATE.pt' if arm=='A0' else a.runtime/f'fold{fold}/TRAIN/BEST_PRIVATE.pt'


def checks(ctx):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx
 ad=json.loads((a.runtime/'public/PRETRAIN_AUDIT.json').read_text());assert ad['status']=='PASS' and ad['binding']==binding
 for fold,r in enumerate(ad['folds'],1):assert c.sha(a.runtime/f'fold{fold}/BANK_PRIVATE.pt')==r['bank_sha256']
 return ad


def model(ctx,seed=42):return ctx[7].source_free_factory(ctx[5].SourceA0)(seed)


def prepare(ctx):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx
 assert not (a.runtime/'public/PRETRAIN_AUDIT.json').exists()
 columns=helper.parse_names(banks[0]['features']);mask=helper.mask_for(columns,'w/o ZDELTA')
 assert (~mask).any() and all(q['view']=='zdelta' for q,k in zip(columns,mask) if not k)
 rows=[];tests=0
 for fold,d in enumerate(banks,1):
  x,pre=core.prepare(d['raw_x'],d['patient'],d['train']);assert np.array_equal(x,d['x'])
  for name in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pre[name],d['pre'][name])
  z=helper.masked(x,mask);assert np.array_equal(z[:,mask],x[:,mask]) and np.all(z[:,~mask]==0)
  # Label-free held-out perturbation must never affect FIT-fitted statistics.
  perturbed=d['raw_x'].copy();perturbed[d['val']]=perturbed[d['val']]*7+13
  px,pp=core.prepare(perturbed,d['patient'],d['train'])
  for name in ['mean','scale','var','imputer_statistics']:assert np.array_equal(pp[name],pre[name])
  assert np.array_equal(px[d['train']],x[d['train']]);tests+=5
  frozen=hist.frozen_transform(d['raw_x'],d['patient'],d['pre'],core);assert np.array_equal(frozen,x);tests+=1
  seed=42+1009*fold;core.seed_all(seed);m=model(ctx,seed).eval()
  old=torch.load(checkpoint(a,'A0',fold),map_location='cpu',weights_only=False)
  assert core.state_hash(m.state_dict())==old['binding']['initial_hash'];tests+=1
  assert isinstance(m.network[0],torch.nn.LayerNorm) and m.network[0].normalized_shape==(88,) and m.network[0].eps==1e-5
  assert sum(p.numel() for p in m.parameters())==8817 and not any(k.startswith(('V','u','source')) for k in m.state_dict());tests+=2
  with torch.inference_mode():
   tx=torch.from_numpy(z[:8]);assert torch.equal(m(tx,torch.zeros(8,dtype=torch.long)),m(tx,torch.ones(8,dtype=torch.long)));tests+=1
  dd=copy.deepcopy(d);dd.update(x=z,mask=mask,pre=pre)
  path=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';assert not path.exists();core.torch_write(path,dd)
  rows.append(dict(fold=fold,bank_sha256=c.sha(path),original_bank_sha256=ref['bank_sha256'][fold-1],
   original_x_sha256=c.digest(x.tolist()),preprocessor_sha256=c.digest({k:np.asarray(v).tolist() for k,v in pre.items()}),
   FIT_patients=len(set(d['patient'][d['train']])),validation_patients=len(set(d['patient'][d['val']])),masked_columns=int((~mask).sum()),
   retained_columns=int(mask.sum()),initial_weights_exact=True,LayerNorm_unchanged=True,source_free=True,FIT_only_preprocessing_bitwise=True))
 # Exact deterministic CPU checkpoint/optimizer/RNG resume on synthetic data.
 rng=np.random.default_rng(123);sx=rng.normal(size=(24,88)).astype(np.float32);sy=np.tile([0,1],12);sp=np.repeat(np.arange(8),3).astype(str);sg=np.tile(np.arange(4),6)
 tr=np.arange(18);va=np.arange(18,24);old_factory=trainer.SourceA0;trainer.SourceA0=helper.source_free_factory(original.SourceA0)
 try:
  trainer.train(sx,sy,sp,sg,tr,va,a.runtime/'engineering/continuous',42,'synthetic',device='cpu',max_epochs=3,penalty=0.)
  trainer.train(sx,sy,sp,sg,tr,va,a.runtime/'engineering/resumed',42,'synthetic',device='cpu',max_epochs=3,stop_after=1,penalty=0.)
  trainer.train(sx,sy,sp,sg,tr,va,a.runtime/'engineering/resumed',42,'synthetic',device='cpu',max_epochs=3,penalty=0.)
 finally:trainer.SourceA0=old_factory
 q=[torch.load(a.runtime/f'engineering/{s}/LAST_PRIVATE.pt',map_location='cpu',weights_only=False) for s in ['continuous','resumed']]
 assert core.state_hash(q[0]['model'])==core.state_hash(q[1]['model']) and q[0]['history']==q[1]['history'];tests+=2
 for k in q[0]['optimizer']['state']:
  for name,v in q[0]['optimizer']['state'][k].items():assert torch.equal(v,q[1]['optimizer']['state'][k][name]);tests+=1
 pd.DataFrame([dict(**q,retained=bool(k)) for q,k in zip(columns,mask)]).to_csv(a.runtime/'public/FEATURE_MASK_AUDIT.csv',index=False)
 c.write_json(a.runtime/'public/PRETRAIN_AUDIT.json',dict(status='PASS',binding=binding,folds=rows,engineering_assertions=tests,
  synthetic_resume_exact=True,Patient_Z_retained=True,raw_EEG_access=False,outer_access=False,parameters=8817,normalization='original LayerNorm88',torch=torch.__version__))
 print('A0_NO_ZDELTA_PRETRAIN_PASS',tests,flush=True)


def train(ctx):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx;checks(ctx)
 assert not (a.runtime/'DEVELOPMENT_FREEZE.json').exists()
 rows=[];histories=[];durations=[]
 for fold in range(1,6):
  bp=a.runtime/f'fold{fold}/BANK_PRIVATE.pt';d=torch.load(bp,map_location='cpu',weights_only=False)
  seed=42+1009*fold;old_factory=trainer.SourceA0;trainer.SourceA0=helper.source_free_factory(original.SourceA0);t=time.perf_counter()
  try:trainer.train(d['x'],d['y'],d['patient'],d['g'],d['train'],d['val'],bp.parent/'TRAIN',seed,c.digest(dict(binding=binding,bank=c.sha(bp))),device='cuda',penalty=0.)
  finally:trainer.SourceA0=old_factory
  for arm in lock['arms']:
   cp=checkpoint(a,arm,fold);ck=torch.load(cp,map_location='cpu',weights_only=False)
   rows.append(dict(arm=arm,fold=fold,epoch=ck['epoch'],threshold=ck['threshold'],checkpoint_sha256=c.sha(cp),bank_sha256=c.sha(bp),baseline_reused=arm=='A0'))
  cp=checkpoint(a,'A0_NO_ZDELTA',fold);done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==c.sha(cp)
  last=torch.load(cp.parent/'LAST_PRIVATE.pt',map_location='cpu',weights_only=False)
  histories.extend(dict(fold=fold,**h) for h in last['history']);durations.append(dict(fold=fold,seconds=time.perf_counter()-t,epochs=last['epoch']))
  c.write_json(a.runtime/'public/TRAIN_PROGRESS.json',dict(status='RUNNING',folds_complete=fold,total=5));print('A0_NO_ZDELTA_SELECTED',fold,ck['epoch'],flush=True)
 assert context()[9]==binding
 c.write_json(a.runtime/'DEVELOPMENT_FREEZE.json',dict(status='PASS',binding=binding,selections=rows,outer_features_or_labels_accessed=False))
 pd.DataFrame(rows).to_csv(a.runtime/'public/MODEL_SELECTION.csv',index=False)
 pd.DataFrame(histories).to_csv(a.runtime/'public/TRAINING_HISTORY.csv',index=False)
 pd.DataFrame(durations).to_csv(a.runtime/'public/TRAINING_EFFICIENCY.csv',index=False)
 c.write_json(a.runtime/'public/TRAIN_PROGRESS.json',dict(status='FIVE_NEW_FOLDS_COMPLETE',folds_complete=5,total=5))


def selections(ctx):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx;checks(ctx)
 fr=json.loads((a.runtime/'DEVELOPMENT_FREEZE.json').read_text());assert fr['status']=='PASS' and fr['binding']==binding and len(fr['selections'])==10
 for r in fr['selections']:
  cp=checkpoint(a,r['arm'],r['fold']);bp=a.runtime/f"fold{r['fold']}/BANK_PRIVATE.pt"
  assert c.sha(cp)==r['checkpoint_sha256'] and c.sha(bp)==r['bank_sha256']
  ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold']
 return fr


def summary(ctx,rows,phase):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx;pub=a.runtime/'public';df=pd.DataFrame(rows)
 metrics=c.METRICS+['predicted_ez_fraction','TP','FP','TN','FN'];expected=(65,47) if phase=='DEVELOPMENT' else (80,80)
 assert all((len(q),q.patient.nunique())==expected for _,q in df.groupby('arm'))
 out=df.groupby('arm')[metrics].mean().reset_index();out['parameters']=8817;out.to_csv(pub/f'{phase}_SUMMARY.csv',index=False)
 for key,label in [('fold','FOLD'),('center','CENTER')]:
  q=[dict(arm=arm,**{key:v},patients_or_appearances=len(part),channels=int(part.channels.sum()),**part[metrics].mean().to_dict()) for (arm,v),part in df.groupby(['arm',key])]
  pd.DataFrame(q).to_csv(pub/f'{phase}_BY_{label}.csv',index=False)
 if phase=='DEVELOPMENT':df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().reset_index().to_csv(pub/'DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
 left=df[df.arm=='A0_NO_ZDELTA'].set_index(['fold','patient']).sort_index();right=df[df.arm=='A0'].set_index(['fold','patient']).sort_index();assert left.index.equals(right.index)
 delta=left[metrics].to_numpy()-right[metrics].to_numpy();boot=c.cluster_bootstrap(left.index.get_level_values('patient'),delta,10000)
 pd.DataFrame([dict(metric=m,delta=float(np.nanmean(delta[:,j])),ci_low=float(np.nanpercentile(boot[:,j],2.5)),ci_high=float(np.nanpercentile(boot[:,j],97.5)),
  draws=10000,seed=42,unique_clusters=expected[1],retained_appearances=expected[0],thresholds_reoptimized=False,comparison='A0_NO_ZDELTA minus A0') for j,m in enumerate(metrics)]).to_csv(pub/f'{phase}_PAIRED_BOOTSTRAP.csv',index=False)
 core.torch_write(a.runtime/f'{phase}_METRICS_PRIVATE.pt',rows)
 target={'ez_auroc':.710667,'ez_auprc':.518235,'macro_f1':.638080} if phase=='DEVELOPMENT' else {'ez_auroc':.739768,'ez_auprc':.537887,'macro_f1':.624919}
 for k,v in target.items():assert abs(out.set_index('arm').loc['A0',k]-v)<5e-7


def development(ctx):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx;fr=selections(ctx)
 assert not (a.runtime/'public/DEVELOPMENT_STATUS.json').exists()
 fs=c.sha(a.runtime/'DEVELOPMENT_FREEZE.json');rows=[];files={}
 for fold in range(1,6):
  d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
  for arm in lock['arms']:
   cp=checkpoint(a,arm,fold);ck=torch.load(cp,map_location='cpu',weights_only=False)
   path=a.runtime/f'fold{fold}/{arm}_VAL_PRIVATE.pt'
   if arm=='A0':
    cell=torch.load(path.parent/'B0_VAL_PRIVATE.pt',map_location='cpu',weights_only=False)
   else:
    m=model(ctx).cuda().eval();m.load_state_dict(ck['model'],strict=True)
    with torch.inference_mode():ll=m(torch.as_tensor(d['x'][d['val']],device='cuda'),torch.as_tensor(d['g'][d['val']],device='cuda'));ss=torch.sigmoid(ll).cpu().numpy()
    sel=core.select_threshold(d['y'][d['val']],ss,d['patient'][d['val']])
    for k in ['macro_f1','ez_f1','threshold']:assert abs(sel[k]-ck[k])<1e-12
    cell=dict(indices=d['val'],logits=ll.cpu().numpy(),scores=ss,threshold=ck['threshold'],checkpoint_sha256=c.sha(cp))
   cell['binding']=fs;core.torch_write(path,cell);files[f'{arm}_{fold}']=c.sha(path)
   rows.extend(patient_rows(c,d,d['val'],cell['scores'],ck['threshold'],arm,fold))
 c.write_json(a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json',dict(binding=fs,files=files))
 summary(ctx,rows,'DEVELOPMENT')
 c.write_json(a.runtime/'public/DEVELOPMENT_STATUS.json',dict(status='ONE_FROZEN_DEVELOPMENT_PASS_COMPLETE',appearances=65,unique_patients=47,no_outer_access=True))
 print('DEVELOPMENT_COMPLETE',flush=True)


def freeze_outer(ctx):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx;fr=selections(ctx)
 assert json.loads((a.runtime/'public/DEVELOPMENT_STATUS.json').read_text())['status']=='ONE_FROZEN_DEVELOPMENT_PASS_COMPLETE'
 assert not (a.runtime/'FROZEN_BEFORE_OUTER.json').exists()
 oldseal=a.previous/'OUTER_PREDICTIONS_SEAL.json';assert c.sha(oldseal)==lock['baseline_outer_seal_sha256'];os=json.loads(oldseal.read_text())
 ds=json.loads((a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json').read_text());rows=[];paths={'development_freeze':a.runtime/'DEVELOPMENT_FREEZE.json','development_seal':a.runtime/'DEVELOPMENT_PREDICTION_SEAL.json','baseline_outer_seal':oldseal}
 for fold in range(1,6):
  d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
  for arm in lock['arms']:
   cp=checkpoint(a,arm,fold);ck=torch.load(cp,map_location='cpu',weights_only=False);vp=a.runtime/f'fold{fold}/{arm}_VAL_PRIVATE.pt';assert c.sha(vp)==ds['files'][f'{arm}_{fold}'];val=torch.load(vp,map_location='cpu',weights_only=False)
   m=model(ctx).eval();m.load_state_dict(ck['model'],strict=True);x=banks[fold-1]['x'] if arm=='A0' else d['x']
   with torch.inference_mode():ll=m(torch.from_numpy(x[d['val']]),torch.from_numpy(d['g'][d['val']]));ss=torch.sigmoid(ll).numpy()
   ld=float(np.abs(ll.numpy()-val['logits']).max());sd=float(np.abs(ss-val['scores']).max())
   assert ld<=lock['logit_replay_absolute_tolerance'] and sd<=lock['probability_replay_absolute_tolerance'] and np.array_equal(ss>=ck['threshold'],val['scores']>=ck['threshold'])
   rows.append(dict(arm=arm,fold=fold,logit_cpu_cuda_drift=ld,probability_cpu_cuda_drift=sd,decisions_exact=True))
   paths[f'{arm}_checkpoint{fold}']=cp;paths[f'{arm}_validation{fold}']=vp
  paths[f'bank{fold}']=a.runtime/f'fold{fold}/BANK_PRIVATE.pt'
  pp=a.previous/f'no_source/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt';assert c.sha(pp)==os['files'][f'no_source_{fold}'];paths[f'A0_outer{fold}']=pp
 freeze=dict(status='PASS',development_binding=binding,hashes={k:c.sha(v) for k,v in paths.items()},selections=fr['selections'],replay=rows,outer_features_and_global_labels_unread=True)
 c.write_json(a.runtime/'FROZEN_BEFORE_OUTER.json',freeze);c.write_json(a.runtime/'public/FREEZE_AUDIT.json',freeze)
 print('ALL_FIVE_MODELS_PREPROCESSORS_THRESHOLDS_FROZEN_BEFORE_OUTER',flush=True)


def outer(ctx):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx;selections(ctx)
 assert not (a.runtime/'public/OUTER_STATUS.json').exists()
 fr=json.loads((a.runtime/'FROZEN_BEFORE_OUTER.json').read_text());assert fr['development_binding']==binding
 for k,h in fr['hashes'].items():
  if k.startswith('A0_outer'):p=a.previous/f"no_source/fold{k[-1]}/OUTER_PREDICTIONS_PRIVATE.pt"
  elif k=='baseline_outer_seal':p=a.previous/'OUTER_PREDICTIONS_SEAL.json'
  elif k.startswith('bank'):p=a.runtime/f'fold{k[-1]}/BANK_PRIVATE.pt'
  elif '_checkpoint' in k:arm,num=k.split('_checkpoint');p=checkpoint(a,arm,int(num))
  elif '_validation' in k:arm,num=k.split('_validation');p=a.runtime/f'fold{num}/{arm}_VAL_PRIVATE.pt'
  else:p=a.runtime/('DEVELOPMENT_FREEZE.json' if k=='development_freeze' else 'DEVELOPMENT_PREDICTION_SEAL.json')
  assert c.sha(p)==h,k
 fs=c.sha(a.runtime/'FROZEN_BEFORE_OUTER.json');raw=hist.read_raw(a.features);cells={};files={};visited=[]
 for fold in range(1,6):
  d=torch.load(a.runtime/f'fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False);ids=hist.role_ids(raw,fold)
  di=np.flatnonzero(np.isin(raw['patient'],list(ids['fit']|ids['validation'])));ix=np.flatnonzero(np.isin(raw['patient'],list(ids['test'])))
  for k in ['patient','channel','center']:assert np.array_equal(raw[k][di],d[k])
  assert np.array_equal(raw['x'][di],d['raw_x'],equal_nan=True)
  assert len(ids['test'])==lock['outer']['test_patients_by_fold'][fold-1] and len(ix)==lock['outer']['test_channels_by_fold'][fold-1]
  assert set(d['pre']['fit_patient_ids'])==ids['fit'] and not set(d['patient'])&ids['test'];visited.extend(ix.tolist())
  xx=helper.masked(hist.frozen_transform(raw['x'][ix],raw['patient'][ix],d['pre'],core),d['mask'])
  for arm in lock['arms']:
   cp=checkpoint(a,arm,fold);rec=next(r for r in fr['selections'] if r['arm']==arm and r['fold']==fold)
   path=a.previous/f'no_source/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt' if arm=='A0' else a.runtime/f'{arm}_fold{fold}_OUTER_PRIVATE.pt'
   if arm=='A0':cell=torch.load(path,map_location='cpu',weights_only=False)
   elif path.exists():
    assert json.loads(path.with_suffix('.sha256.json').read_text())==dict(binding=fs,sha256=c.sha(path));cell=torch.load(path,map_location='cpu',weights_only=False)
    assert cell['binding']==fs and cell['checkpoint_sha256']==c.sha(cp)
   else:
    ck=torch.load(cp,map_location='cpu',weights_only=False);m=model(ctx).eval();m.load_state_dict(ck['model'],strict=True)
    with torch.inference_mode():ll=m(torch.from_numpy(xx),torch.zeros(len(xx),dtype=torch.long));ss=torch.sigmoid(ll).numpy()
    cell=dict(binding=fs,indices=ix,logits=ll.numpy(),scores=ss,threshold=rec['threshold'],checkpoint_sha256=c.sha(cp))
    core.torch_write(path,cell);c.write_json(path.with_suffix('.sha256.json'),dict(binding=fs,sha256=c.sha(path)))
   assert np.array_equal(cell['indices'],ix) and cell['threshold']==rec['threshold'] and np.isfinite(cell['scores']).all() and 'y' not in cell
   files[f'{arm}_{fold}']=c.sha(path);cells[arm,fold]=cell
  print('OUTER_SCORES_SEALED',fold,flush=True)
 assert sorted(visited)==list(range(7635))
 c.write_json(a.runtime/'OUTER_PREDICTION_SEAL.json',dict(binding=fs,files=files,all_new_predictions_before_full_y=True))
 with np.load(a.features,allow_pickle=True) as f:y=f['y']
 assert len(y)==7635 and int((y==0).sum())==1743 and int((y==1).sum())==5892
 rows=[];pooled=[]
 for arm in lock['arms']:
  scores=np.empty(7635);pred=np.zeros(7635,bool)
  for fold in range(1,6):
   cell=cells[arm,fold];ix=cell['indices'];scores[ix]=cell['scores'];pred[ix]=cell['scores']>=cell['threshold']
   d=banks[fold-1];di=np.flatnonzero(np.isin(raw['patient'],d['patient']));assert np.array_equal(y[di],d['y'])
   dd=dict(raw,y=y);rows.extend(patient_rows(c,dd,ix,cell['scores'],cell['threshold'],arm,fold))
  order=np.lexsort((raw['channel'].astype(str),raw['patient'].astype(str)));q=c.metric(y[order],scores[order],0.,pred=pred[order]);q.pop('ez_mrr');q.pop('top1_is_ez');pooled.append(dict(arm=arm,unit='secondary_pooled_channels',**q))
 summary(ctx,rows,'OUTER');pd.DataFrame(pooled).to_csv(a.runtime/'public/OUTER_POOLED_CHANNEL_METRICS.csv',index=False)
 assert context()[9]==binding
 c.write_json(a.runtime/'public/OUTER_STATUS.json',dict(status='ONE_FROZEN_A0_NO_ZDELTA_OUTER_PASS_COMPLETE',patients_per_arm=80,channels_per_arm=7635,EZ=1743,NEZ=5892,new_inference_cells=5,reused_A0_cells=5,
  all_new_predictions_sealed_before_full_y=True,freeze_sha256=fs,seal_sha256=c.sha(a.runtime/'OUTER_PREDICTION_SEAL.json'),scientific_status=lock['scientific_status']))
 print('ONE_FROZEN_A0_NO_ZDELTA_OUTER_PASS_COMPLETE',flush=True)


def verify(ctx):
 a,lock,ref,c,core,original,trainer,helper,banks,binding,hist=ctx;selections(ctx);metrics=c.METRICS+['predicted_ez_fraction','TP','FP','TN','FN'];total=0;drift=0.
 for phase,clusters,appearances in [('DEVELOPMENT',47,65),('OUTER',80,80)]:
  assert (a.runtime/f'public/{phase}_STATUS.json').exists();seal=json.loads((a.runtime/f'{phase}_PREDICTION_SEAL.json').read_text());rows=[]
  if phase=='OUTER':
   with np.load(a.features,allow_pickle=True) as f:raw={k:f[k] for k in ['y','patient','channel','center']}
  for fold in range(1,6):
   d=banks[fold-1] if phase=='DEVELOPMENT' else raw
   for arm in lock['arms']:
    path=a.runtime/f'fold{fold}/{arm}_VAL_PRIVATE.pt' if phase=='DEVELOPMENT' else a.previous/f'no_source/fold{fold}/OUTER_PREDICTIONS_PRIVATE.pt' if arm=='A0' else a.runtime/f'{arm}_fold{fold}_OUTER_PRIVATE.pt'
    assert c.sha(path)==seal['files'][f'{arm}_{fold}'];cell=torch.load(path,map_location='cpu',weights_only=False);ix=cell['indices']
    r=next(r for r in selections(ctx)['selections'] if r['arm']==arm and r['fold']==fold);assert cell['threshold']==r['threshold']
    for pat in sorted(set(d['patient'][ix])):
     k=np.flatnonzero(d['patient'][ix]==pat);k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')];ii=ix[k]
     im=independent(d['y'][ii],cell['scores'][k],cell['threshold']);cm=c.metric(d['y'][ii],cell['scores'][k],cell['threshold'])
     for name in im:
      assert np.isclose(im[name],cm[name],atol=1e-12,rtol=0,equal_nan=True),(phase,fold,arm,name)
      if np.isfinite(im[name]):drift=max(drift,abs(im[name]-cm[name]))
     rows.append(dict(arm=arm,fold=fold,patient=pat,center=str(d['center'][ii[0]]),**im));total+=1
  df=pd.DataFrame(rows);assert len(df)==2*appearances and df.patient.nunique()==clusters
  for filename,keys in [(f'{phase}_SUMMARY.csv',['arm']),(f'{phase}_BY_FOLD.csv',['arm','fold']),(f'{phase}_BY_CENTER.csv',['arm','center'])]:
   actual=pd.read_csv(a.runtime/'public'/filename).set_index(keys).sort_index();expected=df.groupby(keys)[metrics].mean().sort_index()
   assert actual.index.equals(expected.index) and np.allclose(actual[metrics],expected,atol=1e-12,rtol=0,equal_nan=True)
  if phase=='DEVELOPMENT':
   aa=pd.read_csv(a.runtime/'public/DEVELOPMENT_UNIQUE_ID_SUMMARY.csv').set_index('arm').sort_index();bb=df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().sort_index();assert np.allclose(aa[metrics],bb,atol=1e-12,rtol=0,equal_nan=True)
  ll=df[df.arm=='A0_NO_ZDELTA'].set_index(['fold','patient']).sort_index();rr=df[df.arm=='A0'].set_index(['fold','patient']).sort_index();assert ll.index.equals(rr.index)
  ids=ll.index.get_level_values('patient').to_numpy().astype(str);unique=np.unique(ids);delta=ll[metrics].to_numpy()-rr[metrics].to_numpy()
  draws=np.random.default_rng(42).integers(0,len(unique),size=(10000,len(unique)));counts=np.stack([(draws==i).sum(1) for i in range(len(unique))],axis=1)
  sums=np.stack([np.nansum(delta[ids==p],axis=0) for p in unique]);n=np.stack([np.isfinite(delta[ids==p]).sum(0) for p in unique]);num=counts@sums;den=counts@n
  boot=np.divide(num,den,out=np.full_like(num,np.nan),where=den>0);actual=pd.read_csv(a.runtime/f'public/{phase}_PAIRED_BOOTSTRAP.csv').set_index('metric')
  for j,name in enumerate(metrics):assert np.allclose([np.nanmean(delta[:,j]),*np.nanpercentile(boot[:,j],[2.5,97.5])],actual.loc[name,['delta','ci_low','ci_high']].astype(float),atol=1e-12,rtol=0,equal_nan=True)
 assert context()[9]==binding
 c.write_json(a.runtime/'public/INDEPENDENT_VERIFICATION.json',dict(status='PASS',metric_cells=total,max_metric_drift=drift,verified_contrast_intervals=2*len(metrics),all_aggregates_verified=True,
  no_new_model_inference=True,all_checkpoints_banks_sources_protocol_thresholds_unchanged=True,development_unique_clusters=47,outer_unique_clusters=80,bootstrap_draws=10000))
 print('A0_NO_ZDELTA_INDEPENDENT_VERIFICATION_PASS',flush=True)


if __name__=='__main__':
 ctx=context();globals()[ctx[0].phase](ctx)
