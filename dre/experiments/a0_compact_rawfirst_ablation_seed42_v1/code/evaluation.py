"""Private freeze -> development replay -> seal all new outer scores -> clinical labels.

R0 and feature stages are separate scientific passes. Historical F0/F6/F7
use only admitted sealed scores. All reports use patient-equal estimands.
"""
import json
import numpy as np
import pandas as pd
import torch
from audit_sources import context,historical,metadata,patient_rows
from dependencies import load
from raw_alignment import RawBankStore
from raw_only_model import constructor as raw_constructor
from verify_feature_masks import factory
from training import path
from bootstrap import compare


def aggregate(a,o,rows,phase,stage):
 lock,ref,c,core,original,trainer,helper,banks,manifest,bind=o;pub=a.runtime/'public';df=pd.DataFrame(rows);metrics=lock['metrics']
 expected=(65,47) if phase=='DEVELOPMENT' else (80,80)
 for arm,q in df.groupby('arm'):
  assert (len(q),q.patient.nunique())==expected
  if phase=='OUTER':assert int(q.channels.sum())==7635
 summary=df.groupby('arm')[metrics].mean().reset_index();summary['parameters']=summary.arm.map(lambda x:18153 if x=='R0' else lock['matrix'][x]['parameters'])
 summary.to_csv(pub/f'{stage}_{phase}_SUMMARY.csv',index=False)
 for field,label in [('fold','FOLD'),('center','CENTER')]:
  records=[dict(arm=arm,**{field:v},patients_or_appearances=len(q),channels=int(q.channels.sum()),**q[metrics].mean().to_dict()) for (arm,v),q in df.groupby(['arm',field])]
  pd.DataFrame(records).to_csv(pub/f'{stage}_{phase}_BY_{label}.csv',index=False)
 if phase=='DEVELOPMENT':df.groupby(['arm','patient'])[metrics].mean().groupby('arm').mean().reset_index().to_csv(pub/f'{stage}_DEVELOPMENT_UNIQUE_ID_SUMMARY.csv',index=False)
 effects=[]
 for arm in sorted(set(df.arm)-{'F0'}):effects.extend(dict(comparison='F0 minus '+arm,**r) for r in compare(c,df[df.arm=='F0'],df[df.arm==arm],metrics))
 pd.DataFrame(effects).to_csv(pub/f'{stage}_{phase}_BOOTSTRAP.csv',index=False)
 core.torch_write(a.runtime/f'{stage}_{phase}_METRICS_PRIVATE.pt',rows)


def run(a,raw):
 o=context(a);lock,ref,c,core,original,trainer,helper,banks,manifest,bind=o;pub=a.runtime/'public';stage='RAW' if raw else 'FEATURE'
 assert not (pub/f'{stage}_OUTER_STATUS.json').exists(),'Completed scientific evaluation cannot rerun'
 assert json.loads((a.runtime/'EXPERIMENT_FREEZE.json').read_text())['binding']==bind
 dfreeze=a.runtime/f'{stage}_DEVELOPMENT_FREEZE.json';fr=json.loads(dfreeze.read_text());assert fr['binding']==bind and fr['status']=='PASS'
 arms=['R0'] if raw else ['F1','F2','F3','F4','F5'];assert len(fr['selections'])==5*len(arms)
 store=RawBankStore(a.runtime,c) if raw else None
 if raw:assert c.sha(a.runtime/'RAW_REGISTRY_PRIVATE.json')==fr['raw_registry_sha256']
 absolute=load('compact_outer_absolute',a.source/'d2_no_patient_z_outer_seed42_v1/code/run_ablation.py')
 hist=load('compact_outer_pr',a.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py')
 meta=metadata(a);hashes={'experiment_freeze':c.sha(a.runtime/'EXPERIMENT_FREEZE.json'),'development_freeze':c.sha(dfreeze)};devcells={};devrows=[];replay=[]
 for r in fr['selections']:
  arm,fold=r['arm'],r['fold'];bp=a.original/f'fold{fold}/BANK_PRIVATE.pt' if raw else a.runtime/f'features/{arm}/fold{fold}/BANK_PRIVATE.pt';cp=path(a,arm,fold)/'BEST_PRIVATE.pt'
  assert c.sha(cp)==r['checkpoint_sha256'] and c.sha(bp)==r['bank_sha256'];hashes[f'{arm}_checkpoint{fold}']=c.sha(cp);hashes[f'{arm}_bank{fold}']=c.sha(bp)
  d=torch.load(bp,map_location='cpu',weights_only=False);ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold']
  vp=a.runtime/f'{stage}_scores/{arm}_fold{fold}_VAL_PRIVATE.pt'
  if vp.exists():
   assert json.loads(vp.with_suffix('.sha256.json').read_text())==dict(binding=c.sha(dfreeze),sha256=c.sha(vp))
   cell=torch.load(vp,map_location='cpu',weights_only=False)
   assert cell['checkpoint_sha256']==c.sha(cp) and cell['bank_sha256']==c.sha(bp) and np.array_equal(cell['indices'],d['val'])
   assert cell['threshold']==ck['threshold'] and cell['binding']==c.sha(dfreeze)
   hashes[f'{arm}_validation{fold}']=c.sha(vp);devcells[arm,fold]=cell
   devrows.extend(patient_rows(c,d,d['val'],cell['scores'],ck['threshold'],arm,fold));replay.append(cell['replay']);continue
  fac=raw_constructor(original,a.source,store,{k:d[k] for k in ['patient','channel']},lock['raw']['encoder_chunk']) if raw else factory(original,helper,arm)
  device='cuda' if raw else 'cpu';m=fac().to(device).eval();m.load_state_dict(ck['model'],strict=True)
  assert sum(p.numel() for p in m.parameters())==r['parameters'];va=d['val']
  x=np.arange(len(d['patient']),dtype=np.float32)[:,None] if raw else d['x']
  with torch.inference_mode():ll=m(torch.as_tensor(x[va],device=device),torch.as_tensor(d['g'][va],device=device));scores=torch.sigmoid(ll).cpu().numpy()
  # Raw replay uses same CUDA backend. Feature CPU replay is compared with
  # exact original CUDA validation checkpoint-selection outputs.
  if not raw:
   m=m.cuda()
   with torch.inference_mode():vl=m(torch.as_tensor(x[va],device='cuda'),torch.as_tensor(d['g'][va],device='cuda'));vs=torch.sigmoid(vl).cpu().numpy()
   ld=float(np.abs(ll.cpu().numpy()-vl.cpu().numpy()).max());sd=float(np.abs(scores-vs).max())
   assert ld<=lock['parity']['logit_cpu_cuda_atol'] and sd<=lock['parity']['probability_cpu_cuda_atol'] and np.array_equal(scores>=ck['threshold'],vs>=ck['threshold'])
  else:vs=scores;ld=sd=0.
  selected=core.select_threshold(d['y'][va],vs,d['patient'][va])
  for k in ['threshold','macro_f1','ez_f1']:assert abs(selected[k]-ck[k])<1e-12
  rep=dict(arm=arm,fold=fold,logit_CPU_CUDA_drift=ld,probability_drift=sd,selection_replay_exact=True)
  cell=dict(indices=va,logits=ll.cpu().numpy(),scores=scores,selection_scores=vs,threshold=ck['threshold'],checkpoint_sha256=c.sha(cp),bank_sha256=c.sha(bp),binding=c.sha(dfreeze),replay=rep)
  core.torch_write(vp,cell);c.write_json(vp.with_suffix('.sha256.json'),dict(binding=c.sha(dfreeze),sha256=c.sha(vp)))
  hashes[f'{arm}_validation{fold}']=c.sha(vp);devcells[arm,fold]=cell;devrows.extend(patient_rows(c,d,va,scores,ck['threshold'],arm,fold));replay.append(rep)
  del m
 refs=['F0'] if raw else ['F0','F6','F7'];olddev=torch.load(a.runtime/'HISTORICAL_DEVELOPMENT_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False)
 devrows.extend(r for r in olddev if r['arm'] in refs)
 aggregate(a,o,devrows,'DEVELOPMENT',stage)
 outerfreeze=a.runtime/f'{stage}_FROZEN_BEFORE_OUTER.json'
 if outerfreeze.exists():
  previous=json.loads(outerfreeze.read_text());assert previous['binding']==bind and previous['hashes']==hashes and previous['selections']==fr['selections']
 else:c.write_json(outerfreeze,dict(status='PASS',binding=bind,hashes=hashes,selections=fr['selections'],replay=replay,new_outer_scores_and_labels_unread=True))
 fs=c.sha(outerfreeze);c.write_json(pub/f'{stage}_FREEZE_AUDIT.json',dict(status='PASS',freeze_sha256=fs,hashes=hashes,selections=fr['selections'],replay=replay))
 cells={};files={};visited={arm:[] for arm in arms}
 for fold in range(1,6):
  outerids=set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']=='test')]);ix=np.flatnonzero(np.isin(meta['patient'],list(outerids)))
  assert len(outerids)==lock['outer_fold_patients'][fold-1] and len(ix)==lock['outer_fold_channels'][fold-1]
  for arm in arms:
   r=next(r for r in fr['selections'] if r['arm']==arm and r['fold']==fold);cp=path(a,arm,fold)/'BEST_PRIVATE.pt';assert c.sha(cp)==r['checkpoint_sha256']
   pp=a.runtime/f'{stage}_scores/{arm}_fold{fold}_OUTER_PRIVATE.pt'
   if pp.exists():
    assert json.loads(pp.with_suffix('.sha256.json').read_text())==dict(binding=fs,sha256=c.sha(pp));cell=torch.load(pp,map_location='cpu',weights_only=False);assert cell['binding']==fs and cell['checkpoint_sha256']==c.sha(cp)
   else:
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    if raw:
     md={k:meta[k][ix] for k in ['patient','channel']};m=raw_constructor(original,a.source,store,md,lock['raw']['encoder_chunk'])().cuda().eval();m.load_state_dict(ck['model'],strict=True)
     with torch.inference_mode():ll=m(torch.arange(len(ix),device='cuda',dtype=torch.float32)[:,None]);ss=torch.sigmoid(ll).cpu().numpy()
    else:
     d=torch.load(a.runtime/f'features/{arm}/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
     # Access only the unlabeled x member after all stage selections freeze.
     with np.load(a.features,allow_pickle=True) as f:xx=f['x'][ix]
     xx=absolute.frozen_absolute(xx,d['pre']) if arm=='F1' else hist.frozen_transform(xx,meta['patient'][ix],d['pre'],core)
     xx=helper.masked(xx,d['mask']);assert np.all(xx[:,~d['mask']]==0)
     m=factory(original,helper,arm)().eval();m.load_state_dict(ck['model'],strict=True)
     with torch.inference_mode():ll=m(torch.from_numpy(xx),torch.zeros(len(xx),dtype=torch.long));ss=torch.sigmoid(ll).numpy()
    cell=dict(binding=fs,indices=ix,logits=ll.cpu().numpy(),scores=ss,threshold=r['threshold'],checkpoint_sha256=c.sha(cp))
    core.torch_write(pp,cell);c.write_json(pp.with_suffix('.sha256.json'),dict(binding=fs,sha256=c.sha(pp)));del m
   assert np.array_equal(cell['indices'],ix) and cell['threshold']==r['threshold'] and np.isfinite(cell['scores']).all() and 'y' not in cell
   cells[arm,fold]=cell;files[f'{arm}_{fold}']=c.sha(pp);visited[arm].extend(ix.tolist())
  print('OUTER_FOLD_SEALED',stage,fold,flush=True)
 for arm in arms:assert sorted(visited[arm])==list(range(7635))
 c.write_json(a.runtime/f'{stage}_OUTER_PREDICTION_SEAL.json',dict(binding=fs,files=files,all_new_stage_predictions_before_full_y=True))
 # This is the FIRST full-y member read by this new evaluation process.
 with np.load(a.features,allow_pickle=True) as f:y=f['y']
 assert len(y)==7635 and int((y==0).sum())==1743 and int((y==1).sum())==5892
 rows=[];pooled=[]
 for arm in arms:
  all_scores=np.empty(7635);all_pred=np.zeros(7635,bool)
  for fold in range(1,6):
   cell=cells[arm,fold];ix=cell['indices'];all_scores[ix]=cell['scores'];all_pred[ix]=cell['scores']>=cell['threshold']
   rows.extend(patient_rows(c,dict(meta,y=y),ix,cell['scores'],cell['threshold'],arm,fold))
  order=np.lexsort((meta['channel'].astype(str),meta['patient'].astype(str)));q=c.metric(y[order],all_scores[order],0.,pred=all_pred[order]);q.pop('ez_mrr');q.pop('top1_is_ez');pooled.append(dict(arm=arm,unit='secondary_pooled_channels',**q))
 oldouter=torch.load(a.runtime/'HISTORICAL_OUTER_METRICS_PRIVATE.pt',map_location='cpu',weights_only=False);rows.extend(r for r in oldouter if r['arm'] in refs)
 aggregate(a,o,rows,'OUTER',stage);pd.DataFrame(pooled).to_csv(pub/f'{stage}_OUTER_POOLED_CHANNELS.csv',index=False)
 assert context(a)[-1]==bind
 c.write_json(pub/f'{stage}_OUTER_STATUS.json',dict(status='ONE_FROZEN_'+stage+'_OUTER_PASS_COMPLETE',new_cells=5 if raw else 25,patients_per_arm=80,channels_per_arm=7635,
  full_experiment_matrix_frozen_before_raw_outcomes=True,all_new_scores_before_global_y=True,scientific_status=lock['scientific_status'],freeze_sha256=fs,seal_sha256=c.sha(a.runtime/f'{stage}_OUTER_PREDICTION_SEAL.json')))
 print('ONE_FROZEN_'+stage+'_OUTER_PASS_COMPLETE',flush=True)
