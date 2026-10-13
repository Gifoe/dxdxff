"""Authoritative historical compact identity and whole-matrix admission lock."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pandas as pd
import torch
from dependencies import dependencies,load
from metrics import independent


def arguments():
 p=argparse.ArgumentParser();p.add_argument('--runtime',type=Path,default=Path(r'D:\a0_compact_rawfirst_ablation_seed42_v1_runtime'))
 p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));opt=p.parse_args();root=Path(__file__).parents[1]
 return SimpleNamespace(**vars(opt),original=Path(r'C:\a0_source_posterior_decoder_seed42_runtime'),previous=Path(r'C:\d2_ictal_ablation_seed42_v1_runtime'),
  compact=Path(r'C:\d2_no_source_no_zdelta_outer_seed42_v1_runtime'),features=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz'),protocol=root/'PROTOCOL_LOCK.json',root=root)


def context(a):
 ref,c,core,original,trainer,helper,banks,origbind=dependencies(a);lock=json.loads(a.protocol.read_text())
 assert list(lock['matrix'])==['F0','R0','F1','F2','F3','F4','F5','F6','F7'] and lock['new_cells']==30
 mp=a.previous/'public/FEATURE_GROUP_MANIFEST.json';assert c.sha(mp)==lock['feature_manifest_sha256'];manifest=json.loads(mp.read_text())
 paths={'protocol':a.protocol,'manifest':mp}
 for folder in ['code','tests']:
  for path in sorted((a.root/folder).glob('*.py')):paths[folder+'_'+path.name]=path
 source_files={'raw_encoder':a.source/'rawtiny_patient_relative_seed42_v1/code/models.py','raw_alignment':a.source/'neuroez_c/dual_view_data.py',
  'raw_normalizer':a.source/'neuroez_c/raw_brainbert_data.py','no_patient_z_transform':a.source/'d2_no_patient_z_outer_seed42_v1/code/run_ablation.py',
  'outer_transform':a.source/'d2_ictal_outer_seed42_v1/code/evaluate_outer.py'}
 for k,path in source_files.items():
  key=k+'_source_sha256' if k in ['raw_encoder','raw_alignment','raw_normalizer'] else k+'_sha256'
  assert c.sha(path)==lock[key];paths[k]=path
 for k,root in [('F0',a.compact),('F6_F7',a.previous)]:
  for name,file in [('freeze','FROZEN_BEFORE_OUTER.json'),('seal','OUTER_PREDICTIONS_SEAL.json')]:
   path=root/file;assert c.sha(path)==lock[f'{k}_{name}_sha256'];paths[k+'_'+name]=path
 bind={'original':origbind,'new':{k:c.sha(v) for k,v in paths.items()}}
 return lock,ref,c,core,original,trainer,helper,banks,manifest,bind


def historical(a,arm,fold):
 if arm=='F0':
  fr=json.loads((a.compact/'FROZEN_BEFORE_OUTER.json').read_text());r=fr['selections'][fold-1];root=a.compact/f'fold{fold}'
  return root/'BANK_PRIVATE.pt',root/'TRAIN/BEST_PRIVATE.pt',root/'OUTER_PREDICTIONS_PRIVATE.pt',dict(r,threshold=r['nez_threshold'])
 name,key=('w/o Source','no_source') if arm=='F6' else ('w/o ZDELTA','no_zdelta')
 fr=json.loads((a.previous/'FROZEN_BEFORE_OUTER.json').read_text());r=next(r for r in fr['selections'] if r['arm']==name and r['fold']==fold);root=a.previous/f'{key}/fold{fold}'
 return root/'BANK_PRIVATE.pt',root/'TRAIN/BEST_PRIVATE.pt',root/'OUTER_PREDICTIONS_PRIVATE.pt',dict(r,threshold=r['nez_threshold'])


def metadata(a):
 with np.load(a.features,allow_pickle=True) as f:return {k:f[k] for k in ['patient','channel','center','features','split_patient','split_role','split_fold']}


def patient_rows(c,d,ix,scores,tau,arm,fold):
 rows=[]
 for pid in sorted(set(d['patient'][ix])):
  k=np.flatnonzero(d['patient'][ix]==pid);k=k[np.argsort(d['channel'][ix][k].astype(str),kind='stable')];ii=ix[k]
  rows.append(dict(arm=arm,patient=pid,fold=fold,center=str(d['center'][ii[0]]),**c.metric(d['y'][ii],scores[k],tau)))
 return rows


def main():
 a=arguments();o=context(a);lock,ref,c,core,original,trainer,helper,banks,manifest,bind=o;pub=a.runtime/'public'
 assert not (a.runtime/'EXPERIMENT_FREEZE.json').exists()
 frozen_old=json.loads((a.compact/'FROZEN_BEFORE_OUTER.json').read_text())
 assert c.sha(a.source/'d2_no_source_no_zdelta_outer_seed42_v1/code/run_combo.py')==frozen_old['binding']['runner']
 assert c.sha(a.source/'d2_no_source_no_zdelta_outer_seed42_v1/PROTOCOL_LOCK.json')==frozen_old['binding']['new_protocol']
 seals={arm:json.loads(((a.compact if arm=='F0' else a.previous)/'OUTER_PREDICTIONS_SEAL.json').read_text())['files'] for arm in ['F0','F6','F7']}
 meta=metadata(a);recs=[];hashes={};devrows=[]
 for arm in ['F0','F6','F7']:
  spec=lock['matrix'][arm];mask=np.asarray([q['view'] in spec['views'] for q in manifest['columns']]);assert mask.sum()==spec['retained']
  for fold,d in enumerate(banks,1):
   bp,cp,pp,r=historical(a,arm,fold);assert c.sha(bp)==r['bank_sha256'] and c.sha(cp)==r['checkpoint_sha256']
   pskey=f'w/o Source + w/o ZDELTA_{fold}' if arm=='F0' else ('no_source_' if arm=='F6' else 'no_zdelta_')+str(fold)
   assert c.sha(pp)==seals[arm][pskey]
   b=torch.load(bp,map_location='cpu',weights_only=False);assert np.array_equal(b['x'],helper.masked(d['x'],mask))
   for k in ['patient','channel','features','y','g','train','val']:assert np.array_equal(b[k],d[k])
   for k in ['mean','scale','var','imputer_statistics']:assert np.array_equal(b['pre'][k],d['pre'][k])
   ck=torch.load(cp,map_location='cpu',weights_only=False);seed=42+1009*fold;core.seed_all(seed)
   m=(original.SourceA0(seed) if arm=='F7' else helper.source_free_factory(original.SourceA0)(seed)).cuda().eval()
   assert sum(p.numel() for p in m.parameters())==spec['parameters'] and isinstance(m.network[0],torch.nn.LayerNorm)
   assert core.state_hash(m.state_dict())==ck['binding']['initial_hash']
   assert ck['binding']['seed']==seed and ck['binding']['max_epochs']==30 and ck['binding']['penalty']==(0.001 if arm=='F7' else 0.)
   m.load_state_dict(ck['model'],strict=True);assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold']
   if arm=='F0':assert ck['epoch']==[3,3,2,3,4][fold-1] and ck['threshold']==[.465,.24,.32,.335,.41][fold-1]
   va=d['val']
   with torch.inference_mode():ll=m(torch.as_tensor(b['x'][va],device='cuda'),torch.as_tensor(b['g'][va],device='cuda'));ss=torch.sigmoid(ll).cpu().numpy()
   sel=core.select_threshold(d['y'][va],ss,d['patient'][va])
   for k in ['macro_f1','ez_f1','threshold']:assert abs(sel[k]-ck[k])<1e-12
   vp=a.runtime/f'historical/{arm}_fold{fold}_VAL_PRIVATE.pt';core.torch_write(vp,dict(indices=va,logits=ll.cpu().numpy(),scores=ss,threshold=ck['threshold'],checkpoint_sha256=c.sha(cp)))
   devrows.extend(patient_rows(c,d,va,ss,ck['threshold'],arm,fold))
   recs.append(dict(arm=arm,fold=fold,epoch=ck['epoch'],threshold=ck['threshold'],parameters=spec['parameters'],retained_columns=spec['retained'],bank_sha256=c.sha(bp),checkpoint_sha256=c.sha(cp),outer_prediction_sha256=c.sha(pp),validation_replay_sha256=c.sha(vp),validation_reselected_matches=True))
   for tag,path in [('bank',bp),('checkpoint',cp),('outer',pp),('validation',vp)]:hashes[f'{arm}_{tag}{fold}']=c.sha(path)
 # Explicit Stage I historical-label provenance verification. No new training.
 with np.load(a.features,allow_pickle=True) as f:y=f['y']
 assert len(y)==7635 and int((y==0).sum())==1743 and int((y==1).sum())==5892
 rows=[]
 for arm in ['F0','F6','F7']:
  visited=[]
  for fold in range(1,6):
   _,_,pp,r=historical(a,arm,fold);cell=torch.load(pp,map_location='cpu',weights_only=False);ix=cell['indices'];visited.extend(ix.tolist())
   test=set(meta['split_patient'][(meta['split_fold']==fold)&(meta['split_role']=='test')]);assert set(meta['patient'][ix])==test and len(test)==lock['outer_fold_patients'][fold-1]
   assert cell['threshold']==r['threshold'];part=patient_rows(c,dict(meta,y=y),ix,cell['scores'],r['threshold'],arm,fold)
   for q in part:
    k=np.flatnonzero(meta['patient'][ix]==q['patient']);k=k[np.argsort(meta['channel'][ix][k].astype(str),kind='stable')];im=independent(y[ix[k]],cell['scores'][k],r['threshold'])
    for name in im:assert np.isclose(im[name],q[name],atol=1e-12,rtol=0,equal_nan=True)
   rows.extend(part)
  assert sorted(visited)==list(range(7635))
 df=pd.DataFrame(rows);means=df.groupby('arm')[c.METRICS].mean()
 for arm,values in {'F0':(.7369771785860808,.5306525381402717,.6242355736555874,.4146706708280264,.6942666410497292,.5875),'F6':(.7397681005685868,.537887185507606,.6249192618886414,.4194045035283591,.7286195483681281,.6375)}.items():
  assert np.allclose(means.loc[arm,['ez_auroc','ez_auprc','macro_f1','ez_f1','ez_mrr','top1_is_ez']],values,atol=1e-12,rtol=0)
 assert abs(means.loc['F7','ez_auroc']-.733994)<5e-7
 core.torch_write(a.runtime/'HISTORICAL_OUTER_METRICS_PRIVATE.pt',rows);core.torch_write(a.runtime/'HISTORICAL_DEVELOPMENT_METRICS_PRIVATE.pt',devrows)
 means.reset_index().to_csv(pub/'HISTORICAL_OUTER_SUMMARY.csv',index=False);pd.DataFrame(devrows).groupby('arm')[c.METRICS].mean().reset_index().to_csv(pub/'HISTORICAL_DEVELOPMENT_SUMMARY.csv',index=False)
 c.write_json(pub/'BASELINE_REPRODUCTION.json',dict(status='PASS',F0_authoritative_source='d2_no_source_no_zdelta_outer_seed42_v1',binding=bind,selections=recs,
  historical_score_only_outer=True,historical_labels_used_only_for_reuse_verification=True,new_outcomes_viewed=False,patients_per_arm=80,parameters_F0=8817))
 c.write_json(a.runtime/'EXPERIMENT_FREEZE.json',dict(status='PASS',binding=bind,historical_hashes=hashes,matrix=lock['matrix'],protocol_sha256=c.sha(a.protocol),new_outcomes_viewed=False))
 c.write_json(pub/'EXPERIMENT_MATRIX.json',dict(status='FROZEN_BEFORE_RAW_NEW_OUTCOMES',matrix=lock['matrix'],new_cells=30,experiment_freeze_sha256=c.sha(a.runtime/'EXPERIMENT_FREEZE.json')))
 print('AUTHORITATIVE_COMPACT_BASELINE_AND_FULL_MATRIX_FREEZE_PASS',flush=True)


if __name__=='__main__':main()
