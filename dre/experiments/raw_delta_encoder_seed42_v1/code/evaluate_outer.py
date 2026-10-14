"""User-authorized frozen Outer addition. No training/selection/cache re-extraction."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from io_utils import sha,write,save,read,load
from dependencies import context
from compact_delta_classifier import CompactDelta
from raw_store import Store
from train_dynamics import make,predict,error_rows
from evaluate import METRICS,patient_rows
from bootstrap import compare

ROOT=Path(__file__).parents[1]
FILES=['evaluate_outer.py','dependencies.py','io_utils.py','compact_delta_classifier.py','raw_delta_encoder.py','raw_store.py','temporal_shuffle.py','train_dynamics.py','evaluate.py','bootstrap.py']

def frozen_transform(x,patient,pre,core):
 z=core.patient_z(x,patient);stats=np.asarray(pre['imputer_statistics']);assert np.isfinite(stats).all() and (np.asarray(pre['scale'])>0).all()
 out=((np.where(np.isnan(z),stats,z)-pre['mean'])/pre['scale']).astype(np.float32)
 assert out.shape==x.shape and np.isfinite(out).all();return out

def role_ids(meta,fold):
 ix=meta['split_fold']==fold;roles={r:set(meta['split_patient'][ix&(meta['split_role']==r)]) for r in ['fit','validation','test']}
 assert all(roles[a].isdisjoint(roles[b]) for a,b in [('fit','validation'),('fit','test'),('validation','test')]);return roles

def outer_context(a):
 runtime=a.runtime;out=runtime/'outer_private_v1';pub=out/'public';pub.mkdir(parents=True,exist_ok=True)
 lock=json.loads((ROOT/'OUTER_PROTOCOL_AMENDMENT.json').read_text());assert sha(ROOT/'PROTOCOL_LOCK.json')==lock['original_protocol_sha256']
 c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,runtime)
 for gate,test in [('STAGE_A_GATE.json','test_stage_a.py'),('STAGE_B_GATE.json','test_stage_b.py'),('STAGE_C_GATE.json','test_stage_c.py')]:
  binding=json.loads((runtime/'results'/gate).read_text())['binding']
  for file,digest in binding['code'].items():assert sha(ROOT/'code'/file)==digest
  assert sha(ROOT/'tests'/test)==binding['tests' if gate!='STAGE_B_GATE.json' else 'test']
 assert json.loads((runtime/'results/INDEPENDENT_VERIFICATION.json').read_text())['status']=='PASS'
 bfr=json.loads((runtime/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text());cfr=json.loads((runtime/'STAGE_C_MODEL_FREEZE_PRIVATE.json').read_text());store=Store(runtime)
 meta=rec['meta'];assert len(meta['patient'])==7635 and len(set(meta['patient']))==80 and len(set(zip(meta['patient'],meta['channel'])))==7635
 sels=[];all_test=[]
 for fold,d in enumerate(banks,1):
  roles=role_ids(meta,fold);ii=np.flatnonzero(np.isin(meta['patient'],list(roles['test'])));assert len(roles['test'])==lock['cohort']['patients_by_fold'][fold-1] and len(ii)==lock['cohort']['channels_by_fold'][fold-1];all_test.extend(ii.tolist())
  assert set(d['patient'][d['train']])==roles['fit'] and set(d['patient'][d['val']])==roles['validation'] and set(d['patient']).isdisjoint(roles['test'])
  hd,cp,r=hist[fold-1];ck=torch.load(cp,map_location='cpu',weights_only=False);assert ck['epoch']==r['epoch'] and ck['threshold']==r['threshold'];assert set(hd['pre']['fit_patient_ids'])==roles['fit']
  sels.append(dict(arm='D0',fold=fold,checkpoint=str(cp),checkpoint_sha256=sha(cp),bank=None,epoch=ck['epoch'],threshold=ck['threshold'],preprocessor_sha256=c.digest({k:hd['pre'][k].tolist() for k in ['imputer_statistics','mean','scale','var']})))
  for arm in ['E1','D1','D2','D3']:
   ref=next(q for q in (bfr if arm in ['E1','D1'] else cfr)['selections'] if q['arm']==arm and q['fold']==fold);cp=runtime/f'{arm}/fold{fold}/TRAIN/BEST_PRIVATE.pt';bp=runtime/f"{arm if arm in ['E1','D1'] else 'D1'}/fold{fold}/BANK_PRIVATE.pt";assert sha(cp)==ref['checkpoint_sha256'];ck=torch.load(cp,map_location='cpu',weights_only=False);bd=torch.load(bp,map_location='cpu',weights_only=False);assert ck['epoch']==ref['epoch'] and ck['threshold']==ref['threshold'];assert set(bd['pre']['fit_patient_ids'])==roles['fit']
   bankref=next(q for q in bfr['selections'] if q['arm']==(arm if arm in ['E1','D1'] else 'D1') and q['fold']==fold);assert sha(bp)==bankref['bank_sha256']
   if arm in ['D2','D3']:assert ref['D1_checkpoint_sha256']==sha(runtime/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt')
   done=json.loads((cp.parent/'COMPLETE_PRIVATE.json').read_text());assert done['best_sha256']==sha(cp) and done['last_sha256']==sha(cp.parent/'LAST_PRIVATE.pt')
   # Replay frozen preprocessing on legal FIT/VAL only; no fitting or TEST labels.
   key='E14' if arm=='E1' else 'R14';xx=frozen_transform(rec[key][d['archive_indices']],d['patient'],bd['pre'],core);assert np.array_equal(xx,bd['x'])
   sels.append(dict(arm=arm,fold=fold,checkpoint=str(cp),checkpoint_sha256=sha(cp),bank=str(bp),bank_sha256=sha(bp),epoch=ck['epoch'],threshold=ck['threshold'],preprocessor_sha256=c.digest({k:bd['pre'][k].tolist() for k in ['imputer_statistics','mean','scale','var']})))
 assert sorted(all_test)==list(range(7635))
 binding=dict(amendment=sha(ROOT/'OUTER_PROTOCOL_AMENDMENT.json'),original_protocol=sha(ROOT/'PROTOCOL_LOCK.json'),reconstruction=sha(runtime/'RECONSTRUCTION_PRIVATE.pkl'),Stage_B=sha(runtime/'STAGE_B_MODEL_FREEZE_PRIVATE.json'),Stage_C=sha(runtime/'STAGE_C_MODEL_FREEZE_PRIVATE.json'),Raw_registry=sha(runtime/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),dependencies=deps,source={f:sha(ROOT/'code'/f) for f in FILES},test=sha(ROOT/'tests/test_outer.py'),device='cuda',torch_version=torch.__version__,numpy_version=np.__version__)
 return lock,c,core,fac,records,rec,banks,hist,store,sels,binding,out,pub

def freeze(a):
 lock,c,core,fac,records,rec,banks,hist,store,sels,binding,out,pub=outer_context(a);target=out/'FROZEN_BEFORE_OUTER_PRIVATE.json';obj=dict(status='PASS',binding=binding,selections=sels,all25_models_and_scalers_locked=True,full_test_y_unread=True,patients=80,channels=7635)
 if target.exists():assert json.loads(target.read_text())==obj
 else:write(target,obj)
 public=[{k:v for k,v in r.items() if k not in ['checkpoint','bank']} for r in sels];write(pub/'OUTER_FREEZE_AUDIT.json',dict(status='PASS',binding=binding,selections=public,freeze_sha256=sha(target),all25_models_and_scalers_locked=True,full_test_y_unread=True,patients=80,channels=7635,no_retraining=True,no_threshold_or_scaler_refit=True))
 print('RAW_DELTA_ALL25_MODELS_FROZEN_OUTER_LABELS_UNREAD',flush=True)

def checked_cached(path,binding,checkpoint,indices):
 q=read(path);assert q['binding']==binding and q['checkpoint_sha256']==checkpoint and np.array_equal(q['indices'],indices) and 'y' not in q;return q

def dynamic_cell(model,bank,global_indices,store,mode,device,path,binding,checkpoint,arm,fold):
 if path.exists():return checked_cached(path,binding,checkpoint,global_indices)
 N=len(global_indices);result={k:np.empty(N,np.float32) for k in ['logit','raw_logit','base_logit','residual']};extras={k:np.empty(N,np.float32) for k in ['gamma0','shuffle','absolute']} if arm=='D2' else {};count=0
 for pid in sorted(set(bank['patient'])):
  ii=np.flatnonzero(bank['patient']==pid);gix=global_indices[ii];pp=path.parent/f'{arm}_fold{fold}_patients_private/{c_digest(str(pid))}.pkl'
  if pp.exists():q=checked_cached(pp,binding,checkpoint,gix)
  else:
   q=predict(model,bank,ii,store,mode,device,interventions=arm=='D2');q.update(indices=gix,binding=binding,checkpoint_sha256=checkpoint);save(pp,q)
  for key in result:result[key][ii]=q[key]
  for key in extras:extras[key][ii]=q['interventions'][key]
  count+=1
 result.update(interventions=extras,indices=global_indices,binding=binding,checkpoint_sha256=checkpoint);save(path,result);print('OUTER_DYNAMIC_PATIENTS_SEALED',arm,fold,count,flush=True);return result

def c_digest(value):
 import hashlib
 return hashlib.sha256(value.encode()).hexdigest()[:24]

def infer(a):
 lock,c,core,fac,records,rec,banks,hist,store,sels,binding,out,pub=outer_context(a);fp=out/'FROZEN_BEFORE_OUTER_PRIVATE.json';fr=json.loads(fp.read_text());assert fr['binding']==binding and fr['selections']==sels
 if (pub/'OUTER_STATUS.json').exists():assert json.loads((pub/'OUTER_STATUS.json').read_text())['freeze_sha256']==sha(fp);print('OUTER_ALREADY_COMPLETE_NO_NEW_INFERENCE',flush=True);return
 archive=Path(r'C:\a0_patient_local_label_reliability_seed42_runtime\gate\FEATURES_PRIVATE.npz');assert sha(archive)==rec['binding']['archive']
 with np.load(archive,allow_pickle=True) as z:full88=z['x']  # no y access until every prediction is sealed
 meta=rec['meta'];files={};cells={};parity=[]
 for r in sels:
  arm,fold=r['arm'],r['fold'];cp=Path(r['checkpoint']);ix=np.flatnonzero(np.isin(meta['patient'],list(role_ids(meta,fold)['test'])));pp=out/f'scores_private/{arm}_fold{fold}.pkl';ck=torch.load(cp,map_location='cpu',weights_only=False)
  if pp.exists():q=checked_cached(pp,sha(fp),r['checkpoint_sha256'],ix)
  else:
   if arm=='D0':
    hd=hist[fold-1][0];xx=frozen_transform(full88[ix],meta['patient'][ix],hd['pre'],core);mask=hd['mask'];xx[:,~mask]=0;model=fac(42+1009*fold).cuda().eval();model.load_state_dict(ck['model'],strict=True)
    with torch.inference_mode():ll=model(torch.as_tensor(xx,device='cuda'),torch.zeros(len(ix),dtype=torch.long,device='cuda')).cpu().numpy()
    q=dict(logit=ll,indices=ix,binding=sha(fp),checkpoint_sha256=sha(cp));save(pp,q);del model
   else:
    bd=torch.load(Path(r['bank']),map_location='cpu',weights_only=False);xx=frozen_transform(rec['E14' if arm=='E1' else 'R14'][ix],meta['patient'][ix],bd['pre'],core)
    if arm in ['E1','D1']:
     model=CompactDelta(fac(42+1009*fold).network,records).cuda().eval();model.load_state_dict(ck['model'],strict=True)
     with torch.inference_mode():ll=model(torch.as_tensor(xx,device='cuda')).cpu().numpy()
     q=dict(logit=ll,indices=ix,binding=sha(fp),checkpoint_sha256=sha(cp));save(pp,q);del model
    else:
     model=make(core,fac,records,a.runtime/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt',42+1009*fold,'cuda');model.load_state_dict(ck['model'],strict=True);model.eval();bank=dict(patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],x=xx)
     q=dynamic_cell(model,bank,ix,store,'true' if arm=='D2' else 'shuffle','cuda',pp,sha(fp),sha(cp),arm,fold);assert core.state_hash(model.state_dict())==core.state_hash(ck['model']);del model
  assert np.isfinite(q['logit']).all();q['threshold']=r['threshold'];cells[arm,fold]=q;files[f'{arm}_{fold}']=sha(pp);print('FROZEN_OUTER_CELL_SEALED',arm,fold,len(ix),flush=True)
 for fold in range(1,6):
  q=cells['D2',fold];base=cells['D1',fold];delta=float(np.abs(q['base_logit']-base['logit']).max());assert delta<=2e-6 and np.array_equal(q['interventions']['gamma0'],q['base_logit']);assert np.array_equal(torch.sigmoid(torch.from_numpy(q['base_logit'])).numpy()>=base['threshold'],torch.sigmoid(torch.from_numpy(base['logit'])).numpy()>=base['threshold']);parity.append(dict(fold=fold,embedded_D1_logit_max_difference=delta,gamma0_exact_embedded_logit=True,D1_original_threshold_decisions_exact=True))
 seal=out/'OUTER_PREDICTIONS_SEAL_PRIVATE.json';obj=dict(freeze_sha256=sha(fp),files=files,all25_predictions_complete_before_y=True)
 if seal.exists():assert json.loads(seal.read_text())==obj
 else:write(seal,obj)
 # First full-cohort label read in this new evaluation, after all25 scores sealed.
 with np.load(archive,allow_pickle=True) as z:y=z['y']
 assert len(y)==7635 and int((y==0).sum())==1743 and int((y==1).sum())==5892
 rs=[];pooled=[];errors=[]
 for fold,d in enumerate(banks,1):assert np.array_equal(y[d['archive_indices']],d['y'])
 for arm in lock['models']:
  scores=np.empty(7635);decisions=np.zeros(7635,bool);visited=[]
  for fold in range(1,6):
   q=cells[arm,fold];ix=q['indices'];score=torch.sigmoid(torch.from_numpy(q['logit'])).numpy();tau=q['threshold'];bank=dict(patient=meta['patient'][ix],channel=meta['channel'][ix],center=meta['center'][ix],y=y[ix],val=np.arange(len(ix)));rs.extend(patient_rows(c,bank,score,tau,arm,fold));scores[ix]=score;decisions[ix]=score>=tau;visited.extend(ix.tolist())
   if arm=='D2':
    for name,ll in q['interventions'].items():rs.extend(patient_rows(c,bank,torch.sigmoid(torch.from_numpy(ll)).numpy(),tau,'D2_'+name,fold))
    rs.extend(patient_rows(c,bank,torch.sigmoid(torch.from_numpy(q['raw_logit'])).numpy(),.5,'D2_auxiliary_head_diagnostic',fold));errors.extend(error_rows(bank,q,tau,fold))
  assert sorted(visited)==list(range(7635));order=np.lexsort((meta['channel'],meta['patient']));p=c.metric(y[order],scores[order],0,pred=decisions[order]);p.pop('ez_mrr');p.pop('top1_is_ez');pooled.append(dict(arm=arm,unit='secondary_pooled_channels',**p))
 df=pd.DataFrame(rs)
 for arm,g in df.groupby('arm'):assert len(g)==g.patient.nunique()==80 and int(g.channels.sum())==7635
 save(out/'OUTER_METRICS_PRIVATE.pkl',rs);df.groupby('arm')[METRICS].mean().reset_index().to_csv(pub/'OUTER_SUMMARY.csv',index=False)
 for key in ['fold','center']:
  pd.DataFrame([dict(arm=arm,**{key:value},patients=len(g),channels=int(g.channels.sum()),**g[METRICS].mean().to_dict()) for (arm,value),g in df.groupby(['arm',key])]).to_csv(pub/f'OUTER_BY_{key.upper()}.csv',index=False)
 boot=[]
 for left,right in lock['bootstrap']['comparisons']:boot.extend(dict(comparison=f'{left} minus {right}',**v) for v in compare(c,df[df.arm==left],df[df.arm==right],METRICS))
 pd.DataFrame(boot).to_csv(pub/'OUTER_PAIRED_BOOTSTRAP.csv',index=False);absolute=[]
 for arm,g in df.groupby('arm'):
  b=c.cluster_bootstrap(g.patient.to_numpy(),g[METRICS].to_numpy(),10000)
  for j,m in enumerate(METRICS):absolute.append(dict(arm=arm,metric=m,estimate=float(g[m].mean()),ci_low=float(np.nanpercentile(b[:,j],2.5)),ci_high=float(np.nanpercentile(b[:,j],97.5)),patients=80,draws=10000,seed=42))
 pd.DataFrame(absolute).to_csv(pub/'OUTER_ABSOLUTE_INTERVALS.csv',index=False);pd.DataFrame(pooled).to_csv(pub/'OUTER_POOLED_CHANNELS.csv',index=False);pd.DataFrame(errors).to_csv(pub/'OUTER_ERROR_AND_RANKING.csv',index=False)
 df[df.arm.eq('D2')|df.arm.str.startswith('D2_')].groupby('arm')[METRICS].mean().reset_index().to_csv(pub/'OUTER_INTERVENTIONS.csv',index=False);write(pub/'OUTER_D1_IDENTITY.json',dict(status='PASS',folds=parity,no_weights_changed=True))
 assert outer_context(a)[10]==binding
 write(pub/'OUTER_STATUS.json',dict(status='ONE_FROZEN_OUTER_PASS_COMPLETE',scientific_status=lock['scientific_status'],patients_per_arm=80,channels_per_arm=7635,models=lock['models'],freeze_sha256=sha(fp),seal_sha256=sha(seal),all25_predictions_sealed_before_y=True,FIT_VAL_label_parity=True,no_retraining=True,no_threshold_or_scaler_refit=True,no_new_extraction=True,original_Development_terminal_unchanged=True));print('RAW_DELTA_OUTER_PASS_COMPLETE',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('phase',choices=['freeze','evaluate']);p.add_argument('--runtime',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));a=p.parse_args();(freeze if a.phase=='freeze' else infer)(a)
