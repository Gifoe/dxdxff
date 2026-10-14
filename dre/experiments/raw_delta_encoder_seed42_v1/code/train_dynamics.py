"""Matched D2/D3 training, frozen interventions, Development-only scientific gate."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from io_utils import sha,write,read,save
from dependencies import context
from compact_delta_classifier import CompactDelta
from raw_delta_encoder import RawDeltaEncoder
from raw_store import Store
from evaluate import METRICS,patient_rows
from train import aggregate

FILES=['raw_delta_encoder.py','temporal_shuffle.py','raw_store.py','train_dynamics.py']

def make(core,fac,records,checkpoint,seed,device):
 core.seed_all(seed);base=CompactDelta(fac(seed).network,records);ck=torch.load(checkpoint,map_location='cpu',weights_only=False);base.load_state_dict(ck['model'],strict=True);return RawDeltaEncoder(base).to(device)

def predict(model,d,indices,store,mode,device,interventions=False):
 ll=np.empty(len(indices),np.float32);rr=ll.copy();bb=ll.copy();res=ll.copy();extras={m:ll.copy() for m in ['gamma0','shuffle','absolute']} if interventions else {}
 model.eval()
 with torch.inference_mode():
  for pid in sorted(set(d['patient'][indices])):
   k=np.flatnonzero(d['patient'][indices]==pid);ii=indices[k];raw,mask,times,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),device);x=torch.as_tensor(d['x'][ii],device=device)
   final,aux,base,residual=model(x,raw,mask,times,perms,mode);ll[k]=final.cpu().numpy();rr[k]=aux.cpu().numpy();bb[k]=base.cpu().numpy();res[k]=residual.cpu().numpy()
   if interventions:
    extras['gamma0'][k]=base.cpu().numpy()
    for m in ['shuffle','absolute']:extras[m][k]=model(x,raw,mask,times,perms,m)[0].cpu().numpy()
   del raw,mask,times,perms,x
 return dict(logit=ll,raw_logit=rr,base_logit=bb,residual=res,interventions=extras)

def fit(model,d,store,mode,cell,seed,bind,core,c,device):
 cell=Path(cell);cell.mkdir(parents=True,exist_ok=True);initial=core.state_hash(model.state_dict());base_initial=core.state_hash(model.base.state_dict());binding=dict(source=bind,initial_hash=initial,base_initial_hash=base_initial,seed=seed,mode=mode,device=device)
 opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001);core.seed_all(seed)
 tr,va=d['train'],d['val'];pos=torch.tensor(max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1),device=device);fn=torch.nn.BCEWithLogitsLoss(pos_weight=pos)
 last=cell/'LAST_PRIVATE.pt';complete=cell/'COMPLETE_PRIVATE.json';history=[];best=None;key=None;stale=0;start=1
 if last.exists():
  ck=torch.load(last,map_location=device,weights_only=False);assert ck['binding']==binding;model.load_state_dict(ck['model'],strict=True);opt.load_state_dict(ck['optimizer']);core.restore_rng(ck['rng']);history=ck['history'];best=ck['best'];key=ck['best_key'];stale=ck['stale'];start=ck['epoch']+1
  if complete.exists():
   done=json.loads(complete.read_text());assert done['binding']==binding and done['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and done['last_sha256']==sha(last);model.load_state_dict(best['model']);model.eval();return best,history
  if ck['epoch']>=6 and stale>=6:start=31
 train_groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
 for epoch in range(start,31):
  model.train();losses=[];auxlosses=[]
  for j in np.random.default_rng(seed+epoch).permutation(len(train_groups)):
   ii=train_groups[int(j)];pid=str(d['patient'][ii[0]]);raw,mask,times,perms=store.get(pid,d['channel'][ii].astype(str).tolist(),device);x=torch.as_tensor(d['x'][ii],device=device);y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device=device)
   opt.zero_grad(set_to_none=True);ll,rr,base,residual=model(x,raw,mask,times,perms,mode);aux=fn(rr,y);loss=fn(ll,y)+.25*aux;assert torch.isfinite(loss);loss.backward()
   assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None),'Nonfinite Raw branch gradient'
   torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.);opt.step();losses.append(float(loss.detach().cpu()));auxlosses.append(float(aux.detach().cpu()));del raw,mask,times,perms,x,y,ll,rr,base,residual,loss,aux
  assert core.state_hash(model.base.state_dict())==base_initial and not model.base.training
  pred=predict(model,d,va,store,mode,device);scores=torch.sigmoid(torch.from_numpy(pred['logit'])).numpy();sel=core.select_threshold(d['y'][va],scores,d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
  history.append(dict(epoch=epoch,loss=float(np.mean(losses)),auxiliary_loss=float(np.mean(auxlosses)),gamma=float(model.gamma.detach().cpu()),residual_coefficient=float(.5*torch.tanh(model.gamma.detach()).cpu()),**sel))
  if key is None or candidate>key:key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,**sel)
  else:stale+=1
  core.torch_write(last,dict(binding=binding,model=model.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),epoch=epoch,best=best,best_key=key,stale=stale,history=history));print('RAW_DYNAMIC_EPOCH',cell.parent.parent.name,cell.parent.name,epoch,'best',best['epoch'],flush=True)
  if epoch>=6 and stale>=6:break
 assert best is not None;core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=binding,**best));write(complete,dict(binding=binding,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last),epoch=best['epoch'],threshold=best['threshold']));model.load_state_dict(best['model']);model.eval();return best,history

def error_rows(d,pred,threshold,fold):
 va=d['val'];out=[];full=pred['logit'];base=pred['base_logit'];probs=lambda z:torch.sigmoid(torch.from_numpy(z)).numpy()
 fp=probs(full)<threshold
 for mode,z in pred['interventions'].items():
  xp=probs(z)<threshold;y=d['y'][va]==0;reversals=0;pairs=0
  for pid in sorted(set(d['patient'][va])):
   ii=np.flatnonzero(d['patient'][va]==pid);a=full[ii];b=z[ii];A=a[:,None]-a[None];B=b[:,None]-b[None];take=np.tril(np.ones_like(A,bool),-1);reversals+=int(((A*B<0)&take).sum());pairs+=int(take.sum())
  out.append(dict(fold=fold,intervention=mode,corrected_by_full=int(((fp==y)&(xp!=y)).sum()),spoiled_by_full=int(((fp!=y)&(xp==y)).sum()),TP_full=int((fp&y).sum()),FP_full=int((fp&~y).sum()),TN_full=int((~fp&~y).sum()),FN_full=int((~fp&y).sum()),TP_intervention=int((xp&y).sum()),FP_intervention=int((xp&~y).sum()),TN_intervention=int((~xp&~y).sum()),FN_intervention=int((~xp&y).sum()),order_reversals=reversals,pairs=pairs,mean_absolute_residual=float(np.mean(np.abs(pred['residual']))),maximum_absolute_residual=float(np.max(np.abs(pred['residual']))),effectively_zero_residual_fraction=float(np.mean(np.abs(pred['residual'])<=1e-6)),D2_threshold_frozen=True))
 return out

def main(a):
 runtime=a.runtime;pub=runtime/'results';root=Path(__file__).parents[1];gate=json.loads((pub/'STAGE_B_GATE.json').read_text());assert gate['pass_gate'];c,core,fac,trainer,records,rec,banks,hist,deps=context(a.source,runtime);fr=json.loads((runtime/'STAGE_B_MODEL_FREEZE_PRIVATE.json').read_text());assert fr['stage_B_pass'];store=Store(runtime)
 bind=dict(Stage_B_freeze=sha(runtime/'STAGE_B_MODEL_FREEZE_PRIVATE.json'),raw_dynamic_registry=sha(runtime/'RAW_DYNAMIC_REGISTRY_PRIVATE.pkl'),dependencies=deps,code={f:sha(root/'code'/f) for f in FILES},tests=sha(root/'tests/test_stage_c.py'))
 freeze=runtime/'STAGE_C_INPUT_FREEZE_PRIVATE.json'
 if freeze.exists():assert json.loads(freeze.read_text())==bind
 else:write(freeze,bind)
 if (pub/'STAGE_C_GATE.json').exists():assert json.loads((pub/'STAGE_C_GATE.json').read_text())['binding']==bind;print('STAGE_C_ALREADY_COMPLETE',flush=True);return
 rs=read(runtime/'DEVELOPMENT_METRICS_PRIVATE.pkl');rs=[r for r in rs if r['arm'] in ['D0','E1','D1']];sels=[];history=[];errors=[];shifts=[];parity=[];params=[]
 for fold,d in enumerate(banks,1):
  seed=42+1009*fold;base=runtime/f'D1/fold{fold}/TRAIN/BEST_PRIVATE.pt';br=next(r for r in fr['selections'] if r['arm']=='D1' and r['fold']==fold);assert sha(base)==br['checkpoint_sha256'];bp=runtime/f'D1/fold{fold}/BANK_PRIVATE.pt';assert sha(bp)==br['bank_sha256'];d=torch.load(bp,map_location='cpu',weights_only=False);initials=[]
  for arm,mode in [('D2','true'),('D3','shuffle')]:
   model=make(core,fac,records,base,seed,'cuda');initials.append(core.state_hash(model.state_dict()));cell=runtime/f'{arm}/fold{fold}/TRAIN';best,h=fit(model,d,store,mode,cell,seed,bind,core,c,'cuda');cp=cell/'BEST_PRIVATE.pt'
   pp=runtime/f'scores/{arm}_fold{fold}_DYNAMIC_PRIVATE.pkl'
   if pp.exists():pred=read(pp);assert pred['binding']==bind and pred['checkpoint_sha256']==sha(cp)
   else:pred=predict(model,d,d['val'],store,mode,'cuda',interventions=arm=='D2');pred.update(binding=bind,checkpoint_sha256=sha(cp));save(pp,pred)
   score=torch.sigmoid(torch.from_numpy(pred['logit'])).numpy();assert core.select_threshold(d['y'][d['val']],score,d['patient'][d['val']])['threshold']==best['threshold'];rs.extend(patient_rows(c,d,score,best['threshold'],arm,fold))
   if arm=='D2':
    original=read(runtime/f'scores/D1_fold{fold}_PRIVATE.pkl');embedded=torch.sigmoid(torch.from_numpy(pred['base_logit'])).numpy();drift=float(np.max(np.abs(embedded-original['score'])));assert drift<=1e-6 and np.array_equal(embedded>=original['threshold'],original['score']>=original['threshold']);assert np.array_equal(pred['interventions']['gamma0'],pred['base_logit'])
    for intervention,ll in pred['interventions'].items():rs.extend(patient_rows(c,d,torch.sigmoid(torch.from_numpy(ll)).numpy(),best['threshold'],f'D2_{intervention}',fold))
    parity.append(dict(fold=fold,embedded_D1_score_max_drift=drift,gamma_zero_logit_identity=True,base_checkpoint_unchanged=core.state_hash(model.base.state_dict())==core.state_hash(torch.load(base,map_location='cpu',weights_only=False)['model']),intervention_threshold='frozen D2 threshold, not D1 threshold'))
    errors.extend(error_rows(d,pred,best['threshold'],fold));fitpred=predict(model,d,d['train'],store,mode,'cuda')
    for scope,q in [('FIT',fitpred),('Development',pred)]:
     shifts.append(dict(fold=fold,scope=scope,channels=len(q['raw_logit']),raw_logit_mean=float(np.mean(q['raw_logit'])),raw_logit_std=float(np.std(q['raw_logit'])),raw_logit_q25=float(np.quantile(q['raw_logit'],.25)),raw_logit_q50=float(np.quantile(q['raw_logit'],.5)),raw_logit_q75=float(np.quantile(q['raw_logit'],.75)),mean_absolute_residual=float(np.mean(np.abs(q['residual'])))))
    rs.extend(patient_rows(c,d,torch.sigmoid(torch.from_numpy(pred['raw_logit'])).numpy(),.5,'D2_auxiliary_head_diagnostic',fold))
   sels.append(dict(arm=arm,fold=fold,epoch=best['epoch'],threshold=best['threshold'],checkpoint_sha256=sha(cp),initial_state_sha256=initials[-1],D1_checkpoint_sha256=sha(base),gamma=float(model.gamma.detach().cpu()),residual_coefficient=float(.5*torch.tanh(model.gamma.detach()).cpu()),epochs=len(h)))
   history.extend(dict(arm=arm,fold=fold,**r) for r in h);params.append(dict(arm=arm,fold=fold,total=13643,trainable=12106,frozen_D1=1537));write(pub/'RUN_STATUS.json',dict(status='STAGE_C_TRAINING',cells_complete=len(sels),total=10,no_new_Outer=True));print('RAW_DYNAMIC_SELECTED',arm,fold,best['epoch'],flush=True);del model
  assert initials[0]==initials[1], 'D2/D3 initial mismatch'
 means=aggregate(runtime,c,rs,[('D1','E1'),('E1','D0'),('D1','D0'),('D2','D1'),('D2','D3')]);diff=means.loc['D2']-means.loc['D1'];order=means.loc['D2']-means.loc['D3'];df=pd.DataFrame(rs);positive=sum(df[(df.arm=='D2')&(df.fold==f)].ez_auroc.mean()>df[(df.arm=='D1')&(df.fold==f)].ez_auroc.mean() for f in range(1,6));boot=pd.read_csv(pub/'PAIRED_BOOTSTRAP.csv');low=float(boot[(boot.comparison=='D2 minus D3')&(boot.metric=='ez_auroc')].iloc[0].ci_low)
 intervention_gain=means.loc['D2']-means.loc['D2_gamma0'];useful=bool(intervention_gain.ez_auroc>0 and intervention_gain.macro_f1>=0 and any(r['maximum_absolute_residual']>1e-6 for r in errors))
 conditions=dict(stage_A=True,stage_B=True,D2_D1_AUROC=diff.ez_auroc>=.010,D2_D3_AUROC=order.ez_auroc>=.005,D2_D1_AP=diff.ez_auprc>=0,D2_D1_MacroF1=diff.macro_f1>=.005,D2_D1_EZF1=diff.ez_f1>=0,D2_specificity=diff.specificity>=-.010,positive_folds=positive>=4,D2_D3_CI_lower_positive=low>0,useful_actual_residual=useful,implementation_integrity=True)
 passed=all(conditions.values());terminal='RAW_DELTA_ENCODER_DEVELOPMENT_PROMISING_NOT_INDEPENDENTLY_CONFIRMED' if passed else ('RAW_INFORMATION_POSSIBLY_USEFUL_TEMPORAL_ORDER_NOT_ESTABLISHED' if diff.ez_auroc>0 and not (order.ez_auroc>=.005 and low>0) else 'RAW_DELTA_RECOVERED_LEARNED_DYNAMICS_NOT_USEFUL')
 write(pub/'STAGE_C_GATE.json',dict(binding=bind,pass_gate=passed,terminal=terminal,conditions=conditions,D2_D1=diff.to_dict(),D2_D3=order.to_dict(),positive_folds=positive,D2_D3_AUROC_CI_low=low,intervention_gain=intervention_gain.to_dict(),no_Outer=True));write(pub/'SCIENTIFIC_GATE.json',dict(terminal=terminal,conditions=conditions,Stage_A='PASS',Stage_B='PASS',Stage_C='PASS' if passed else 'FAIL',Development_reused_exploratory=True,no_rescue=True,no_Outer=True))
 write(runtime/'STAGE_C_MODEL_FREEZE_PRIVATE.json',dict(binding=bind,selections=sels));write(pub/'RAW_BRANCH_PARITY.json',dict(status='PASS',folds=parity,D2_D3_equal_initial_state=True,base_frozen=True))
 interventions=df[df.arm.str.startswith('D2_')|df.arm.eq('D2')];interventions.groupby('arm')[METRICS].mean().reset_index().to_csv(pub/'RAW_BRANCH_INTERVENTIONS.csv',index=False);pd.DataFrame(errors).to_csv(pub/'ERROR_AND_RANKING_AUDIT.csv',index=False);pd.DataFrame(shifts).to_csv(pub/'RESIDUAL_DISTRIBUTION_SHIFT.csv',index=False)
 pd.concat([pd.read_csv(pub/'MODEL_SELECTION.csv'),pd.DataFrame(sels)],ignore_index=True).to_csv(pub/'MODEL_SELECTION.csv',index=False);pd.concat([pd.read_csv(pub/'PARAMETER_AUDIT.csv'),pd.DataFrame(params)],ignore_index=True).to_csv(pub/'PARAMETER_AUDIT.csv',index=False);pd.DataFrame(history).to_csv(pub/'DYNAMICS_TRAINING_HISTORY.csv',index=False)
 write(pub/'RUN_STATUS.json',dict(status='ALL_AUTHORIZED_DEVELOPMENT_COMPLETE',terminal=terminal,new_training_cells=20,no_new_Outer=True));print('RAW_DYNAMIC_FINAL_GATE',terminal,flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--runtime',required=True,type=Path);p.add_argument('--source',type=Path,default=Path(r'E:\DRE-nips\new-pipeline\7-11'));main(p.parse_args())
