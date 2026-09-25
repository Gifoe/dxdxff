from __future__ import annotations
import copy,contextlib,hashlib,random,time,numpy as np,torch
import torch.nn.functional as F
def seed_all(seed):
 # pandas fold IDs are commonly numpy integer scalars; Python 3.11's
 # random.seed accepts only builtin numeric types.
 seed=int(seed);random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
class EMA:
 def __init__(self,model,decay):self.model=copy.deepcopy(model).eval();self.decay=decay;[p.requires_grad_(False) for p in self.model.parameters()]
 @torch.no_grad()
 def update(self,model):
  for dst,src in zip(self.model.parameters(),model.parameters()):dst.mul_(self.decay).add_(src.detach(),alpha=1-self.decay)
def batches(keys,cohort,size,seed):
 rng=np.random.default_rng(seed);pos=[k for k in keys if cohort.load(k)['outcome_success']==1];neg=[k for k in keys if cohort.load(k)['outcome_success']==0];rng.shuffle(pos);rng.shuffle(neg);out=[]
 while pos or neg:
  row=[]
  for src in (pos,neg,pos,neg):
   if src and len(row)<size:row.append(src.pop())
  if row:out.append(row)
 return out
def loss_fn(outputs,labels,weight):
 logits=torch.stack([x['logit'].float() for x in outputs]);y=torch.tensor(labels,device=logits.device,dtype=torch.float32);bce=F.binary_cross_entropy_with_logits(logits,y);pos=logits[y==1];neg=logits[y==0];rank=F.softplus(-(pos[:,None]-neg[None,:])).mean() if len(pos) and len(neg) else logits.new_zeros(());return bce+weight*rank,{'bce':float(bce.detach()),'rank':float(rank.detach())}
def train_epoch(model,ema,cohort,keys,opt,device,epoch,args):
 model.train();opt.zero_grad(set_to_none=True);rows=[];wins=0;start=time.time();gn=0.;batch_rows=batches(keys,cohort,args.batch_size,args.seed+epoch)
 for bi,batch in enumerate(batch_rows):
  views=[cohort.view(k,(epoch+int.from_bytes(hashlib.sha256(k.encode()).digest()[:2],'big'))%args.n_deterministic_views) for k in batch];ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
  with ctx:outputs=model(views)
  wins+=sum(x['n_windows_encoded'] for x in outputs);loss,parts=loss_fn(outputs,[v['outcome_success'] for v in views],args.rank_weight);(loss/args.gradient_accumulation_steps).backward()
  if (bi+1)%args.gradient_accumulation_steps==0 or bi==len(batch_rows)-1:gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip));opt.step();ema.update(model);opt.zero_grad(set_to_none=True)
  rows.append({'loss':float(loss.detach()),**parts})
 return {'train_total':float(np.mean([r['loss'] for r in rows])),'train_bce':float(np.mean([r['bce'] for r in rows])),'train_rank':float(np.mean([r['rank'] for r in rows])),'gradient_norm':gn,'epoch_wall_time_seconds':time.time()-start,'n_windows_encoded':wins}
@torch.inference_mode()
def infer(model,cohort,keys,device,n_views):
 model.eval();pred=[];views=[];seizures=[];patients=[];stats=[];attn=[]
 for key in sorted(keys):
  values=[];base=cohort.load(key)
  for vid in range(n_views):
   view=cohort.view(key,vid);ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
   with ctx:o=model([view])[0]
   logit=float(o['logit'].float().cpu());values.append(logit);views.append({'patient_key':key,'center':base['center'],'view_id':vid,'logit':logit,'probability_success':float(torch.sigmoid(torch.tensor(logit))),'outcome_true':base['outcome_success']})
   for si,(u,a) in enumerate(zip(o['seizure_embeddings'],o['seizure_audit'])):
    seizures.append({'patient_key':key,'view_id':vid,'seizure_id':a['seizure_id'],**{f'embedding_{i}':float(x) for i,x in enumerate(u)},'outcome_true':base['outcome_success']});stats.append({'patient_key':key,'view_id':vid,**a,'outcome_true':base['outcome_success']});attn.append({'patient_key':key,'view_id':vid,'seizure_id':a['seizure_id'],'ez_to_nez_norm':a['cross_ez_to_nez_norm'],'nez_to_ez_norm':a['cross_nez_to_ez_norm']})
   patients.append({'patient_key':key,'view_id':vid,**{f'embedding_{i}':float(x) for i,x in enumerate(o['patient_embedding'])},'outcome_true':base['outcome_success']})
  logit=float(np.mean(values));p=float(torch.sigmoid(torch.tensor(logit)));pred.append({'patient_key':key,'center':base['center'],'outcome_true':base['outcome_success'],'probability_success':p,'logit':logit,'prediction_05':int(p>=.5),'view_logit_std':float(np.std(values))})
 return pred,{'views':views,'seizures':seizures,'patients':patients,'stats':stats,'attention':attn}
