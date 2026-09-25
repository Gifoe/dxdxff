from __future__ import annotations
import copy,contextlib,random,time,numpy as np,torch
from .losses import channel_loss,outcome_loss
def seed_all(x):x=int(x);random.seed(x);np.random.seed(x);torch.manual_seed(x);torch.cuda.manual_seed_all(x)
class EMA:
 def __init__(self,m,d=.98):self.model=copy.deepcopy(m).eval();self.d=d;[p.requires_grad_(False) for p in self.model.parameters()]
 @torch.no_grad()
 def update(self,m):
  for a,b in zip(self.model.parameters(),m.parameters()):a.mul_(self.d).add_(b.detach(),alpha=1-self.d)
def batches(keys,meta,n,seed):
 rng=np.random.default_rng(int(seed));a=[k for k in keys if meta[k]['outcome_success']==1];b=[k for k in keys if meta[k]['outcome_success']==0];rng.shuffle(a);rng.shuffle(b);out=[]
 while a or b:
  r=[]
  for z in (a,b,a,b):
   if z and len(r)<n:r.append(z.pop())
  out.append(r)
 return out
def class_weights(keys,meta,device):
 y=[meta[k]['outcome_success'] for k in keys];n0=max(y.count(0),1);n1=max(y.count(1),1);return torch.tensor([len(y)/(2*n0),len(y)/(2*n1)],device=device)
def epoch(model,ema,cohort,keys,meta,opt,device,args,stage,seed):
 model.train();opt.zero_grad(set_to_none=True);rows=[];g=[];win=0;start=time.time();bs=batches(keys,meta,args.batch_size,seed);cw=class_weights(keys,meta,device)
 for i,keys0 in enumerate(bs):
  views=[cohort.view(k,(seed+i+j)%args.train_views) for j,k in enumerate(keys0)];ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
  with ctx:o=model(views)
  ch=channel_loss([x['channel_rows'] for x in o]);logits=torch.stack([x['logit'].float() for x in o]);y=torch.tensor([v['outcome_success'] for v in views],device=device);out=outcome_loss(logits,y,cw);loss=ch if stage=='a' else out if stage=='b1' else out+args.channel_loss_weight*ch;(loss/args.gradient_accumulation_steps).backward();win+=sum(x['n_windows_encoded'] for x in o)
  if (i+1)%args.gradient_accumulation_steps==0 or i==len(bs)-1:
   g.append(float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],args.gradient_clip)));opt.step();opt.zero_grad(set_to_none=True)
   if ema:ema.update(model)
  rows.append((float(loss.detach()),float(out.detach()),float(ch.detach())))
 return {'train_total':float(np.mean([r[0] for r in rows])),'train_outcome':float(np.mean([r[1] for r in rows])),'train_channel':float(np.mean([r[2] for r in rows])),'gradient_norm':float(np.mean(g)),'n_windows_encoded':win,'epoch_wall_time_seconds':time.time()-start}
@torch.inference_mode()
def infer(model,cohort,keys,meta,device,nviews):
 model.eval();pred=[];audit=[];ch=[]
 for k in sorted(keys):
  vals=[];base=meta[k]
  for v in range(nviews):
   ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
   with ctx:o=model([cohort.view(k,v)])[0]
   val=float(o['logit'].float().cpu());vals.append(val)
   for a in o['seizure_audit']:audit.append({'patient_key':k,'view_id':v,**a,'outcome_true':base['outcome_success']})
   for lg,la in o['channel_rows']:ch += [{'patient_key':k,'logit':float(x),'target_nez':int(y)} for x,y in zip(lg,la)]
  z=float(np.mean(vals));p=float(torch.sigmoid(torch.tensor(z)));pred.append({'patient_key':k,'center':base['center'],'outcome_true':base['outcome_success'],'probability_success':p,'probability_failure':1-p,'logit_success':z,'prediction_05':int(p>=.5),'view_logit_std':float(np.std(vals))})
 return pred,audit,ch
