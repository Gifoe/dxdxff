from __future__ import annotations
import copy, random, time, gc, numpy as np, torch
import torch.nn.functional as F
from .raw_dataset import patient_tensors
from .losses import trace_loss, pairwise_logistic

def seed_all(seed): random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
def batches(patients,size,seed):
 rng=np.random.default_rng(seed); pos=[x for x in patients if x.outcome_success]; neg=[x for x in patients if not x.outcome_success];rng.shuffle(pos);rng.shuffle(neg); out=[]
 while pos or neg:
  b=[]
  for source in (pos,neg,pos,neg):
   if source and len(b)<size:b.append(source.pop())
  if b: out.append(b)
 return out

class ModelEMA:
 def __init__(self, model, decay=.995): self.decay=decay; self.model=copy.deepcopy(model).eval(); [p.requires_grad_(False) for p in self.model.parameters()]
 @torch.no_grad()
 def update(self, model):
  for ema, current in zip(self.model.parameters(), model.parameters()): ema.mul_(self.decay).add_(current.detach(),alpha=1-self.decay)
  for ema, current in zip(self.model.buffers(), model.buffers()): ema.copy_(current)
 def state_dict(self): return self.model.state_dict()
 def load_state_dict(self, state): self.model.load_state_dict(state)

def is_cuda_oom(error): return isinstance(error,torch.cuda.OutOfMemoryError) or ('cuda out of memory' in str(error).lower())
def memory_snapshot(device, budget_fraction=.75):
 if device.type!='cuda': return {'peak_memory_allocated_gb':0.,'peak_memory_reserved_gb':0.,'total_dedicated_vram_gb':0.,'vram_budget_gb':0.}
 total=torch.cuda.get_device_properties(device).total_memory
 return {'peak_memory_allocated_gb':torch.cuda.max_memory_allocated(device)/2**30,'peak_memory_reserved_gb':torch.cuda.max_memory_reserved(device)/2**30,'total_dedicated_vram_gb':total/2**30,'vram_budget_gb':total*budget_fraction/2**30}
def workload(batch):
 return [{'patient_key':p.patient_key,'seizures':len(p.seizures),'channels':int(sum(len(s['labels']) for s in p.seizures)),'windows':int(sum(len(__import__('neuroez_c.task2.trace_rawmil.raw_dataset',fromlist=['window_index']).window_index(s))*len(s['labels']) for s in p.seizures))} for p in batch]
def train_epoch(model,patients,opt,ema,device,seed,args,epoch):
 device=torch.device(device);model.train(); model.configure_encoder(args.current_train_window_batch_size,args.gradient_checkpoint_encoder,True); values=[]; opt.zero_grad(set_to_none=True); accumulation=int(args.gradient_accumulation_steps); pending=0; oom_backoffs=0; flat_windows=chunks=0; started=time.perf_counter()
 for batch_index,batch in enumerate(batches(patients,args.batch_size,seed)):
  rng=np.random.default_rng(seed+batch_index); tensors=[patient_tensors(p,'cpu',rng,args.max_seizures_train,args.max_ez_channels_train,args.max_nez_channels_train,args.max_windows_per_phase_train) for p in batch]
  if any(not x for x in tensors): continue
  while True:
   try:
    out=model(tensors); labels=[p.outcome_success for p in batch]; loss,parts=trace_loss(out,labels,epoch); (loss/accumulation).backward(); break
   except RuntimeError as error:
    if not (getattr(args,'auto_window_batch_backoff',True) and device.type=='cuda' and is_cuda_oom(error)): raise
    del error; gc.collect(); opt.zero_grad(set_to_none=True);torch.cuda.empty_cache();pending=0;oom_backoffs+=1;old=args.current_train_window_batch_size;new=old//2
    if new < args.min_window_batch_size:
     info=memory_snapshot(device,args.vram_budget_fraction);raise RuntimeError(f'CUDA OOM at minimum window batch size={old}; patient workload={workload(batch)}; memory={info}')
    args.current_train_window_batch_size=new;model.configure_encoder(new,args.gradient_checkpoint_encoder,True);print(f'[TRACE-RawMIL][OOM] train window batch {old} -> {new}; retrying patient batch; workload={workload(batch)}',flush=True)
  flat_windows+=model.last_n_flat_windows;chunks+=model.last_encoder_chunks
  pending+=1
  grad_norm=np.nan
  if pending == accumulation:
   grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip_norm)); opt.step();ema.update(model);opt.zero_grad(set_to_none=True);pending=0
  batch_logits=torch.stack([x['logit'] for x in out]).detach();values.append({'total':float(loss.detach()),**{k:(float(v) if torch.is_tensor(v) else v) for k,v in parts.items()},'success_logit_mean':float(torch.stack([x['logit'] for x,p in zip(out,batch) if p.outcome_success]).detach().mean()) if any(p.outcome_success for p in batch) else np.nan,'failure_logit_mean':float(torch.stack([x['logit'] for x,p in zip(out,batch) if not p.outcome_success]).detach().mean()) if any(not p.outcome_success for p in batch) else np.nan,'logit_std':float(batch_logits.std(unbiased=False)),'gradient_norm':grad_norm})
 if pending:
  grad_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip_norm)); opt.step();ema.update(model);opt.zero_grad(set_to_none=True)
  values[-1]['gradient_norm']=grad_norm
 if not values:return {}
 result={k:float(np.nanmean([x[k] for x in values])) for k in values[0]};result.update({'oom_backoff_count':oom_backoffs,'n_flat_windows':flat_windows,'encoder_chunks':chunks,'windows_per_second':flat_windows/max(time.perf_counter()-started,1e-9),**memory_snapshot(device,args.vram_budget_fraction)});return result

@torch.inference_mode()
def infer(model,patients,device,window_batch_size=4096,auto_backoff=True,min_window_batch_size=128,vram_budget_fraction=.75):
 device=torch.device(device);model.eval(); model.configure_encoder(window_batch_size,False,device.type=='cuda'); rows=[]; details=[];oom_backoffs=0;flat_windows=chunks=0
 for p in patients:
  # rng=None and no maxima means all seizures, channels, and raw windows in fixed source order.
  data=patient_tensors(p,'cpu',rng=None,max_seizures=None,max_ez=None,max_nez=None,max_windows=None)
  if not data:continue
  while True:
   try: o=model([data])[0];break
   except RuntimeError as error:
    if not (auto_backoff and device.type=='cuda' and is_cuda_oom(error)): raise
    del error;gc.collect();torch.cuda.empty_cache();old=model.encoder_window_batch_size;new=old//2;oom_backoffs+=1
    if new < min_window_batch_size: raise RuntimeError(f'CUDA OOM in deterministic inference at minimum window batch size={old}; patient workload={workload([p])}; memory={memory_snapshot(device,vram_budget_fraction)}')
    model.configure_encoder(new,False,True);print(f'[TRACE-RawMIL][OOM] eval window batch {old} -> {new}; retrying patient {p.patient_key}',flush=True)
  flat_windows+=model.last_n_flat_windows;chunks+=model.last_encoder_chunks; rows.append({'patient_key':p.patient_key,'center':p.center,'outcome_true':p.outcome_success,'logit':float(o['logit'].cpu()),'probability_success':float(torch.sigmoid(o['logit']).cpu()),'patient_residual_strength':float(o['rho'].cpu()),'n_seizures':len(data)})
  for si,(beta,detail) in enumerate(zip(o['beta'],o['details'])):
   run=p.seizures[si]
   for cid,logit,alpha in zip(detail['channel_ids'].cpu().tolist(),detail['logits'].cpu().tolist(),detail['alpha'].cpu().tolist()):details.append({'patient_key':p.patient_key,'center':p.center,'seizure_id':run['seizure_id'],'channel_name':run['names'][cid],'channel_label_nez':1,'residual_logit':logit,'residual_probability':float(torch.sigmoid(torch.tensor(logit))),'sparse_attention_weight':alpha,'is_selected_by_sparsemax':bool(alpha>0),'seizure_reliability_beta':float(beta.cpu()),'outcome_true':p.outcome_success})
 model.last_infer_performance={'current_eval_window_batch_size':model.encoder_window_batch_size,'oom_backoff_count':oom_backoffs,'n_flat_windows':flat_windows,'encoder_chunks':chunks,**memory_snapshot(device,vram_budget_fraction)};return rows,details

def validation_statistics(rows):
 logits=torch.tensor([x['logit'] for x in rows],dtype=torch.float); labels=torch.tensor([x['outcome_true'] for x in rows],dtype=torch.float)
 bce=F.binary_cross_entropy_with_logits(logits,labels); rank=pairwise_logistic(logits,labels)
 return {'validation_bce':float(bce),'validation_rank':float(rank),'validation_selection_loss':float(bce+.2*rank)}

def better_checkpoint(candidate_selection_loss, best_selection_loss):
 return candidate_selection_loss < best_selection_loss
