from __future__ import annotations
import copy,contextlib,random,time,numpy as np,torch
from . import MODEL_VERSION
from .losses import loss_fn
from .evaluation import metric_row,selection_score
def seed_all(seed):random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
class EMA:
 def __init__(self,m,decay=.98):self.model=copy.deepcopy(m).eval();self.decay=decay;[p.requires_grad_(False) for p in self.model.parameters()]
 @torch.no_grad()
 def update(self,m):
  for a,b in zip(self.model.parameters(),m.parameters()):a.mul_(self.decay).add_(b.detach(),alpha=1-self.decay)
def balanced_batches(keys,cohort,size,seed):
 rng=np.random.default_rng(seed);pos=[k for k in keys if cohort.load(k)['outcome_success']==1];neg=[k for k in keys if cohort.load(k)['outcome_success']==0];rng.shuffle(pos);rng.shuffle(neg);out=[]
 while pos or neg:
  b=[]
  for src in (pos,neg,pos,neg):
   if src and len(b)<size:b.append(src.pop())
  if b:out.append(b)
 return out
def train_epoch(model,ema,cohort,keys,opt,device,epoch,args):
 model.train();opt.zero_grad(set_to_none=True);stats=[];start=time.time();steps=0;epoch_windows=0;batches=balanced_batches(keys,cohort,args.batch_size,args.seed+epoch)
 for bi,batch in enumerate(batches):
  views=[cohort.view(k,(epoch+cohort.seed%4+int.from_bytes(__import__('hashlib').sha256(k.encode()).digest()[:2],'big'))%args.n_deterministic_views) for k in batch];ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
  with ctx:outputs=model(views)
  epoch_windows+=sum(x['n_windows_encoded'] for x in outputs);loss,parts=loss_fn(outputs,[v['outcome_success'] for v in views],args.rank_weight);(loss/args.gradient_accumulation_steps).backward();steps+=1
  if steps%args.gradient_accumulation_steps==0 or bi==len(batches)-1:gn=float(torch.nn.utils.clip_grad_norm_(model.parameters(),args.gradient_clip));opt.step();ema.update(model);opt.zero_grad(set_to_none=True)
  stats.append({'loss':float(loss.detach()),**parts})
 return {'train_total':float(np.mean([x['loss'] for x in stats])),'train_bce':float(np.mean([x['bce'] for x in stats])),'train_rank':float(np.mean([x['rank'] for x in stats])),'epoch_wall_time_seconds':time.time()-start,'n_windows_encoded':epoch_windows,'gradient_norm':gn}
@torch.inference_mode()
def infer(model,cohort,keys,device,n_views):
 model.eval();rows=[];view_rows=[];components=[];seizure_rows=[];channel_rows=[];vdr_rows=[];vdr_node_rows=[];rec_rows=[]
 for key in sorted(keys):
  logits=[];outs=[]
  for vid in range(n_views):
   view=cohort.view(key,vid);ctx=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else contextlib.nullcontext()
   with ctx:o=model([view])[0]
   logit=float(o['logit'].float().cpu());logits.append(logit);outs.append((view,o));view_rows.append({'patient_key':key,'view_id':vid,'logit':logit,'probability_success':float(torch.sigmoid(torch.tensor(logit))),'outcome_true':view['outcome_success']})
  logit=float(np.mean(logits));p=float(torch.sigmoid(torch.tensor(logit)));base=cohort.load(key);rows.append({'patient_key':key,'center':base['center'],'outcome_true':base['outcome_success'],'logit':logit,'probability_success':p,'prediction_05':int(p>=.5),'view_logit_std':float(np.std(logits))})
  escape=[];vd=[];rec=[]
  for view,o in outs:
   rec.append(o['recurrence_meta'])
   for s in o['seizures']:
    prop=s['prop'];v=s['vdr'];seizure_rows.append({'patient_key':key,'view_id':view['view_id'],'seizure_id':s['seizure_id'],'escape_burden':float(prop['stats'][5]) if prop else 0.})
    if prop:
     escape.append(prop['stats']);
     for name,lg,pr,tau,st,de,sel in zip(prop['names'],prop['logits'],prop['probability'],prop['tau'],prop['strength'],prop['delay'],prop['selected']):channel_rows.append({'patient_key':key,'seizure_id':s['seizure_id'],'channel_name':name,'channel_label_nez':1,'escape_logit':float(lg),'escape_probability':float(pr),'recruitment_time':float(tau),'propagation_strength':float(st),'expected_delay':float(de),'selected_top_q10':bool(sel),'outcome_true':base['outcome_success']})
    if v:
     vd.append(v['stats']);st=v['selected_nez_tau'];vdr_rows.append({'model_version':MODEL_VERSION,'patient_key':key,'view_id':view['view_id'],'seizure_id':s['seizure_id'],'uses_true_ez_supernode':v['uses_true_ez_supernode'],'ez_tau':float(v['ez_tau']),'selected_nez_tau_mean':float(st.mean()) if len(st) else np.nan,'selected_nez_tau_min':float(st.min()) if len(st) else np.nan,'selected_nez_tau_max':float(st.max()) if len(st) else np.nan,'ez_self_similarity':float(v['ez_self_similarity']),'n_nodes':v['n_nodes'],'n_nez_nodes':v['n_nez_nodes'],'ez_outgoing_burden':float(v['stats'][0]),'nez_residual_density':float(v['stats'][1]),'vdr_one':float(v['stats'][2]),'vdr_two':float(v['stats'][3]),'rho_nez':float(v['stats'][4])});vdr_node_rows.append({'patient_key':key,'view_id':view['view_id'],'seizure_id':s['seizure_id'],'node_type':'EZ_SUPERNODE','node_index':0,'channel_name':'__TRUE_EZ_SUPERNODE__','recruitment_time':float(v['ez_tau']),'selected_top_q10':False})
     selected_names=[prop['names'][i] for i,x in enumerate(prop['selected']) if bool(x)]
     for ni,(name,tau) in enumerate(zip(selected_names,st),1):vdr_node_rows.append({'patient_key':key,'view_id':view['view_id'],'seizure_id':s['seizure_id'],'node_type':'NEZ_TOP_Q10','node_index':ni,'channel_name':name,'recruitment_time':float(tau),'selected_top_q10':True})
  e=torch.stack(escape).mean(0) if escape else torch.zeros(10);vv=torch.stack(vd).mean(0) if vd else torch.zeros(5);rm=rec[0] if rec else {};rec_rows.append({'patient_key':key,**rm});components.append({'patient_key':key,'outcome_true':base['outcome_success'],'probability_success':p,'escape_burden_mean':float(e[5]),'escape_burden_max':float(e[1]),'vdr_one_mean':float(vv[2]),'vdr_two_mean':float(vv[3]),'rho_nez_mean':float(vv[4]),'recurrent_escape':rm.get('recurrent_escape',0.),'topq10_jaccard':rm.get('topq10_jaccard',0.),'recurrence_present':rm.get('recurrence_present',0),'view_logit_std':float(np.std(logits))})
 return rows,{'views':view_rows,'components':components,'seizures':seizure_rows,'channels':channel_rows,'vdr':vdr_rows,'vdr_nodes':vdr_node_rows,'recurrence':rec_rows}
