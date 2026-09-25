from __future__ import annotations
import math,torch
from torch import nn
from .encoder import RawWindowEncoder
from .recruitment import RecruitmentDynamics
from .propagation import PropagationEscape
from .virtual_disconnection import VirtualDisconnection
from .recurrence import recurrence_features
class TRACEDRELiteV2(nn.Module):
 def __init__(self,disable_side_features=False,disable_recruitment_dynamics=False,disable_propagation_escape=False,disable_virtual_disconnection=False,disable_cross_seizure_recurrence=False,enable_edge_correction=False):
  super().__init__();self.use_side=not disable_side_features;self.use_prop=not disable_propagation_escape;self.use_vdr=not disable_virtual_disconnection and self.use_prop;self.use_recurrence=not disable_cross_seizure_recurrence;self.encoder=RawWindowEncoder(use_side=self.use_side);self.recruit=RecruitmentDynamics(not disable_recruitment_dynamics);self.prop=PropagationEscape(self.use_prop,use_side_features=self.use_side);self.vdr=VirtualDisconnection(enabled=self.use_vdr,edge_correction=enable_edge_correction);self.seizure=nn.Sequential(nn.Linear(192,64),nn.LayerNorm(64),nn.GELU(),nn.Linear(64,32),nn.LayerNorm(32),nn.GELU());dim=64+(10 if self.use_prop else 0)+(5 if self.use_vdr else 0)+(9 if self.use_recurrence else 0);self.head=nn.Sequential(nn.Linear(dim,48),nn.LayerNorm(48),nn.GELU(),nn.Dropout(.2),nn.Linear(48,16),nn.GELU(),nn.Linear(16,1))
 def _seizure(self,s,device):
  w=s['windows'].to(device).float();side=s['side_features'].to(device);c,t=w.shape[:2];h=self.encoder(w.reshape(-1,1,500),side.reshape(-1,side.shape[-1]) if self.use_side else None).reshape(c,t,32);mask=s['window_mask'].to(device);phase=s['phase_ids'].to(device);times=s['relative_times_sec'].to(device);labels=s['channel_labels_nez'].to(device);r=self.recruit(h,mask,phase,times);parts=torch.cat([r['pre'],r['onset']-r['pre'],r['spread']-r['onset'],r['spread']-r['pre'],r['onset'],r['spread']],1);u=self.seizure(parts.mean(0));prop=self.prop(r,labels,side,mask)
  if prop is not None:
   if 'ez_descriptor' not in prop or 'ez_tau' not in prop or prop['ez_descriptor'].shape!=(199,):raise RuntimeError('invalid true-EZ propagation descriptor')
   if int(prop['selected'].sum())>12:raise RuntimeError('Top-Q10 selected more than 12 NEZ nodes')
  vdr=self.vdr(prop)
  if vdr is not None and vdr['n_nodes']>13:raise RuntimeError('VDR graph exceeds 13 nodes')
  detail={'seizure_id':s['seizure_id'],'representation':u,'prop':prop,'vdr':vdr}
  if prop is not None:detail['prop'].update({'names':[s['channel_names'][i] for i in prop['nez_indices'].tolist()]})
  return detail
 def forward(self,views):
  outputs=[];device=next(self.parameters()).device
  for view in views:
   seizures=[self._seizure(s,device) for s in view['seizures']];u=torch.stack([s['representation'] for s in seizures]);pieces=[u.mean(0),torch.logsumexp(u,0)-math.log(len(u))];prop_rows=[s['prop'] for s in seizures if s['prop'] is not None]
   if self.use_prop:pieces.append(torch.stack([p['stats'] for p in prop_rows]).mean(0) if prop_rows else u.new_zeros(10))
   if self.use_vdr:pieces.append(torch.stack([s['vdr']['stats'] for s in seizures if s['vdr'] is not None]).mean(0) if any(s['vdr'] is not None for s in seizures) else u.new_zeros(5))
   rec,rec_meta=recurrence_features(prop_rows,self.use_recurrence)
   if self.use_recurrence:pieces.append(rec)
   logit=self.head(torch.cat(pieces)).squeeze();n_windows=sum(int(s['windows'].shape[0]*s['windows'].shape[1]) for s in view['seizures']);outputs.append({'logit':logit,'seizures':seizures,'recurrence':rec,'recurrence_meta':rec_meta,'n_windows_encoded':n_windows})
  return outputs
 def parameter_count(self):return sum(p.numel() for p in self.parameters())
 def enabled_modules(self):return {'side_features':self.use_side,'side_features_used_by_raw_encoder':self.use_side,'side_features_used_by_propagation':self.prop.use_side_features,'recruitment_dynamics':self.recruit.enabled,'propagation_escape':self.use_prop,'virtual_disconnection':self.use_vdr,'true_ez_supernode':self.use_vdr,'cross_seizure_recurrence':self.use_recurrence,'edge_correction':getattr(self.vdr,'edge_correction',False)}
