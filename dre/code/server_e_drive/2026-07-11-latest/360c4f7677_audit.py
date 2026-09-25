from __future__ import annotations
import torch
from . import MODEL_VERSION,CACHE_VERSION
def verify_patient(patient):
 key=patient.get('patient_key','<unknown>');out=patient.get('outcome_success')
 if out not in (0,1):raise ValueError(f'{key}: outcome_success must be 0 or 1')
 if not patient.get('seizures'):raise ValueError(f'{key}: no valid seizure')
 for s in patient['seizures']:
  sid=s.get('seizure_id','<unknown>');w=s['windows'];side=s['side_features'];mask=s['window_mask'].bool();labels=s['channel_labels_nez'];ph=s['phase_ids']
  if w.ndim!=3 or w.shape[-1]!=500:raise ValueError(f'{key}/{sid}: windows must be [C,T,500]')
  c,t=w.shape[:2]
  if side.shape!=(c,t,12) or mask.shape!=(c,t) or labels.shape!=(c,) or ph.shape!=(t,):raise ValueError(f'{key}/{sid}: channel/side/mask/label shapes disagree')
  if not torch.isfinite(w[mask]).all() or not torch.isfinite(side[mask]).all():raise ValueError(f'{key}/{sid}: non-finite valid windows')
  if not torch.isin(labels,torch.tensor([0,1])).all():raise ValueError(f'{key}/{sid}: labels must only be 0/1')
  if not (labels==0).any() or not (labels==1).any():raise ValueError(f'{key}/{sid}: requires true EZ and NEZ')
  for phase in range(3):
   if not ((ph==phase)[None,:]&mask).any():raise ValueError(f'{key}/{sid}: phase {phase} has no valid window')
 return True
def model_audit(model):
 count=model.parameter_count()
 if count>=300000:raise RuntimeError(f'parameter count {count} is not <300000')
 return {'model_version':MODEL_VERSION,'cache_version':CACHE_VERSION,'parameter_count':count,'under_300k':True,**model.module_parameter_counts(),'enabled_modules':['raw_window_encoder','side_feature_encoder','phase_temporal_encoder','true_ez_nez_routing','dual_set_statistics','bidirectional_cross_set_attention','cross_seizure_mean_std_max','patient_outcome_head'],'disabled_modules':['task1_localization_head','task1_loss','p2_checkpoint','predicted_ez','recruitment_time','propagation_escape','top_q10','virtual_disconnection','graph_neural_network','clinical_branch'],'label_direction':{'NEZ':1,'EZ':0,'success':1,'failure':0},'uses_counts_as_model_input':False,'uses_task1_checkpoint':False,'uses_task1_loss':False}
