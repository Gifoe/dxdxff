import json,torch,numpy as np
from pathlib import Path
from neuroez_c.task2.trace_dre_lite.cache_builder import make_signature,signature_hash,atomic_torch
from neuroez_c.task2.trace_dre_lite.cached_dataset import make_view
from neuroez_c.task2.trace_dre_lite.schema import ViewConfig
from neuroez_c.task2.trace_dre_lite.propagation import PropagationEscape
from neuroez_c.task2.trace_dre_lite.virtual_disconnection import VirtualDisconnection
from neuroez_c.task2.trace_dre_lite.recurrence import recurrence_features
from neuroez_c.task2.trace_dre_lite.model import TRACEDRELiteV2
def seizure(name='s',c=8,t=12):
 return {'seizure_id':name,'windows':torch.randn(c,t,500).clamp(-8,8).half(),'window_mask':torch.ones(c,t,dtype=torch.bool),'phase_ids':torch.tensor([0]*4+[1]*4+[2]*4),'relative_times_sec':torch.arange(t).float()-4,'side_features':torch.randn(c,t,12),'channel_names':[f'c{i}' for i in range(c)],'channel_labels_nez':torch.tensor([0,0]+[1]*(c-2)),'channel_quality':torch.ones(c,4)}
def patient():return {'patient_key':'lzu:p','center':'lzu','outcome_success':1,'seizures':[seizure('s1'),seizure('s2')]}
def test_signature_repeatable_and_changes(tmp_path):
 a=tmp_path/'a';b=tmp_path/'b';o=tmp_path/'outcome';a.write_text('x');b.write_text('y');o.write_text('z');s=make_signature(a,b,o);assert signature_hash(s)==signature_hash(make_signature(a,b,o));o.write_text('changed');assert signature_hash(s)!=signature_hash(make_signature(a,b,o))
def test_atomic_shard_and_deterministic_views(tmp_path):
 p=tmp_path/'x.pt';atomic_torch(p,patient());assert p.exists() and not p.with_suffix('.pt.tmp').exists();a=make_view(patient(),42,0,ViewConfig());b=make_view(patient(),42,0,ViewConfig());assert [x['seizure_id'] for x in a['seizures']]==[x['seizure_id'] for x in b['seizures']];assert make_view(patient(),42,1,ViewConfig())['view_id']==1
def test_propagation_delay_positive_and_virtual_backward():
 c,t=2,10;curve=torch.zeros(c,t);curve[0,2]=5;curve[1,4]=5;r={'curve':curve,'tau':torch.tensor([2.,4.]),'pre':torch.randn(c,32),'onset':torch.randn(c,32),'spread':torch.randn(c,32)};side=torch.randn(c,t,12);mask=torch.ones(c,t,dtype=torch.bool);prop=PropagationEscape()(r,torch.tensor([0,1]),side,mask);assert prop['delay'][0]>0;v=VirtualDisconnection()(prop);assert v['n_nodes']<=13 and torch.isfinite(v['stats']).all() and torch.diag(v['adjacency']).eq(0).all();v['stats'].sum().backward()
def test_propagation_quantile_is_float32_under_bf16_autocast():
 c,t=5,10;r={'curve':torch.randn(c,t),'tau':torch.arange(c).float(),'pre':torch.randn(c,32),'onset':torch.randn(c,32),'spread':torch.randn(c,32)};side=torch.randn(c,t,12);mask=torch.ones(c,t,dtype=torch.bool)
 with torch.autocast('cpu',dtype=torch.bfloat16):prop=PropagationEscape()(r,torch.tensor([0,1,1,1,1]),side,mask)
 assert prop['logits'].dtype==torch.float32 and prop['probability'].dtype==torch.float32 and prop['stats'].dtype==torch.float32 and torch.isfinite(prop['stats']).all()
def test_recurrence_increases_for_repeated_top_channel():
 s=[{'names':['a','b'],'logits':torch.tensor([3.,0.]),'tau':torch.tensor([2.,4.]),'selected':torch.tensor([True,False])} for _ in range(2)];f,m=recurrence_features(s);assert m['recurrence_present']==1 and f[0]>0 and m['topq10_jaccard']==1
def test_single_seizure_and_small_nez_forward_backward(monkeypatch):
 import scipy.signal
 monkeypatch.setattr(scipy.signal,'resample_poly',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('hot path')));v={'patient_key':'p','center':'x','outcome_success':1,'view_id':0,'seizures':[seizure(c=4)]};m=TRACEDRELiteV2();o=m([v])[0];o['logit'].backward();assert torch.isfinite(o['logit']) and m.parameter_count()<250000
def test_all_ablation_flags_forward():
 v={'patient_key':'p','center':'x','outcome_success':0,'view_id':0,'seizures':[seizure(c=4)]}
 for flag in ('disable_side_features','disable_recruitment_dynamics','disable_propagation_escape','disable_virtual_disconnection','disable_cross_seizure_recurrence'):
  m=TRACEDRELiteV2(**{flag:True});assert torch.isfinite(m([v])[0]['logit'])
