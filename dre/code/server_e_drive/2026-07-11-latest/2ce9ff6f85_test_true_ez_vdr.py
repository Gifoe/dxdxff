import torch
from neuroez_c.task2.trace_dre_lite.propagation import PropagationEscape
from neuroez_c.task2.trace_dre_lite.virtual_disconnection import VirtualDisconnection
def inputs(ez_shift=0.,nez_shift=0.,n_nez=5):
 c=2+n_nez;t=12;ez=torch.arange(2);nez=torch.arange(2,c);pre=torch.randn(c,32);on=torch.randn(c,32);spread=torch.randn(c,32);pre[ez]+=ez_shift;on[ez]+=ez_shift;spread[ez]+=ez_shift;pre[nez]+=nez_shift;on[nez]+=nez_shift;spread[nez]+=nez_shift;curve=torch.randn(c,t);tau=torch.tensor([2.,4.]+[5.+i for i in range(n_nez)]);r={'pre':pre,'onset':on,'spread':spread,'curve':curve,'tau':tau};labels=torch.tensor([0,0]+[1]*n_nez);side=torch.randn(c,t,12);mask=torch.ones(c,t,dtype=torch.bool);return r,labels,side,mask
def test_true_ez_descriptor_shape_tau_and_counterfactual():
 torch.manual_seed(1);m=PropagationEscape();base=inputs();a=m(*base);r,l,s,mask=base;r2={k:(v.clone() if torch.is_tensor(v) else v) for k,v in r.items()};r2['pre'][:2]+=3;r2['onset'][:2]+=3;r2['spread'][:2]+=3;b=m(r2,l,s,mask);assert a['ez_descriptor'].shape==(199,) and a['nez_descriptor'].shape[1]==199 and torch.allclose(a['ez_tau'],torch.tensor(3.));assert not torch.allclose(a['ez_descriptor'],b['ez_descriptor']);r3={k:v.clone() for k,v in r.items()};r3['pre'][2:]+=5;r3['onset'][2:]+=5;r3['spread'][2:]+=5;c=m(r3,l,s,mask);assert torch.allclose(a['ez_descriptor'][:128],c['ez_descriptor'][:128]) and all(torch.isfinite(x).all() for x in (a['ez_descriptor'],a['nez_descriptor'],a['ez_tau']))
def manual_prop(n=2):
 d=torch.ones(n,199);return {'ez_descriptor':torch.ones(199),'nez_descriptor':d,'descriptor':d,'selected':torch.ones(n,dtype=torch.bool),'ez_tau':torch.tensor(1.),'nez_tau':torch.arange(n).float()*2+3,'tau':torch.arange(n).float()*2+3,'nez_indices':torch.arange(n)+2,'ez_self_similarity':torch.tensor(.8)}
def test_true_ez_node_tau_composition_and_direction():
 v=VirtualDisconnection()(manual_prop(2));a=v['adjacency'];assert v['ez_tau']==3-2 and v['n_nodes']==3 and v['n_nez_nodes']==2 and v['uses_true_ez_supernode'] is True and torch.diag(a).eq(0).all();assert a[0,1]>a[1,0] and a[0,2]>a[2,0]
def test_small_graphs_finite_and_at_most_13_nodes():
 for n in (1,2,3,12):
  v=VirtualDisconnection()(manual_prop(n));assert v['n_nodes']==n+1 and v['n_nodes']<=13 and torch.isfinite(v['stats']).all()
def test_vdr_backward_base_and_edge_correction():
 for correction in (False,True):
  m=VirtualDisconnection(edge_correction=correction);v=m(manual_prop(4));v['stats'].sum().backward();assert torch.isfinite(m.alpha.grad) and torch.isfinite(m.beta.grad) and all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.node.parameters());
  if correction:assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.edge.parameters())
