import torch
from neuroez_c.task2.trace_aux_stable.model import TraceAuxStable
from neuroez_c.task2.trace_aux_stable.losses import channel_loss
from neuroez_c.task2.trace_aux_stable.residual_pooling import sparsemax
def seizure(labels):
 c=len(labels);return {'seizure_id':'s','windows':torch.randn(c,3,500),'side_features':torch.randn(c,3,12),'window_mask':torch.ones(c,3,dtype=torch.bool),'phase_ids':torch.tensor([0,1,2]),'channel_labels_nez':torch.tensor(labels),'channel_names':[str(i) for i in range(c)]}
def test_routing_sparsemax_and_gradients():
 m=TraceAuxStable().eval();o=m([{'patient_key':'p','center':'x','outcome_success':1,'view_id':0,'seizures':[seizure([0,1,0,1])]}])[0];loss=channel_loss([o['channel_rows']])+o['logit'];loss.backward();assert o['seizure_audit'][0]['n_ez']==2 and o['seizure_audit'][0]['n_nez']==2 and any(p.grad is not None for p in m.window_encoder.parameters()) and any(p.grad is not None for p in m.channel_aux.parameters())
def test_sparsemax_properties_and_patient_balanced_loss():
 x=torch.tensor([1.,2.,-1.],requires_grad=True);w=sparsemax(x);w.sum().backward();assert w.ge(0).all() and torch.allclose(w.sum(),torch.tensor(1.)) and x.grad is not None
 a=(torch.zeros(21),torch.tensor([0]+[1]*20));b=(torch.zeros(12),torch.tensor([0]*10+[1]*2));assert torch.allclose(channel_loss([[a,b]]),torch.tensor(.693147, dtype=torch.float32),atol=1e-4)
