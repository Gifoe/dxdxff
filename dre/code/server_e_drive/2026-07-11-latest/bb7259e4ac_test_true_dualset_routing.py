import torch,pytest
from neuroez_c.task2.true_dualset.model import TrueEZNEZDualSet
def seizure(labels=(0,1,0,1),t=3):
 c=len(labels);return {'seizure_id':'s','windows':torch.randn(c,t,500),'side_features':torch.randn(c,t,12),'window_mask':torch.ones(c,t,dtype=torch.bool),'phase_ids':torch.tensor([0,1,2]),'channel_labels_nez':torch.tensor(labels),'channel_names':[str(i) for i in range(c)]}
def view(labels=(0,1,0,1)):return {'patient_key':'p','center':'x','outcome_success':1,'view_id':0,'seizures':[seizure(labels)]}
def test_true_label_routing_and_label_sensitivity():
 m=TrueEZNEZDualSet().eval();a=m([view()])[0];b=m([view((1,0,0,1))])[0];assert a['seizure_audit'][0]['n_ez']==2 and a['seizure_audit'][0]['n_nez']==2 and not torch.allclose(a['logit'],b['logit'])
def test_empty_ez_or_nez_is_error():
 m=TrueEZNEZDualSet()
 with pytest.raises(ValueError):m([view((1,1,1,1))])
 with pytest.raises(ValueError):m([view((0,0,0,0))])
