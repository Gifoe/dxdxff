import torch
from neuroez_c.task2.true_dualset.model import TrueEZNEZDualSet
from test_true_dualset_routing import seizure
def test_group_permutation_invariant_eval():
 torch.manual_seed(2);s=seizure((0,0,1,1,1));perm=torch.tensor([1,0,4,2,3]);q={k:(v[perm] if isinstance(v,torch.Tensor) and v.ndim and v.shape[0]==5 else v) for k,v in s.items()};q['channel_names']=[s['channel_names'][i] for i in perm];m=TrueEZNEZDualSet().eval();a=m([{'patient_key':'p','center':'x','outcome_success':1,'view_id':0,'seizures':[s]}])[0]['logit'];b=m([{'patient_key':'p','center':'x','outcome_success':1,'view_id':0,'seizures':[q]}])[0]['logit'];assert torch.allclose(a,b,atol=1e-5)
