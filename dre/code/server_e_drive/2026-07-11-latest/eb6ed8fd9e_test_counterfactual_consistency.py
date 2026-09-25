import torch
from neuroez_c.task2.nvr_features import make_counterfactual_targets
from neuroez_c.task2.nvr_loss import nvr_loss


def test_counterfactual_target_edits_and_hinge_direction():
    value={"target":torch.tensor([[True,True,False,False]]),"channel_mask":torch.ones(1,4,dtype=torch.bool),"reliable_abnormality":torch.tensor([[.8,.7,.9,.2]])}
    plus,minus,valid=make_counterfactual_targets(value); assert plus.sum()==3 and minus.sum()==1 and valid.item()
    parameter=torch.nn.Parameter(torch.tensor(0.)); good=nvr_loss(torch.tensor([0.]),torch.tensor([1.]),["A"],[parameter],plus_logit=torch.tensor([1.]),minus_logit=torch.tensor([-1.]),counterfactual_valid=torch.tensor([True]),robust=True)
    assert good["counterfactual_consistency"].item()==0
