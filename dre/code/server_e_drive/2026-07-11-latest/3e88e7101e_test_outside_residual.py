import torch
from neuroez_c.task2.outside_nez_residual import compute_outside_nez_residual


def test_outside_residual_excludes_target():
    q=torch.tensor([[.1,.8,.9]]); risk=1-q; target=torch.tensor([[True,False,False]]); valid=torch.ones_like(target)
    output=compute_outside_nez_residual(q,risk,target,valid)
    assert torch.allclose(output["outside_residual_max"],torch.tensor([.2]))
    assert output["residual_channel"][0,0].item()==0
