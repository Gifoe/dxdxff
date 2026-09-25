import torch
from neuroez_c.task2.seizure_nez_prototype import compute_seizure_nez_stability


def test_persistent_and_worstcase_have_declared_quantiles():
    q=torch.tensor([[[.1,.9],[.5,.8],[.9,.7]]]); valid=torch.ones_like(q,dtype=torch.bool); target=torch.tensor([[True,False]]); channels=torch.ones_like(target)
    output=compute_seizure_nez_stability(q,valid,target,channels)
    assert torch.allclose(output["persistent_abnormality"][0,0],torch.quantile(1-q[0,:,0],.1))
    assert torch.allclose(output["worstcase_abnormality"][0,0],1-torch.quantile(q[0,:,0],.1))
