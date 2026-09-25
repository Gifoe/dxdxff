import torch
from neuroez_c.task2.residual_set_encoder import ResidualSetEncoder


def test_channel_permutation_invariant_set_output():
    torch.manual_seed(2); model=ResidualSetEncoder(6).eval(); embedding=torch.randn(1,7,6); scalars=torch.randn(1,7,13); target=torch.tensor([[1,1,0,0,0,0,0]],dtype=torch.bool); valid=torch.ones_like(target); risk=torch.rand(1,7)
    first=model(embedding,scalars,target,valid,risk)["set_embedding"]; order=torch.tensor([4,2,6,0,5,1,3])
    second=model(embedding[:,order],scalars[:,order],target[:,order],valid[:,order],risk[:,order])["set_embedding"]
    assert torch.allclose(first,second,atol=1e-6)
