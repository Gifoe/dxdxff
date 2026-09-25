import torch
from neuroez_c.task2.virtual_target_network import compute_virtual_target_network,virtual_residual_graph


def test_virtual_removal_zeroes_target_and_spectral_ratio():
    graph=torch.tensor([[0.,1.,.2],[1.,0.,.5],[.2,.5,0.]])
    removed=virtual_residual_graph(graph,torch.tensor([True,False,False])); assert removed[0].sum()==0 and removed[:,0].sum()==0
    adjacency=graph.reshape(1,1,1,3,3).expand(1,1,3,3,3).clone(); target=torch.tensor([[True,False,False]]); risk=torch.tensor([[.9,.8,.7]])
    output=compute_virtual_target_network(adjacency,target,risk,risk,torch.ones(1,1,3,3,dtype=torch.bool),torch.ones(1,1,3,dtype=torch.bool))
    expected=output["residual_spectral_radius_spread"]/output["original_spectral_radius_spread"]
    assert torch.allclose(output["residual_spectral_radius_ratio_spread"],expected,atol=1e-6)
