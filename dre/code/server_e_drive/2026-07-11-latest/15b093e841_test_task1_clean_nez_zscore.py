import torch
from task1_clean_nez.anchor import masked_seizure_channel_zscore
def test_seizures_are_independent_across_channel_axis():
    x=torch.tensor([[[1.,2.,3.],[101.,102.,103.]]]); m=torch.ones_like(x,dtype=torch.bool); z=masked_seizure_channel_zscore(x,m)
    assert torch.allclose(z[0,0],z[0,1]); x[0,1]+=1000; assert torch.allclose(z[0,0],masked_seizure_channel_zscore(x,m)[0,0]); assert not torch.all(z[0,0]==0)
def test_single_valid_channel_is_zero():
    x=torch.tensor([[[2.,9.]]]); m=torch.tensor([[[True,False]]]); z=masked_seizure_channel_zscore(x,m); assert torch.equal(z,torch.zeros_like(z))
