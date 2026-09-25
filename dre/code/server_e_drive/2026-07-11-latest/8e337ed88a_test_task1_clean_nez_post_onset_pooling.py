import torch
from task1_clean_nez.temporal import PostOnsetTemporalEncoder
def test_post_onset_early_selection_and_fallback():
    e=PostOnsetTemporalEncoder(1,.5); x=torch.arange(6.).reshape(1,1,6,1,1); sm=torch.ones(1,1,1,dtype=torch.bool); wm=torch.ones(1,1,6,dtype=torch.bool); centers=torch.tensor([[[-10.,-5.,0.,1.,2.,3.]]]); _,chosen,diag=e(x,sm,wm,centers)
    assert torch.equal(torch.where(chosen[0,0,:,0])[0],torch.tensor([2,3])) and diag["post_onset_fallback_mask"].sum()==0
    centers.fill_(-1); wm[0,0,-1]=False; _,chosen,diag=e(x,sm,wm,centers); assert diag["post_onset_fallback_mask"].sum()==1 and not chosen[0,0,-1,0]
