import torch
from task1_clean_nez.model import CleanNEZModel
from task1_clean_nez.losses import CleanNEZLoss

def batch(b=2,s=3,t=5,c=7):
    return {"b0_features":torch.randn(b,s,t,c,36),"physics_features":torch.randn(b,s,t,c,12),"window_centers":torch.arange(t).float().expand(b,s,t),"window_mask":torch.ones(b,s,t,dtype=torch.bool),"seizure_channel_mask":torch.ones(b,s,c,dtype=torch.bool),"seizure_mask":torch.ones(b,s,dtype=torch.bool),"channel_mask":torch.ones(b,c,dtype=torch.bool),"labels_nez":torch.tensor([[1,1,1,1,0,0,0]]*b,dtype=torch.float),"labels_ez":torch.tensor([[0,0,0,0,1,1,1]]*b,dtype=torch.float)}

def test_forward_shapes_and_complement():
    b=batch(); o=CleanNEZModel()(b)
    assert o["score_nez_probability"].shape==(2,7)
    assert o["seizure_p_nez"].shape==(2,3,7)
    assert torch.allclose(o["score_nez_probability"]+o["candidate_risk"],torch.ones(2,7))
    assert torch.isfinite(CleanNEZLoss()(o,b)["loss"])

def test_padding_risk_zero_and_small_initial_fusion():
    b=batch(b=1); b["channel_mask"][0,-1]=False; b["seizure_channel_mask"][0,:,-1]=False; m=CleanNEZModel(); o=m(b)
    assert o["score_nez_probability"][0,-1]==0 and o["candidate_risk"][0,-1]==0
    assert all(0<float(o[k].detach())<.1 for k in ("a_anchor","a_early","a_recurrence"))
