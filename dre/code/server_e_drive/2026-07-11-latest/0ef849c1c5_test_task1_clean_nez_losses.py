import torch
from task1_clean_nez.losses import CleanNEZLoss

def data(ez_risk):
    risk=torch.tensor([[.1,.2,*ez_risk]]); logits=torch.logit(1-risk); z=torch.zeros_like(risk)
    o={"candidate_risk":risk,"final_nez_logit":logits,"anchor_distance":torch.tensor([[.1,.2,2.,.2]]),"seizure_p_nez":torch.ones(1,1,4)*.8,"effective_channel_mask":torch.ones(1,4,dtype=torch.bool)}
    b={"channel_mask":torch.ones(1,4,dtype=torch.bool),"labels_nez":torch.tensor([[1.,1.,0.,0.]]),"labels_ez":torch.tensor([[0.,0.,1.,1.]]),"seizure_mask":torch.ones(1,1,dtype=torch.bool),"seizure_channel_mask":torch.ones(1,1,4,dtype=torch.bool)}
    return o,b

def test_mil_improves_when_one_ez_is_high_risk():
    fn=CleanNEZLoss(); a=fn(*data([.1,.1]))["loss_weak_ez_mil"]; b=fn(*data([.9,.1]))["loss_weak_ez_mil"]
    assert b<a

def test_clean_loss_ignores_ez_logits():
    fn=CleanNEZLoss(); o,b=data([.5,.5]); x=fn(o,b)["loss_clean_nez"]; o["final_nez_logit"][0,2:]=-100; assert torch.equal(x,fn(o,b)["loss_clean_nez"])
