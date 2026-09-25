from __future__ import annotations
import math
import torch
from torch import nn
import torch.nn.functional as F

def _patient_mean(values,mask):
    means=[]
    for v,m in zip(values,mask):
        if m.any(): means.append(v[m].mean())
    return torch.stack(means).mean() if means else values.sum()*0

class CleanNEZLoss(nn.Module):
    def __init__(self,lambda_mil=.30,lambda_anchor=.03,lambda_consistency=.02,ez_core_fraction=.20,hard_nez_multiplier=1.,mil_margin=.10,anchor_margin=1.,anchor_ez_top_fraction=.20,anchor_gamma=.5):
        super().__init__(); self.__dict__.update(locals()); del self.__dict__["self"]
    def forward(self,out,batch):
        valid=out["effective_channel_mask"]; nez=valid&(batch["labels_nez"]>.5); ez=valid&(batch["labels_ez"]>.5)
        clean=_patient_mean(F.softplus(-out["final_nez_logit"]),nez)
        mils=[]; margins=[]
        for i in range(valid.shape[0]):
            er=out["candidate_risk"][i,ez[i]]; nr=out["candidate_risk"][i,nez[i]]; ed=out["anchor_distance"][i,ez[i]]
            if er.numel() and nr.numel():
                ke=max(1,math.ceil(self.ez_core_fraction*er.numel())); kn=min(nr.numel(),max(1,math.ceil(self.hard_nez_multiplier*ke)))
                mils.append(F.softplus(self.mil_margin+torch.topk(nr,kn).values.mean()-torch.topk(er,ke).values.mean()))
            if ed.numel():
                k=max(1,math.ceil(self.anchor_ez_top_fraction*ed.numel())); margins.append(F.relu(self.anchor_margin-torch.topk(ed,k).values).square().mean())
        mil=torch.stack(mils).mean() if mils else clean*0
        compact=_patient_mean(out["anchor_distance"],nez); weak=torch.stack(margins).mean() if margins else clean*0; anchor=compact+self.anchor_gamma*weak
        valid_sz=batch["seizure_mask"][:,:,None]&batch["seizure_channel_mask"]&valid[:,None,:]&nez[:,None,:]; count=valid_sz.sum(1); p=out["seizure_p_nez"]
        mean=(p*valid_sz).sum(1)/count.clamp_min(1); var=(((p-mean[:,None])**2)*valid_sz).sum(1)/count.clamp_min(1); consistency=var[count>=2].mean() if (count>=2).any() else clean*0
        total=clean+self.lambda_mil*mil+self.lambda_anchor*anchor+self.lambda_consistency*consistency
        return {"loss":total,"loss_clean_nez":clean,"loss_weak_ez_mil":mil,"loss_anchor":anchor,"loss_nez_consistency":consistency}
__all__=["CleanNEZLoss"]
