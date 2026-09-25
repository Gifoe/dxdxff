from __future__ import annotations
import torch
import torch.nn.functional as F
def rank_weight(epoch):
    """Stable V1 schedule: BCE-only warm-up, then weak linear ranking."""
    epoch = int(epoch)
    if epoch <= 5:
        return 0.0
    return min(0.15 * (epoch - 5) / 15.0, 0.15)
def pairwise_logistic(logits,labels):
    labels=torch.as_tensor(labels,device=logits.device);pos=logits[labels==1];neg=logits[labels==0]
    return F.softplus(-(pos[:,None]-neg[None,:])).mean() if len(pos) and len(neg) else logits.new_zeros(())
def trace_loss(outputs,labels,epoch=1,*_):
    logits=torch.stack([x['logit'] for x in outputs]);y=torch.as_tensor(labels,dtype=logits.dtype,device=logits.device);bce=F.binary_cross_entropy_with_logits(logits,y);rank=pairwise_logistic(logits,y);w=rank_weight(epoch);return bce+w*rank,{'bce':bce.detach(),'rank_raw':rank.detach(),'rank_weighted':(w*rank).detach(),'rank_weight':w}
