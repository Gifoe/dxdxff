import torch,torch.nn.functional as F
def loss_fn(outputs,labels,rank_weight=0.):
 logits=torch.stack([x['logit'] for x in outputs]).float();y=torch.tensor(labels,device=logits.device,dtype=torch.float);bce=F.binary_cross_entropy_with_logits(logits,y);pos=logits[y==1];neg=logits[y==0];rank=F.softplus(-(pos[:,None]-neg[None,:])).mean() if len(pos) and len(neg) else logits.new_zeros(());return bce+rank_weight*rank,{'bce':float(bce.detach()),'rank':float(rank.detach())}
