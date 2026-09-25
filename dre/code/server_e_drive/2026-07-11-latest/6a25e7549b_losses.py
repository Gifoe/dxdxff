import torch,torch.nn.functional as F
def ssl_losses(pred,target,masked,p1,p2):
 l=F.smooth_l1_loss(pred[masked].float(),target.detach()[masked].float()) if masked.any() else pred.new_zeros(());c=1-F.cosine_similarity(p1.float(),p2.detach().float(),dim=-1).mean() if len(p1) else l.new_zeros(());v=torch.relu(1-p1.float().std(0,unbiased=False)).mean() if len(p1)>1 else l.new_zeros(());return l+.1*c+.01*v,{'mask':l,'consistency':c,'variance':v}
