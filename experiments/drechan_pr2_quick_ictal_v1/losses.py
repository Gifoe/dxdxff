"""Fixed complete-patient objectives; calibration never enters pairwise loss."""
import torch
from torch.nn import functional as F

def pairwise(logit,y,chunk=256):
    ez=logit[y.bool()];nez=logit[~y.bool()]
    if not len(ez) or not len(nez):return logit.sum()*0
    total=logit.new_zeros(())
    for x in ez.split(chunk):total=total+F.softplus(-(x[:,None]-nez[None,:])/.25).sum()
    return total/(len(ez)*len(nez))

def objective(out,y,nez_weight,stage='svm'):
    y=y.to(out['final'].dtype)
    # Historical model weights NEZ positive by nEZ/nNEZ; use exact EZ-logit equivalent.
    weights=torch.where(y.bool(),torch.ones_like(y),torch.full_like(y,float(nez_weight)))
    bce=F.binary_cross_entropy_with_logits(out['final'],y,weight=weights)
    if stage=='drechan':
        # Preserve the original NEZ-logit operations, not just algebraic equivalence.
        nez=1-y;p=(-out['final']).sigmoid()
        a=2*(p*nez).sum()/(p.sum()+nez.sum()).clamp_min(1e-8)
        b=2*((1-p)*(1-nez)).sum()/((1-p).sum()+(1-nez).sum()).clamp_min(1e-8)
        original_bce=F.binary_cross_entropy_with_logits(-out['final'],nez,pos_weight=y.new_tensor(nez_weight))
        base=.75*original_bce+.25*(1-.5*(a+b))
    else:base=bce
    return base+.15*pairwise(out['rank'],y)+.001*out['delta_rank'].square().mean()
