"""Label-free, mask-aware references; every global statistic is fitted on FIT."""
import torch

def fit_global(h):
    x=torch.as_tensor(h,dtype=torch.float64)
    assert x.ndim==2 and len(x)>0 and torch.isfinite(x).all()
    mean=x.mean(0);sd=x.std(0,unbiased=False)
    unit=torch.where(sd>0,sd,torch.ones_like(sd))
    return dict(mean=mean.float(),sd=sd.float(),unit=unit.float(),epsilon=(1e-6*unit).float())

def percentile_rank(x):
    """Average-tie ranks, mapped to [-1,1]; indices carry no gradients."""
    assert x.ndim==2 and len(x)>0 and torch.isfinite(x).all()
    with torch.no_grad():
        result=torch.empty_like(x)
        for k in range(x.shape[1]):
            values,order=torch.sort(x[:,k],stable=True)
            _,counts=torch.unique_consecutive(values,return_counts=True)
            ends=counts.cumsum(0);starts=ends-counts
            average=(starts+ends-1).to(x.dtype)/2+1
            ranks=average.repeat_interleave(counts)
            result[order,k]=2*(ranks-.5)/len(x)-1
        return result

def views(h,mean,sd,unit,epsilon,mode='pr2'):
    assert mode in ['pr2','abs'] and torch.isfinite(h).all()
    absolute=(h-mean)/unit
    alpha=len(h)/(len(h)+16.)
    if mode=='abs':
        v=torch.cat([absolute,absolute.tanh()],1)
        return absolute,v,dict(alpha=alpha,reference_scale=unit.detach(),relative_abs=(absolute.detach().abs()))
    with torch.no_grad():
        median=h.detach().quantile(.5,dim=0)
        mad=(h.detach()-median).abs().quantile(.5,dim=0)
        location=(1-alpha)*mean+alpha*median
        scale=((1-alpha)*sd.square()+alpha*(1.4826*mad).square()+epsilon.square()).sqrt()
        assert torch.isfinite(scale).all() and (scale>0).all()
    relative=(h-location)/scale
    v=torch.cat([relative.tanh(),percentile_rank(h)],1)
    assert torch.isfinite(v).all()
    return absolute,v,dict(alpha=alpha,reference_scale=scale,relative_abs=relative.detach().abs())
