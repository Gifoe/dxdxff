"""Full-window,full-patient encoder gradient caching; no sampled channels/windows."""
import torch
from model import pooling

CHUNK=1024


def rng():return torch.get_rng_state(),torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []


def restore(state):
    torch.set_rng_state(state[0].cpu())
    if state[1]:torch.cuda.set_rng_state_all([v.cpu() for v in state[1]])


def scatter(emb,mask):
    dense=emb.new_zeros((*mask.shape,24));dense[mask]=emb;return dense


def direct(model,raw,mask,times,y=None,pos=None,chunk=CHUNK):
    flat=model.normalized_windows(raw,mask,times)
    h=torch.cat([model.cnn(q[:,None]).flatten(1) for q in flat.split(chunk)])
    z=model.aggregate(scatter(h,mask),mask,times);logit=model.classify(z)
    return (pooling.objective(logit,y,pos,'ez_rank'),logit,z) if y is not None else (logit,z)


def cached(model,raw,mask,times,y=None,pos=None,chunk=CHUNK):
    train=y is not None;flat=model.normalized_windows(raw,mask,times);parts=[];states=[]
    with torch.no_grad():
        for q in flat.split(chunk):
            if train:states.append(rng())
            parts.append(model.cnn(q[:,None]).flatten(1))
    h=torch.cat(parts).detach().requires_grad_(train);assert torch.isfinite(h).all()
    z=model.aggregate(scatter(h,mask),mask,times);logit=model.classify(z)
    assert torch.isfinite(logit).all() and torch.isfinite(z).all()
    if not train:return logit.detach(),z.detach()
    loss=pooling.objective(logit,y,pos,'ez_rank');assert torch.isfinite(loss);loss.backward()
    dh=h.grad.detach();assert torch.isfinite(dh).all();after=rng()
    buffers={k:v.detach().clone() for k,v in model.named_buffers()};offset=0
    try:
        for q,state in zip(flat.split(chunk),states):
            restore(state);emb=model.cnn(q[:,None]).flatten(1)
            emb.backward(dh[offset:offset+len(q)]);offset+=len(q)
    finally:
        with torch.no_grad():
            for k,v in model.named_buffers():v.copy_(buffers[k])
        restore(after)
    return float(loss.detach().cpu()),logit.detach(),z.detach()
