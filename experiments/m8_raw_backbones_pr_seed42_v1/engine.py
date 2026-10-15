"""Full-patient differentiable aggregation without retaining all backbone graphs.

Exact gradient caching: stochastic chunks replay their RNG states, and original
forward BN buffers are restored after replay so there is only one state update.
"""
from contextlib import nullcontext
import numpy as np
import torch
from models.backbones import aggregate


def rng():return torch.get_rng_state(),torch.cuda.get_rng_state()
def restore(q):torch.set_rng_state(q[0].cpu());torch.cuda.set_rng_state(q[1].cpu())
def amp(enabled=True):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled)


def layout(raw,mask,multi,chunk):
    S,C,W,L=raw.shape;assert L==500 and mask.shape==(S,C,W)
    if multi:
        r=raw.permute(0,2,1,3).reshape(S*W,C,L);m=mask.permute(0,2,1).reshape(S*W,C)
        # Windows without any valid channel cannot affect any aggregation.
        ix=torch.nonzero(m.any(1),as_tuple=False).flatten()
    else:
        r=raw[mask];m=None;ix=torch.arange(len(r),device=r.device)
    return r,m,[ix[i:i+chunk] for i in range(0,len(ix),chunk)]


def dense(encoded,mask,multi,indices=None):
    S,C,W=mask.shape;D=encoded.shape[-1]
    if multi:
        q=encoded.new_zeros(S*W,C,D)
        q[indices]=encoded
        return q.reshape(S,W,C,D).permute(0,2,1,3)
    q=encoded.new_zeros(S,C,W,D);q[mask]=encoded;return q


def forward(model,raw,mask,chunk,train=False,target=None,pos_weight=None):
    r,m,chunks=layout(raw,mask,model.multichannel,chunk)
    states=[];outputs=[]
    with torch.no_grad():
        for ix in chunks:
            if train:states.append(rng())
            with amp(train):h=model.encode(r[ix],None if m is None else m[ix])
            assert torch.isfinite(h).all();outputs.append(h.float())
    emb=torch.cat(outputs).detach().requires_grad_(train)
    indices=torch.cat(chunks) if model.multichannel else None
    channel=aggregate(dense(emb,mask,model.multichannel,indices),mask)
    with amp(train):logit=model.classify(channel)
    logit=logit.float();assert torch.isfinite(logit).all()
    if not train:return logit.detach(),channel.detach()
    assert target is not None and pos_weight is not None
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,target,pos_weight=pos_weight)
    assert torch.isfinite(loss)
    loss.backward();grad=emb.grad.detach();assert torch.isfinite(grad).all()
    # Save post-first-forward/head RNG and BN buffers. Replay must not count twice.
    after=rng();buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    offset=0
    try:
        for ix,state in zip(chunks,states):
            restore(state)
            with amp():h=model.encode(r[ix],None if m is None else m[ix])
            h.float().backward(grad[offset:offset+len(ix)])
            offset+=len(ix)
    finally:
        with torch.no_grad():
            for k,v in model.named_buffers():v.copy_(buffers[k])
        restore(after)
    return float(loss.detach()),logit.detach()
