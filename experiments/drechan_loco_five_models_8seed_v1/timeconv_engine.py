"""Original two-pass TimeConv patient engine; only import path is namespaced."""
import torch
from common import TC
aggregate=TC.aggregate

def rng():return torch.get_rng_state(),torch.cuda.get_rng_state()
def restore(q):torch.set_rng_state(q[0].cpu());torch.cuda.set_rng_state(q[1].cpu())
def amp(enabled=True):return torch.autocast('cuda',dtype=torch.bfloat16,enabled=enabled)

def layout(raw,mask,multi,chunk):
    assert not multi
    r=raw[mask];ix=torch.arange(len(r),device=r.device)
    return r,None,[ix[i:i+chunk] for i in range(0,len(ix),chunk)]

def dense(encoded,mask):
    q=encoded.new_zeros(*mask.shape,encoded.shape[-1]);q[mask]=encoded;return q

def forward(model,raw,mask,chunk,train=False,target=None,pos_weight=None):
    r,_,chunks=layout(raw,mask,False,chunk);states=[];outputs=[]
    with torch.no_grad():
        for ix in chunks:
            if train:states.append(rng())
            with amp(train):h=model.encode(r[ix])
            assert torch.isfinite(h).all();outputs.append(h.float())
    emb=torch.cat(outputs).detach().requires_grad_(train)
    channel=aggregate(dense(emb,mask),mask)
    with amp(train):logit=model.classify(channel)
    logit=logit.float();assert torch.isfinite(logit).all()
    if not train:return logit.detach(),channel.detach()
    loss=torch.nn.functional.binary_cross_entropy_with_logits(logit,target,pos_weight=pos_weight)
    assert torch.isfinite(loss);loss.backward();grad=emb.grad.detach();assert torch.isfinite(grad).all()
    after=rng();buffers={k:v.detach().clone() for k,v in model.named_buffers()};offset=0
    try:
        for ix,state in zip(chunks,states):
            restore(state)
            with amp():h=model.encode(r[ix])
            h.float().backward(grad[offset:offset+len(ix)]);offset+=len(ix)
    finally:
        with torch.no_grad():
            for k,v in model.named_buffers():v.copy_(buffers[k])
        restore(after)
    return float(loss.detach()),logit.detach()
