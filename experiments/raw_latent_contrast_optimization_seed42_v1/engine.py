"""Complete-patient, two-pass CNN gradients; pooling and PatientZ are not detached."""
import torch
from model import normalize_windows,objective


def rng():
    return dict(cpu=torch.get_rng_state(),cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore(state):
    torch.set_rng_state(state['cpu'].cpu())
    if state['cuda']:torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])


def dense(emb,mask):
    result=emb.new_zeros((*mask.shape,24));result[mask]=emb;return result


def direct(model,raw,mask,times,y=None,pos=None,kind='macro',chunk=4096):
    flat=normalize_windows(raw[mask])
    emb=torch.cat([model.cnn(q[:,None]).flatten(1) for q in flat.split(chunk)])
    z=model.aggregate(dense(emb,mask),mask,times);logit=model.classify(z)
    return (objective(logit,y,pos,kind),logit,z) if y is not None else (logit,z)


def cached(model,raw,mask,times,y=None,pos=None,kind='macro',chunk=4096):
    train=y is not None;flat=normalize_windows(raw[mask]);states=[];values=[]
    with torch.no_grad():
        for q in flat.split(chunk):
            if train:states.append(rng())
            values.append(model.cnn(q[:,None]).flatten(1))
    emb=torch.cat(values).detach().requires_grad_(train)
    assert torch.isfinite(emb).all()
    z=model.aggregate(dense(emb,mask),mask,times);logit=model.classify(z)
    assert torch.isfinite(logit).all() and torch.isfinite(z).all()
    if not train:return logit.detach(),z.detach()
    loss=objective(logit,y,pos,kind);assert torch.isfinite(loss);loss.backward()
    grad=emb.grad.detach();assert torch.isfinite(grad).all()
    after=rng();buffers={k:v.detach().clone() for k,v in model.named_buffers()}
    offset=0
    try:
        for q,state in zip(flat.split(chunk),states):
            restore(state);h=model.cnn(q[:,None]).flatten(1)
            h.backward(grad[offset:offset+len(q)]);offset+=len(q)
    finally:
        with torch.no_grad():
            for k,v in model.named_buffers():v.copy_(buffers[k])
        restore(after)
    return float(loss.detach().cpu()),logit.detach(),z.detach()
