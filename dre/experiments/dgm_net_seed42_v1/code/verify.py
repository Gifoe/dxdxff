"""Architecture, identity and actual legal-FIT gradient checks before training."""
import numpy as np
import torch
from torch import nn

def check_model(model,d,store,scale,core):
    counts={name:sum(p.numel() for p in module.parameters()) for name,module in [('frozen_D1',model.base),('CNN',model.cnn),('gate',model.gate),('head',model.head)]}
    counts['alpha']=model.alpha.numel()
    assert counts==dict(frozen_D1=1537,CNN=2160,gate=1048,head=6273,alpha=1)
    assert not any(isinstance(m,(nn.GRU,nn.LSTM,nn.MultiheadAttention,nn.Transformer,nn.TransformerEncoder)) for m in model.modules())
    group=d['train'][d['patient'][d['train']]==sorted(set(d['patient'][d['train']]))[0]]
    pid=str(d['patient'][group[0]]);raw,mask,times,delta,perms=store.get(pid,d['channel'][group].astype(str).tolist(),'cuda')
    x=torch.as_tensor(d['x'][group],device='cuda');y=torch.as_tensor(d['y'][group],device='cuda',dtype=torch.float32)
    frozen=core.state_hash(model.base.state_dict()); model.eval()
    with torch.inference_mode():
        a=model(x,raw,mask,times,delta,scale);b=model(x,raw,mask,times,delta,scale,'identity')
        assert torch.equal(a[0],a[2]) and torch.equal(a[1],b[1]) and torch.equal(a[4],torch.ones_like(a[4]))
        u=torch.randn(19,7,device='cuda');assert torch.equal(model.gate(u),torch.ones(19,24,device='cuda'))
    opt=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
    pos=max(int((1-d['y'][d['train']]).sum()),1)/max(int(d['y'][d['train']].sum()),1)
    fn=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda')); grads=[]
    for step in range(2):
        model.train();opt.zero_grad(set_to_none=True);final,aux,base,res,g=model(x,raw,mask,times,delta,scale)
        loss=fn(final,y)+.25*fn(aux,y);loss.backward()
        assert all(p.grad is None for p in model.base.parameters()) and not model.base.training
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        names=['cnn.0.weight','cnn.2.weight','head.0.weight','gate.network.0.weight','gate.network.2.weight','alpha']
        norms={n:float(dict(model.named_parameters())[n].grad.norm().cpu()) for n in names}
        assert norms['cnn.0.weight']>0 and norms['head.0.weight']>0 and norms['gate.network.2.weight']>0 and norms['alpha']>0
        if step==0:assert norms['gate.network.0.weight']==0
        else:assert norms['gate.network.0.weight']>0
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        opt.step();grads.append(dict(step=step+1,**norms))
    assert frozen==core.state_hash(model.base.state_dict())
    with torch.inference_mode():
        a=model.eval()(x,raw,mask,times,delta,scale); b=model(x,raw,mask,times,delta,scale,'identity')
        drift=float((a[1]-b[1]).abs().max().cpu()); assert drift>0
        assert torch.equal(a[2],b[2])
    del raw,mask,times,delta,perms,x,y,final,aux,base,res,g,a,b
    return dict(status='PASS',parameters=counts,total=11019,trainable=9482,legal_FIT_smoke_steps=2,smoke_model_discarded=True,gradients=grads,identity_at_initialization_exact=True,frozen_D1_unchanged=True,no_recurrent_or_attention_module=True,trained_gate_identity_aux_logit_max_change=drift,physiology_requires_grad=False)
