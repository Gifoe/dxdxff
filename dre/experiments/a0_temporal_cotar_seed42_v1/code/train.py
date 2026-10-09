"""Matched residual training; eligible epoch0 and exact epoch-boundary resume."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from base import (seed_all,state_hash,rng_state,restore_rng,sha,json_write,torch_write,
                  select_threshold,patient_metrics,METRICS)
from model import Temporal


def tensors(banks,device):
    return {p:{k:torch.as_tensor(b[k],device=device) for k in ['z','time','mask']} for p,b in banks.items()}


def predict(model,d,banks,indices,device):
    delta=np.zeros(len(d['y']),np.float32); model.eval()
    with torch.no_grad():
        for p in sorted(set(d['patient'][indices])):
            ix=np.flatnonzero(d['patient']==p); b=banks[p]
            value=model(b['z'],b['time'],b['mask']).detach().cpu().numpy()
            assert value.shape==(len(ix),) and np.isfinite(value).all()
            delta[ix]=value
    logits=d['base_logits']+delta
    score=torch.sigmoid(torch.from_numpy(logits[indices])).numpy()
    return score,delta


def train(d,arm,cell,seed,initial,bind,device,max_epochs=30,stop_after=None):
    cell=Path(cell); cell.mkdir(parents=True,exist_ok=True)
    binding={'source':bind,'seed':seed,'arm':arm,'initial':state_hash(initial)}
    model=Temporal(arm).to(device); model.load_state_dict(initial); seed_all(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-3)
    b=tensors(d['banks'],device); tr,va=d['train'],d['val']
    yy=torch.as_tensor(d['y'],dtype=torch.float32,device=device)
    base=torch.as_tensor(d['base_logits'],device=device).detach(); assert not base.requires_grad
    pos=torch.tensor(int((d['y'][tr]==0).sum())/int((d['y'][tr]==1).sum()),device=device)
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=pos)
    train_ids=sorted(set(d['patient'][tr])); groups=[np.flatnonzero(d['patient']==p) for p in train_ids]
    start=1; stale=0; last=cell/'LAST_PRIVATE.pt'; complete=cell/'COMPLETE_PRIVATE.json'
    if last.exists():
        s=torch.load(last,map_location=device,weights_only=False); assert s['binding']==binding
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); restore_rng(s['rng'])
        start=s['epoch']+1; history=s['history']; best=s['best']; best_key=s['best_key']; stale=s['stale']
        if complete.exists():
            assert json.loads(complete.read_text())['last_sha256']==sha(last)
            model.load_state_dict(best['model']); return model.eval(),best,history
        if s['epoch']>=6 and stale>=6: start=max_epochs+1
    else:
        score,delta=predict(model,d,b,va,device); assert (delta==0).all()
        selected=select_threshold(d['y'][va],score,d['patient'][va])
        assert selected['threshold']==d['T0_threshold']
        best={'model':copy.deepcopy(initial),'epoch':0,**selected}
        best_key=(selected['macro_f1'],selected['ez_f1'],0)
        history=[{'epoch':0,'loss':None,**selected,'alpha_bound':.5}]
    for epoch in range(start,max_epochs+1):
        model.train(); losses=[]
        order=np.random.default_rng(seed+epoch).permutation(len(groups))
        for k in order:
            ix=groups[int(k)]; x=b[train_ids[int(k)]]
            opt.zero_grad(set_to_none=True)
            delta=model(x['z'],x['time'],x['mask'])
            loss=lossfn(base[ix]+delta,yy[ix]); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step(); losses.append(float(loss.detach().cpu()))
        score,_=predict(model,d,b,va,device)
        selected=select_threshold(d['y'][va],score,d['patient'][va])
        key=(selected['macro_f1'],selected['ez_f1'],-epoch)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),**selected,'alpha_bound':.5})
        if key>best_key:
            best_key=key; stale=0
            best={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,**selected}
        else: stale+=1
        torch_write(last,{'binding':binding,'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),
            'epoch':epoch,'history':history,'best':best,'best_key':best_key,'stale':stale})
        print('EPOCH',cell.name,epoch,'F1',selected['macro_f1'],'BEST',best['epoch'],flush=True)
        if stop_after==epoch:return None,best,history
        if epoch>=6 and stale>=6:break
    torch_write(cell/'BEST_PRIVATE.pt',{'binding':binding,**best})
    json_write(complete,{'binding':binding,'best_sha256':sha(cell/'BEST_PRIVATE.pt'),'last_sha256':sha(last),
        'selected_epoch':best['epoch'],'threshold':best['threshold']})
    model.load_state_dict(best['model']); return model.eval(),best,history
