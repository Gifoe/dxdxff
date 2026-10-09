"""Original A0 patient-update semantics, scratch W1/W2 and epoch-boundary resume."""
import json
import time
from pathlib import Path
import numpy as np
import torch
from base import seed_all,state_hash,rng_state,restore_rng,sha,json_write,torch_write,select_threshold
from model import make,objective,probability


def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize(device)


def predict(model,x):
    model.eval()
    with torch.no_grad():
        logits=model(x)
        return probability(logits).cpu().numpy(),torch.sigmoid(logits).cpu().numpy()


def train(d,arm,cell,seed,initial,bind,device,max_epochs=30,stop_after=None):
    cell=Path(cell); cell.mkdir(parents=True,exist_ok=True)
    binding={'source':bind,'seed':seed,'arm':arm,'initial_sha256':state_hash(initial)}
    model=make(arm).to(device); model.load_state_dict(initial); seed_all(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    xx=torch.as_tensor(d['x'],device=device); yy=torch.as_tensor(d['y'],dtype=torch.float32,device=device)
    tr,va=d['train'],d['val']; pos=torch.tensor(int((d['y'][tr]==0).sum())/int((d['y'][tr]==1).sum()),device=device)
    ids=sorted(set(d['patient'][tr])); groups=[np.flatnonzero(d['patient']==p) for p in ids]
    assert all(set(ix).issubset(set(tr)) for ix in groups)
    start=1; history=[]; best=None; best_key=None; stale=0; total_seconds=0.; peak=0
    last=cell/'LAST_PRIVATE.pt'; complete=cell/'COMPLETE_PRIVATE.json'
    if last.exists():
        s=torch.load(last,map_location=device,weights_only=False); assert s['binding']==binding
        model.load_state_dict(s['model']); opt.load_state_dict(s['optimizer']); restore_rng(s['rng'])
        history=s['history']; best=s['best']; best_key=s['best_key']; stale=s['stale']; start=s['epoch']+1
        total_seconds=s['train_seconds']; peak=s['peak_gpu_bytes']
        if complete.exists():
            meta=json.loads(complete.read_text()); assert meta['binding']==binding and meta['last_sha256']==sha(last)
            assert meta['best_sha256']==sha(cell/'BEST_PRIVATE.pt')
            model.load_state_dict(best['model']); return model.eval(),best,history,meta
        if s['epoch']>=6 and stale>=6:start=max_epochs+1
    if str(device).startswith('cuda'):torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start,max_epochs+1):
        sync(device); begin=time.perf_counter(); model.train(); losses=[]
        order=np.random.default_rng(seed+epoch).permutation(len(groups))
        for k in order:
            ix=groups[int(k)]; opt.zero_grad(set_to_none=True)
            loss=objective(model(xx[ix]),yy[ix],pos); assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
            opt.step(); losses.append(float(loss.detach().cpu()))
        score,_=predict(model,xx[va]); selected=select_threshold(d['y'][va],score,d['patient'][va])
        key=(selected['macro_f1'],selected['ez_f1'],-epoch)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),**selected})
        if best_key is None or key>best_key:
            best_key=key; stale=0
            best={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,**selected}
        else:stale+=1
        sync(device); total_seconds+=time.perf_counter()-begin
        if str(device).startswith('cuda'):peak=max(peak,torch.cuda.max_memory_allocated(device))
        torch_write(last,{'binding':binding,'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),
            'epoch':epoch,'history':history,'best':best,'best_key':best_key,'stale':stale,'train_seconds':total_seconds,'peak_gpu_bytes':peak})
        print('EPOCH',cell.name,epoch,'F1',selected['macro_f1'],'BEST',best['epoch'],flush=True)
        if stop_after==epoch:return None,best,history,None
        if epoch>=6 and stale>=6:break
    assert best is not None
    torch_write(cell/'BEST_PRIVATE.pt',{'binding':binding,**best})
    meta={'binding':binding,'best_sha256':sha(cell/'BEST_PRIVATE.pt'),'last_sha256':sha(last),
        'selected_epoch':best['epoch'],'threshold':best['threshold'],'train_seconds':total_seconds,
        'peak_gpu_bytes':peak,'completed_epochs':len(history),'pos_weight':float(pos.cpu())}
    json_write(complete,meta); model.load_state_dict(best['model'])
    return model.eval(),best,history,meta


def inference_benchmark(model,x,device):
    model.eval()
    with torch.no_grad():
        for _ in range(10):probability(model(x))
        sync(device); begin=time.perf_counter()
        for _ in range(20):probability(model(x))
        sync(device); elapsed=time.perf_counter()-begin
    return elapsed/20
