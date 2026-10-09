"""Original A0 patient-update semantics plus locked normalized source L2."""
import copy
import json
from pathlib import Path
import numpy as np
import torch
from model import SourceA0
from common import sha
from uas_core import seed_all,state_hash,restore_rng,rng_state,select_threshold,torch_write,json_write,groups

def train(x,y,pid,g,tr,va,cell,seed,bind,device='cuda',max_epochs=30,stop_after=None,penalty=.001):
    cell=Path(cell); cell.mkdir(parents=True,exist_ok=True)
    seed_all(seed); model=SourceA0(seed).to(device)
    initial=copy.deepcopy(model.state_dict()); ih=state_hash(initial)
    binding={'source':bind,'seed':seed,'initial_hash':ih,'penalty':penalty,'max_epochs':max_epochs,'device':device}
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    seed_all(seed) # Same training reseed as original after model construction.
    counts=torch.tensor([len(set(pid[tr][g[tr]==k])) for k in range(4)],dtype=torch.float32,device=device)
    pos=torch.tensor(max(int((1-y[tr]).sum()),1)/max(int(y[tr].sum()),1),device=device)
    fn=torch.nn.BCEWithLogitsLoss(pos_weight=pos)
    last=cell/'LAST_PRIVATE.pt'; complete=cell/'COMPLETE_PRIVATE.json'
    history=[]; best=None; key=None; stale=0; start=1
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False); assert saved['binding']==binding
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer']); restore_rng(saved['rng'])
        history=saved['history']; best=saved['best']; key=saved['best_key']; stale=saved['stale']; start=saved['epoch']+1
        if complete.exists():
            meta=json.loads(complete.read_text()); assert meta['binding']==binding
            assert meta['last_sha256']==sha(last) and meta['best_sha256']==sha(cell/'BEST_PRIVATE.pt')
            model.load_state_dict(best['model']); model.eval(); return model,best,history
        if saved['epoch']>=6 and stale>=6:start=max_epochs+1
    tx=torch.as_tensor(x,device=device); ty=torch.as_tensor(y,dtype=torch.float32,device=device); tg=torch.as_tensor(g,device=device)
    train_groups=[tr[ix] for ix in groups(pid[tr])]
    for epoch in range(start,max_epochs+1):
        model.train(); losses=[]; regs=[]
        for position in np.random.default_rng(seed+epoch).permutation(len(train_groups)):
            ix=train_groups[int(position)]; opt.zero_grad(set_to_none=True)
            reg=model.regularizer(counts); loss=fn(model(tx[ix],tg[ix]),ty[ix])+penalty*reg
            assert torch.isfinite(loss)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); opt.step()
            losses.append(float(loss.detach().cpu())); regs.append(float(reg.detach().cpu()))
        model.eval()
        with torch.no_grad():score=torch.sigmoid(model(tx[va],tg[va])).cpu().numpy()
        sel=select_threshold(y[va],score,pid[va]); candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),'source_l2':float(np.mean(regs)),**sel})
        if key is None or candidate>key:
            key=candidate; stale=0; best={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'epoch':epoch,**sel}
        else:stale+=1
        torch_write(last,{'binding':binding,'model':model.state_dict(),'optimizer':opt.state_dict(),'rng':rng_state(),
            'epoch':epoch,'best':best,'best_key':key,'stale':stale,'history':history})
        print('EPOCH',cell.name,epoch,'best',best['epoch'],flush=True)
        if stop_after and epoch==stop_after:return None,best,history
        if epoch>=6 and stale>=6:break
    assert best is not None
    torch_write(cell/'BEST_PRIVATE.pt',{'binding':binding,**best})
    json_write(complete,{'binding':binding,'last_sha256':sha(last),'best_sha256':sha(cell/'BEST_PRIVATE.pt'),
        'selected_epoch':best['epoch'],'threshold':best['threshold'],'source_fit_patient_counts':counts.cpu().tolist()})
    model.load_state_dict(best['model']); model.eval(); return model,best,history
