"""PR-UAS: exact B0 network and target-only changes; private resumable training."""
from __future__ import annotations

import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score, roc_auc_score


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()


def json_write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial')
    temp.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False),encoding='utf-8')
    temp.replace(path)


def torch_write(path, value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.partial'); torch.save(value,temp); temp.replace(path)


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),
            'torch':torch.get_rng_state(),
            'cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].detach().cpu())
    if torch.cuda.is_available(): torch.cuda.set_rng_state_all([s.detach().cpu() for s in state['cuda']])


class PRMLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.network=torch.nn.Sequential(torch.nn.LayerNorm(88),torch.nn.Linear(88,96),
            torch.nn.GELU(),torch.nn.Dropout(.15),torch.nn.Linear(96,1))

    def forward(self,x):
        return self.network(x).squeeze(-1)


def lambda_epoch(epoch):
    return .25*min(1.,max(0.,(epoch-5)/5.))


def soft_target(y,q,u,epoch):
    # Student sees no teacher labels; q/u are detached OOF probabilities/entropy only.
    return y+lambda_epoch(epoch)*u.detach()*(q.detach()-y)


def patient_z(x, patient):
    out=np.asarray(x,dtype=np.float64).copy()
    for p in np.unique(patient.astype(str)):
        take=patient.astype(str)==p
        a=out[take]; std=np.nanstd(a,axis=0)
        out[take]=(a-np.nanmean(a,axis=0))/np.where(std>1e-8,std,1.)
    return out


def prepare(x, patient, train_indices):
    z=patient_z(x,patient)
    imputer=SimpleImputer(); scaler=StandardScaler()
    imputer.fit(z[train_indices]); scaler.fit(imputer.transform(z[train_indices]))
    result=scaler.transform(imputer.transform(z)).astype(np.float32)
    assert result.shape==x.shape and np.isfinite(result).all()
    return result, {'imputer_statistics':imputer.statistics_, 'mean':scaler.mean_,
                    'scale':scaler.scale_,'var':scaler.var_, 'fit_patient_ids':sorted(set(patient[train_indices]))}


def groups(patient):
    return [np.flatnonzero(patient==p) for p in np.unique(patient)]


def select_threshold(y, score, patient):
    """Vectorized confusion counts, original float64 metric and exact tie order."""
    grid=np.round(np.arange(0.,1.0001,.005),3)
    predicted=np.asarray(score,dtype=float)[:,None]>=grid[None,:]
    values=[]; ezvalues=[]; counts=[]
    for ix in groups(patient):
        a=y[ix,None]; b=predicted[ix]
        tp=((a==1)&b).sum(0); tn=((a==0)&~b).sum(0)
        fp=((a==0)&b).sum(0); fn=((a==1)&~b).sum(0)
        f1=np.divide(2.*tp,2.*tp+fp+fn,out=np.zeros(len(grid)),where=(2*tp+fp+fn)>0)
        f0=np.divide(2.*tn,2.*tn+fp+fn,out=np.zeros(len(grid)),where=(2*tn+fp+fn)>0)
        values.append((f0+f1)/2); ezvalues.append(f0); counts.append((tp,tn,fp,fn))
    values=np.asarray(values); ezvalues=np.asarray(ezvalues)
    tp,tn,fp,fn=np.asarray(counts).sum(axis=0)
    recalls=[]
    if (y==1).any(): recalls.append(tp/(tp+fn))
    if (y==0).any(): recalls.append(tn/(tn+fp))
    balanced=np.mean(recalls,axis=0)
    candidates=[(float(np.mean(values[:,i])),float(np.mean(ezvalues[:,i])),float(balanced[i]),
                 -abs(float(t)-.5),-float(t),float(t)) for i,t in enumerate(grid)]
    best=max(candidates)
    return {'threshold':best[-1],'macro_f1':best[0],'ez_f1':best[1],'pooled_ba':best[2]}


METRICS=['macro_f1','ez_f1','nez_f1','balanced_accuracy','ez_auprc','ez_auroc',
         'ez_mrr','top1_is_ez','sensitivity','specificity','accuracy']


def patient_metrics(y, score, patient, center, threshold):
    rows=[]
    for ix in groups(patient):
        a=y[ix]; p=score[ix]; b=p>=threshold
        tp=int(((a==0)&~b).sum()); fp=int(((a==1)&~b).sum())
        tn=int(((a==1)&b).sum()); fn=int(((a==0)&b).sum())
        ef=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.
        nf=2*tn/(2*tn+fp+fn) if 2*tn+fp+fn else 0.
        se=tp/(tp+fn) if tp+fn else np.nan
        sp=tn/(tn+fp) if tn+fp else np.nan
        ez=a==0; pe=1-np.asarray(p,dtype=float)
        order=np.argsort(-pe,kind='mergesort'); positives=np.flatnonzero(ez[order])
        rows.append({'patient':str(patient[ix[0]]),'center':str(center[ix[0]]),
            'channels':len(ix),'macro_f1':(ef+nf)/2,'ez_f1':ef,'nez_f1':nf,
            'balanced_accuracy':float(np.nanmean([se,sp])),
            'ez_auprc':float(average_precision_score(ez,pe)) if ez.any() else 0.,
            'ez_auroc':float(roc_auc_score(ez,pe)) if 0<ez.sum()<len(ix) else np.nan,
            'ez_mrr':float(1/(positives[0]+1)) if len(positives) else 0.,
            'top1_is_ez':float(ez[order[0]]),'sensitivity':se,'specificity':sp,
            'accuracy':(tp+tn)/len(ix),'TP':tp,'FP':fp,'TN':tn,'FN':fn})
    return pd.DataFrame(rows)


def oof_plan(patient_ids, fold):
    ids=np.asarray(sorted(set(patient_ids)),dtype=str)
    query_groups=np.array_split(np.random.default_rng(42).permutation(ids),4)
    plan=[]
    for k,q in enumerate(query_groups):
        seed=42+1009*fold+100000*(k+1)
        remain=np.asarray(sorted(set(ids)-set(q)))
        order=np.random.default_rng(seed).permutation(remain)
        n=math.ceil(.2*len(order))
        tr,va=sorted(order[n:]),sorted(order[:n])
        validate_exclusion(tr,va,q,ids)
        plan.append({'group':k,'train':tr,'validation':va,'query':sorted(q),'seed':seed})
    assert set(np.concatenate(query_groups))==set(ids)
    return plan


def validate_exclusion(train, val, query, outer_fit):
    tr,va,q=map(set,(train,val,query)); outer=set(outer_fit)
    if tr&va or tr&q or va&q or tr|va|q != outer or not tr or not va or not q:
        raise ValueError('OOF teacher membership leakage/incompleteness')


def mc_predict(model, x, seed):
    seed_all(seed); model.eval()
    for module in model.modules():
        if isinstance(module,torch.nn.Dropout): module.train()
    with torch.no_grad():
        draws=torch.stack([torch.sigmoid(model(x)) for _ in range(10)]).cpu().numpy().astype(float)
    model.eval()
    q=np.clip(draws.mean(axis=0),1e-6,1-1e-6)
    entropy=lambda p:-(p*np.log(p)+(1-p)*np.log1p(-p))/np.log(2.)
    u=entropy(q); expected=entropy(np.clip(draws,1e-6,1-1e-6)).mean(0)
    assert np.isfinite(draws).all() and np.isfinite(u).all()
    assert np.any(np.var(draws,axis=0)>0), 'MC Dropout did not vary'
    return {'q':q,'u':u,'variance':draws.var(0),'expected_entropy':expected,
            'mutual_information':np.maximum(u-expected,0.)}


def state_hash(state):
    h=hashlib.sha256()
    for name,value in sorted(state.items()):
        h.update(name.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def train_model(x,y,patient,train_ix,val_ix,cell,seed,initial,bind,device,arm='A0',q=None,u=None,max_epochs=30,stop_after=None):
    """One patient update, unchanged B0 optimization; epoch-boundary exact resume."""
    cell=Path(cell); cell.mkdir(parents=True,exist_ok=True)
    last=cell/'LAST_PRIVATE.pt'; complete=cell/'COMPLETE_PRIVATE.json'
    binding={'source':bind,'seed':seed,'arm':arm,'initial_hash':state_hash(initial)}
    model=PRMLP().to(device); model.load_state_dict(initial)
    seed_all(seed)
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    positive=max(int(y[train_ix].sum()),1); negative=max(int((1-y[train_ix]).sum()),1)
    pos_weight=torch.tensor(negative/positive,device=device)
    lossfn=torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    history=[]; best_key=None; best=None; stale=0; start=1
    if last.exists():
        saved=torch.load(last,map_location=device,weights_only=False)
        assert saved['binding']==binding, 'Resume source/initial/arm mismatch'
        model.load_state_dict(saved['model']); opt.load_state_dict(saved['optimizer'])
        restore_rng(saved['rng']); history=saved['history']; best=saved['best']
        best_key=saved['best_key']; stale=saved['stale']; start=saved['epoch']+1
        if complete.exists():
            meta=json.loads(complete.read_text()); assert meta['binding']==binding
            assert meta['last_sha256']==sha(last)
            model.load_state_dict(best['model']); model.eval(); return model,best,history
        if saved['epoch']>=6 and stale>=6: start=max_epochs+1
    allx=torch.as_tensor(x,device=device); ally=torch.as_tensor(y,dtype=torch.float32,device=device)
    tq=torch.as_tensor(q,dtype=torch.float32,device=device) if q is not None else None
    tu=torch.as_tensor(u,dtype=torch.float32,device=device) if u is not None else None
    train_ids=patient[train_ix]; train_groups=[train_ix[i] for i in groups(train_ids)]
    for epoch in range(start,max_epochs+1):
        model.train(); losses=[]
        order=np.random.default_rng(seed+epoch).permutation(len(train_groups))
        for position in order:
            ix=train_groups[int(position)]; target=ally[ix]
            if arm=='A1': target=.9*target+.05
            elif arm=='A2':
                assert tq is not None and tu is not None
                target=soft_target(target,tq[ix],tu[ix],epoch)
            opt.zero_grad(set_to_none=True); loss=lossfn(model(allx[ix]),target)
            if not torch.isfinite(loss): raise RuntimeError('Nonfinite BCE')
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            opt.step(); losses.append(float(loss.detach().cpu()))
        model.eval()
        with torch.no_grad(): score=torch.sigmoid(model(allx[val_ix])).cpu().numpy()
        selected=select_threshold(y[val_ix],score,patient[val_ix])
        key=(selected['macro_f1'],selected['ez_f1'],-epoch)
        history.append({'epoch':epoch,'loss':float(np.mean(losses)),**selected,'lambda':lambda_epoch(epoch) if arm=='A2' else 0.})
        if best_key is None or key>best_key:
            best_key=key; stale=0
            best={'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                  'epoch':epoch,**selected}
        else: stale+=1
        torch_write(last,{'binding':binding,'model':model.state_dict(),'optimizer':opt.state_dict(),
                         'rng':rng_state(),'epoch':epoch,'stale':stale,'best_key':best_key,
                         'best':best,'history':history})
        print(f'EPOCH {cell.name} arm={arm} epoch={epoch} val={key[0]:.6f} selected={best["epoch"]}',flush=True)
        if stop_after and epoch==stop_after: return None,best,history
        if epoch>=6 and stale>=6: break
    assert best is not None
    torch_write(cell/'BEST_PRIVATE.pt',{'binding':binding,**best})
    json_write(complete,{'binding':binding,'last_sha256':sha(last),'best_sha256':sha(cell/'BEST_PRIVATE.pt'),
                         'selected_epoch':best['epoch'],'threshold':best['threshold']})
    model.load_state_dict(best['model']); model.eval()
    return model,best,history
