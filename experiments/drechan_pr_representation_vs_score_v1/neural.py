"""Matched end-to-end fits with detached, epoch-bound FIT latent moments."""
import gc
import time
import types
import numpy as np
import torch
from scipy.special import expit
from task_io import *
from math_core import fit_feature,normalize_tensor,StreamingMoments,preserved_eval

def atomic_torch(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);tmp=path.with_name(path.name+'.partial');torch.save(obj,tmp);tmp.replace(path)

def make(p,model,fold,condition):
    if model=='DRECHANNET':
        m=p.own_model(fold);m.raw_pr=False;m.feature_pr=False
        m._base_features=m.features
        def features(self,*args,**kwargs):
            z,activity=self._base_features(*args,**kwargs)
            return normalize_tensor(z,self._condition,self._fit_stats),activity
        m.features=types.MethodType(features,m)
    else:
        m=p.raw_model(fold,False);m._base_classify=m.classify
        def classify(self,z):
            return self._base_classify(z if self._collecting else normalize_tensor(z,self._condition,self._fit_stats))
        m.classify=types.MethodType(classify,m)
    m._condition=condition;m._fit_stats=None;m._collecting=False
    return m

def data(p,model,fold,condition,pre=None):
    d=dict(p.bank(fold))
    if model=='DRECHANNET':
        raw=p.rec['R14'][d['archive_indices']].copy();raw[:,8:10]=0.
        if pre is None:x,pre=fit_feature(raw,d['patient'],d['train'],condition,True)
        else:
            from math_core import query_feature
            x=query_feature(raw,d['patient'],pre)
        d['x']=x.astype(np.float32)
    return d,pre

def groups(d,indices):
    return [np.asarray(indices)[np.asarray(d['patient'])[indices]==pid] for pid in sorted(set(d['patient'][indices]))]

def collect(p,m,d,model,fold):
    running=StreamingMoments();before=p.own.state_hash(m.state_dict());m._collecting=True
    try:
        with preserved_eval(m):
            for n,ii in enumerate(groups(d,d['train']),1):
                if model=='DRECHANNET':z,_=p.own_raw(m,d,ii,fold,True)
                else:_,z=p.raw_forward(m,d,ii)
                running.add(z.detach().cpu().numpy())
                if n%10==0:status('FIT_STATISTICS',model=model,condition=m._condition,fold=fold,patients_complete=n,patients_total=len(groups(d,d['train'])))
    finally:m._collecting=False
    assert p.own.state_hash(m.state_dict())==before
    out=running.result();assert out['n']==len(d['train'])
    return out

def predict(p,m,d,indices,model,fold):
    m.eval();ll=np.empty(len(indices),np.float32);position={int(i):j for j,i in enumerate(indices)}
    with torch.inference_mode():
        for ii in groups(d,indices):
            if model=='DRECHANNET':z,_=p.own_forward(m,d,ii,fold)
            else:z,_=p.raw_forward(m,d,ii)
            jj=[position[int(i)] for i in ii];ll[jj]=z.detach().cpu().numpy()
    return ll

def selected(p,model,y_nez,pid,ll):
    if model=='DRECHANNET':
        q=p.own.probability(ll);v=p.core.select_threshold(y_nez,q,pid)
        return dict(threshold=v['threshold'],orientation='NEZ_LT',macro_f1=v['macro_f1'],ez_f1=v['ez_f1'],parent_selection=v),1-q
    q=1-torch.sigmoid(torch.as_tensor(ll)).numpy();v=p.ev.threshold(1-y_nez,q,pid)
    return dict(threshold=v['threshold'],orientation='EZ_GE',macro_f1=v['validation_macro_f1'],ez_f1=v['validation_positive_f1'],parent_selection=v),q

def fit(p,model,condition,fold,binding):
    cell=ROOT/'models_private'/model/condition/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True)
    done=cell/'COMPLETE.json'
    if done.exists():
        r=jr(done);assert r['binding']==binding and sha(cell/'BEST.pt')==r['checkpoint_sha256'] and sha(cell/'VALIDATION.pkl')==r['validation_sha256'];return r['selection']
    d,pre=data(p,model,fold,condition);save(cell/'PREPROCESSOR.pkl',pre)
    m=make(p,model,fold,condition);initial=p.own.state_hash(m.state_dict());seed=42+1009*fold;p.core.seed_all(seed)
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001 if model=='DRECHANNET' else .0001,**({} if model=='DRECHANNET' else {'fused':True}))
    pos=torch.tensor(float((1-d['y'][d['train']]).sum()/d['y'][d['train']].sum()),device='cuda')
    last=cell/'LAST.pt';history=[];best=None;key_best=None;stale=0;start=1
    if last.exists():
        q=torch.load(last,weights_only=False,map_location='cpu');assert q['binding']==binding and q['initial']==initial
        m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);p.core.restore_rng(q['rng']);m._fit_stats=q['statistics']
        history,best,key_best,stale,start=q['history'],q['best'],q['key_best'],q['stale'],q['epoch']+1
        if q['epoch']>=6 and stale>=6:start=31
    else:
        cache=ROOT/'initial_statistics_private'/f'{model}_fold{fold}.pkl'
        if cache.exists():
            q=read(cache);assert q['binding']==binding and q['initial']==initial;m._fit_stats=q['statistics']
        else:
            m._fit_stats=collect(p,m,d,model,fold);save(cache,dict(binding=binding,initial=initial,statistics=m._fit_stats))
    trgroups=groups(d,d['train'])
    for epoch in range(start,31):
        if __import__('shutil').disk_usage(ROOT).free<5*1024**3:raise RuntimeError('Disk reserve reached; preserve checkpoint, do not delete parents')
        tick=time.perf_counter();m.train();losses=[]
        for n,k in enumerate(np.random.default_rng(seed+epoch).permutation(len(trgroups)),1):
            ii=trgroups[int(k)];y=torch.as_tensor(d['y'][ii],dtype=torch.float32,device='cuda');opt.zero_grad(set_to_none=True)
            if model=='DRECHANNET':
                logit,_=p.own_forward(m,d,ii,fold);loss=p.own.MODEL.objective(logit,y,pos);assert torch.isfinite(loss);loss.backward();loss_value=float(loss.detach())
            else:loss_value,_=p.raw_forward(m,d,ii,True,y,pos)
            assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step()
            if model=='TIMECONV':m.constraints()
            losses.append(loss_value)
            if n%10==0:status('TRAINING',model=model,condition=condition,fold=fold,epoch=epoch,patients_complete=n,patients_total=len(trgroups))
        m._fit_stats=collect(p,m,d,model,fold)
        ll=predict(p,m,d,d['val'],model,fold);sel,prob=selected(p,model,d['y'][d['val']],d['patient'][d['val']],ll)
        candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        if key_best is None or candidate>key_best:
            key_best=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},statistics=m._fit_stats,epoch=epoch,selection=sel,validation_probability=prob.copy(),validation_logit=ll.copy())
        else:stale+=1
        history.append(dict(epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),best_epoch=best['epoch']))
        atomic_torch(last,dict(binding=binding,initial=initial,model=m.state_dict(),optimizer=opt.state_dict(),rng=p.core.rng_state(),statistics=m._fit_stats,history=history,best=best,key_best=key_best,stale=stale,epoch=epoch))
        status('EPOCH_COMPLETE',model=model,condition=condition,fold=fold,epoch=epoch,best_epoch=best['epoch'],seconds=history[-1]['seconds'])
        print('PR_REPR_EPOCH',model,condition,fold,epoch,'best',best['epoch'],'seconds',round(history[-1]['seconds'],2),flush=True)
        if epoch>=6 and stale>=6:break
    assert best is not None
    m.load_state_dict(best['model']);m._fit_stats=best['statistics'];ll=predict(p,m,d,d['val'],model,fold);sel,prob=selected(p,model,d['y'][d['val']],d['patient'][d['val']],ll)
    assert sel==best['selection'] and np.max(np.abs(prob-best['validation_probability']))<=1e-7
    atomic_torch(cell/'BEST.pt',dict(binding=binding,initial=initial,preprocessor_sha256=sha(cell/'PREPROCESSOR.pkl'),**best))
    save(cell/'VALIDATION.pkl',dict(logit_ez=-ll.astype(float),probability_ez=prob,indices=d['archive_indices'][d['val']],selection=sel))
    out=dict(model=model,condition=condition,fold=fold,checkpoint=str(cell/'BEST.pt'),checkpoint_sha256=sha(cell/'BEST.pt'),preprocessor=str(cell/'PREPROCESSOR.pkl'),preprocessor_sha256=sha(cell/'PREPROCESSOR.pkl'),validation=str(cell/'VALIDATION.pkl'),validation_sha256=sha(cell/'VALIDATION.pkl'),initial_sha256=initial,threshold=sel['threshold'],orientation=sel['orientation'],epoch=best['epoch'],epochs=len(history),reused=False)
    write(done,dict(binding=binding,checkpoint_sha256=out['checkpoint_sha256'],validation_sha256=out['validation_sha256'],selection=out))
    del m,opt;gc.collect();torch.cuda.empty_cache();return out
