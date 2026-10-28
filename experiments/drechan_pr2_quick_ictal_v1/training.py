"""Shared deterministic selection, atomic Adam/RNG recovery, and inference."""
import time
import numpy as np
import torch
from scipy.special import expit
from runtime import *
from metrics import validation
from losses import objective

def components(out):
    a={k:out[k].detach().cpu().numpy().astype(np.float64) for k in ['anchor','delta_rank','delta_cal']}
    a['rank']=a['anchor']+a['delta_rank'];a['final']=a['rank']+a['delta_cal'];a['rank_score']=a['rank'].copy()
    return a

def predict(model,data,indices,fold,stage):
    model.eval();result={k:np.empty(len(indices),np.float64) for k in ['anchor','delta_rank','delta_cal','rank','final','rank_score']}
    diagnostic=[]
    with torch.no_grad():
        for patient in sorted(set(data['patient'][indices])):
            jj=np.flatnonzero(data['patient'][indices]==patient);ii=indices[jj]
            if stage=='A':out=model(torch.as_tensor(data['raw12'][ii],dtype=torch.float32),torch.as_tensor(data['base'][ii],dtype=torch.float32))
            else:out=model(*raw_arguments(data,ii,fold))
            a=components(out)
            for k in result:result[k][jj]=a[k]
            for d in out.get('diagnostics',[]):
                scale=d['reference_scale'].detach().cpu().numpy();relative=d['relative_abs'].detach().cpu().numpy()
                diagnostic.append(dict(scale_min=float(scale.min()),scale_median=float(np.median(scale)),scale_max=float(scale.max()),
                                       absolute_relative_median=float(np.median(relative)),absolute_relative_max=float(relative.max())))
    assert all(np.isfinite(x).all() for x in result.values())
    result['scale_diagnostics']=diagnostic;return result

def select(data,ix,output,stage):
    query=subset(data,ix)
    if stage=='A':return validation(query,output,'EZ_GE')
    c=context();pnez=1-expit(output['final'])
    original=c['core'].select_threshold(1-data['y'][ix],pnez,data['patient'][ix])
    scored=validation_at(query,output,float(original['threshold']),'NEZ_LT')
    assert abs(scored['macro_f1']-original['macro_f1'])<1e-12
    assert abs(scored['ez_f1']-original['ez_f1'])<1e-12
    return scored

def validation_at(data,out,t,kind):
    from metrics import patients
    rows=patients(data,out,t,kind)
    return dict(threshold=t,**{m:float(np.mean([r[m] for r in rows])) for m in ['auroc','ap','macro_f1','ez_f1']})

def fit(model,data,fold,arm,stage,binding):
    seed=42+1009*fold;seed_all(seed)
    base=ROOT/'models_private'/arm/f'fold{fold}';base.mkdir(parents=True,exist_ok=True)
    last,best_path,validation_path,done=[base/n for n in ['LAST.pt','BEST.pt','VALIDATION.pkl','COMPLETE.json']]
    cb=dict(binding=binding,arm=arm,fold=fold,initial=state_hash(model.state_dict()))
    if done.exists():
        r=read_json(done);assert r['binding']==digest(cb) and sha(best_path)==r['checkpoint_sha256']
        assert sha(validation_path)==r['validation_sha256']
        cell=load(validation_path)['cell'];cell['validation_sha256']=sha(validation_path);return cell
    lr,wd,max_epochs=(.001,.0001,50) if stage=='A' else (.0003,.001,30)
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=lr,weight_decay=wd)
    tr,va=data['train'],data['val'];ratio=float(data['y'][tr].sum()/(1-data['y'][tr]).sum())
    groups=[tr[data['patient'][tr]==p] for p in sorted(set(data['patient'][tr]))]
    best=None;history=[];stale=0;start=1
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==cb
        model.load_state_dict(q['model']);optimizer.load_state_dict(q['optimizer']);restore_rng(q['rng'])
        best,history,stale,start=q['best'],q['history'],q['stale'],q['epoch']+1
        if start>6 and stale>=6:start=max_epochs+1
    for epoch in range(start,max_epochs+1):
        model.train();losses=[];tick=time.perf_counter()
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];optimizer.zero_grad(set_to_none=True)
            if stage=='A':
                out=model(torch.as_tensor(data['raw12'][ii],dtype=torch.float32),torch.as_tensor(data['base'][ii],dtype=torch.float32));device='cpu'
            else:out=model(*raw_arguments(data,ii,fold));device='cuda'
            y=torch.as_tensor(data['y'][ii],dtype=torch.float32,device=device)
            loss=objective(out,y,ratio,'svm' if stage=='A' else 'drechan')
            assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters() if p.requires_grad)
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();losses.append(float(loss.detach().cpu()))
            del loss,out,y
        output=predict(model,data,va,fold,stage);sel=select(data,va,output,stage)
        key=tuple(sel[k] for k in (['auroc','ap'] if stage=='A' else ['macro_f1','ez_f1']))+(-epoch,)
        if best is None or key>tuple(best['key']):
            best=dict(key=list(key),epoch=epoch,selected=sel,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()});stale=0
        else:stale+=1
        history.append(dict(epoch=epoch,mean_loss=float(np.mean(losses)),seconds=time.perf_counter()-tick,**sel,
                            scale_diagnostics=output.get('scale_diagnostics',[])))
        atomic_torch(last,dict(binding=cb,model=model.state_dict(),optimizer=optimizer.state_dict(),rng=rng_state(),
                              best=best,history=history,stale=stale,epoch=epoch))
        write(PUBLIC/'TRAIN_STATUS.json',dict(stage=stage,arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch']))
        print('PR2_EPOCH',stage,arm,fold,epoch,'best',best['epoch'],'seconds',round(history[-1]['seconds'],3),flush=True)
        if epoch>=6 and stale>=6:break
    assert best is not None
    model.load_state_dict(best['model']);output=predict(model,data,va,fold,stage);sel=select(data,va,output,stage)
    assert sel==best['selected']
    atomic_torch(best_path,dict(binding=cb,model=best['model'],epoch=best['epoch'],selected=sel))
    cell=dict(arm=arm,fold=fold,stage=stage,checkpoint=str(best_path),checkpoint_sha256=sha(best_path),validation=str(validation_path),
              epoch=best['epoch'],threshold=sel['threshold'],threshold_kind='EZ_GE' if stage=='A' else 'NEZ_LT',
              validation_auroc=sel['auroc'],validation_ap=sel['ap'],validation_macro_f1=sel['macro_f1'],
              trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
              added_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)-(11425 if stage=='B' else 0),
              epochs=len(history),training_seconds=float(sum(r['seconds'] for r in history)),reused=False)
    save(validation_path,dict(binding=cb,cell=cell,components=output,history=history))
    cell['validation_sha256']=sha(validation_path)
    write(done,dict(binding=digest(cb),checkpoint_sha256=sha(best_path),validation_sha256=sha(validation_path)))
    # Keep the same cell schema in resumptions without embedding a self-hash.
    return cell
