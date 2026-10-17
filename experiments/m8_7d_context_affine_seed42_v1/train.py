"""Complete-patient frozen-hidden training, ten real additional head fits."""
import gc
import time
from common import *


def linear(q):
    m=torch.nn.Linear(64,1);m.load_state_dict(q['readout']);return m


def fit(fold,d,q,arm,binding,core):
    m=make(arm,linear(q),fold);seed=42+1009*fold
    bind=dict(input=binding,arm=arm,fold=fold,initial=state_hash(m.state_dict()),
        representation_seal=sha(ROOT/'REPRESENTATION_SEAL_PRIVATE.json'),body_checkpoint=q['binding']['body_checkpoint'],
        readout=state_hash(m.readout.state_dict()))
    cell=ROOT/'models_private'/arm/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True)
    last,bp,hp,done=[cell/n for n in ['LAST_PRIVATE.pt','BEST_PRIVATE.pt','VALIDATION_PRIVATE.pkl','COMPLETE_PRIVATE.json']]
    if done.exists():
        v=jr(done);assert v['binding']==digest(bind) and sha(bp)==v['best_sha256'] and sha(hp)==v['validation_sha256']
        v=read(hp);assert v['binding']==bind;return v['summary'],v['rows'],v['history']
    seed_all(seed);params=[p for p in m.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=.0003,weight_decay=.001);tr,va=d['train'],d['val']
    pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    best=key=None;history=[];stale=0;start=1;h=q['hidden']
    if last.exists():
        v=torch.load(last,map_location='cpu',weights_only=False);assert v['binding']==bind
        m.load_state_dict(v['model']);opt.load_state_dict(v['optimizer']);core.restore_rng(v['rng'])
        best,key,history,stale,start=v['best'],v['key'],v['history'],v['stale'],v['epoch']+1
        if v['epoch']>=15 and stale>=15:start=101
    groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))]
    for epoch in range(start,101):
        tick=time.perf_counter();m.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];hh=torch.as_tensor(h[ii],device='cuda');y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);loss=MODEL.objective(m(hh),y,pos);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert all(p.grad is None for p in m.readout.parameters());torch.nn.utils.clip_grad_norm_(params,1.);opt.step();losses.append(float(loss.detach().cpu()))
        sel=core.select_threshold(d['y'][va],probability(predict(m,h,d,va)),d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        history.append(dict(arm=arm,fold=fold,epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**sel))
        assert state_hash(m.readout.state_dict())==bind['readout']
        atomic_torch(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING_CONTEXT_AFFINE',arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch'],new_Outer_passes=0))
        print('CONTEXT_AFFINE_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=15 and stale>=15:break
    assert best['epoch']>=1;m.load_state_dict(best['model']);assert state_hash(m.readout.state_dict())==bind['readout']
    atomic_torch(bp,dict(binding=bind,**best));out=predict(m,h,d,va)
    sel=core.select_threshold(d['y'][va],probability(out),d['patient'][va]);assert all(abs(float(v)-float(best[k]))<1e-12 for k,v in sel.items())
    rows=patient_rows(d,va,out,best['threshold'],arm,fold)
    summary=dict(arm=arm,fold=fold,epochs=len(history),selected_epoch=best['epoch'],threshold=best['threshold'],patients=len(rows),
        parameters=10506 if arm=='G0_GLOBAL' else 12562,trainable_parameters=sum(p.numel() for p in params),checkpoint_sha256=sha(bp),
        initial_state_sha256=bind['initial'],execution_kind='NEW_HEAD_FIT_FROZEN_C2_BODY',**{k:float(np.mean([r[k] for r in rows])) for k in METRICS})
    save(hp,dict(binding=bind,archive_indices=d['archive_indices'][va],logit=out,rows=rows,summary=summary,history=history))
    write(done,dict(binding=digest(bind),best_sha256=sha(bp),validation_sha256=sha(hp)))
    del m,opt;gc.collect();torch.cuda.empty_cache();return summary,rows,history


def main():
    binding,core,fac,records,rec,banks,fr=setup();ad=jr(PUB/'ADMISSION.json');assert ad['status']=='PASS' and ad['input_binding_sha256']==digest(binding)
    summaries=[];rows=[];histories=[];cells=[]
    for fold in range(1,6):
        d=bank(fold,banks);q=cached(fold,d);m,r,state,old=body(fr,fold,core,fac,records);del m
        part=patient_rows(d,d['val'],q['reference_logit'][d['val']],state['threshold'],ARMS[0],fold);rows+=part
        summaries.append(dict(arm=ARMS[0],fold=fold,epochs=0,prior_body_training_epochs=old['summary']['epochs'],
            selected_epoch=state['epoch'],threshold=state['threshold'],patients=len(part),parameters=10378,trainable_parameters=0,
            checkpoint_sha256=r['checkpoint_sha256'],execution_kind='REUSED_EXACT_C2',**{k:float(np.mean([z[k] for z in part])) for k in METRICS}))
        cells.append(dict(r,arm=ARMS[0]))
        for arm in NEW:
            summary,part,history=fit(fold,d,q,arm,binding,core);summaries.append(summary);rows+=part;histories+=history
            root=ROOT/'models_private'/arm/f'fold{fold}'
            cells.append(dict(arm=arm,fold=fold,checkpoint=str(root/'BEST_PRIVATE.pt'),checkpoint_sha256=sha(root/'BEST_PRIVATE.pt'),
                validation=str(root/'VALIDATION_PRIVATE.pkl'),validation_sha256=sha(root/'VALIDATION_PRIVATE.pkl'),
                body_checkpoint=r['checkpoint'],body_checkpoint_sha256=r['checkpoint_sha256']))
            pd.DataFrame(summaries).to_csv(PUB/'DEVELOPMENT_CELLS.csv',index=False);pd.DataFrame(histories).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
    save(ROOT/'DEVELOPMENT_PATIENTS_PRIVATE.pkl',rows);freeze(ROOT/'MODEL_FREEZE_PRIVATE.json',dict(input=binding,cells=cells))
    choice=choose(pd.DataFrame(rows))
    freeze(ROOT/'CANDIDATE_PRIVATE.json',dict(input=binding,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),selection=choice))
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL10_NEW_PLUS5_REUSED_MODELS_FROZEN',actual_new_fits=10,reused_controls=5,epochs=len(histories),
        new_Outer_passes=0,body_updates=0,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),candidate_sha256=sha(ROOT/'CANDIDATE_PRIVATE.json'),goal_established=False))
    print('CONTEXT_AFFINE_ALL_TRAIN_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':main()
