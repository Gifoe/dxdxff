"""Ten actual fresh fits and five hash-replayed C2 references; no Outer."""
import gc
import time
from common import *
from admit import replay_control


def fit(ctx,arm,binding,inputs,core,fac,records):
    fold,d,ss=ctx;seed=42+1009*fold;m=make(core,fac,records,fold,arm)
    cell=ROOT/'models_private'/arm/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True)
    bind=dict(input=binding,arm=arm,fold=fold,initial=state_hash(m.state_dict()))
    last,bp,hp,done=[cell/n for n in ['LAST_PRIVATE.pt','BEST_PRIVATE.pt','VALIDATION_PRIVATE.pkl','COMPLETE_PRIVATE.json']]
    if done.exists():
        q=jr(done);assert q['binding']==digest(bind) and sha(bp)==q['best_sha256'] and sha(hp)==q['validation_sha256']
        q=read(hp);assert q['binding']==bind;return q['summary'],q['rows'],q['history']
    seed_all(seed);opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001);tr,va=d['train'],d['val']
    pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda');best=key=None;history=[];stale=0;start=1
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind;m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer'])
        core.restore_rng(q['rng']);best,key,history,stale,start=q['best'],q['key'],q['history'],q['stale'],q['epoch']+1
        if q['epoch']>=6 and stale>=6:start=31
    groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))]
    for epoch in range(start,31):
        tick=time.perf_counter();m.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];raw,mask,times,delta,_=inputs.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
            x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);logit=m(x,raw,mask,times,delta,ss)['logit'];loss=MODEL.objective(logit,y,pos);assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
            del raw,mask,times,delta,x,y,logit,loss
        sel=core.select_threshold(d['y'][va],probability(predict(m,d,va,inputs,ss)),d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        history.append(dict(arm=arm,fold=fold,epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**sel))
        atomic_torch(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING_CHANNEL_RELATION',arm=arm,fold=fold,epoch=epoch,best_epoch=best['epoch'],new_Outer_passes=0))
        print('CHANNEL_RELATION_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:break
    assert best['epoch']>=1;m.load_state_dict(best['model']);atomic_torch(bp,dict(binding=bind,**best))
    out=predict(m,d,va,inputs,ss);sel=core.select_threshold(d['y'][va],probability(out),d['patient'][va])
    assert all(abs(float(v)-float(best[k]))<1e-12 for k,v in sel.items())
    rows=patient_rows(d,va,out,best['threshold'],arm,fold)
    summary=dict(arm=arm,fold=fold,epochs=len(history),selected_epoch=best['epoch'],threshold=best['threshold'],patients=len(rows),
        parameters=sum(p.numel() for p in m.parameters()),checkpoint_sha256=sha(bp),initial_state_sha256=bind['initial'],
        execution_kind='NEW_FIT',**{k:float(np.mean([r[k] for r in rows])) for k in METRICS})
    save(hp,dict(binding=bind,archive_indices=d['archive_indices'][va],logit=out,rows=rows,summary=summary,history=history))
    write(done,dict(binding=digest(bind),best_sha256=sha(bp),validation_sha256=sha(hp)))
    del m,opt;gc.collect();torch.cuda.empty_cache();return summary,rows,history


def main():
    binding,core,fac,records,rec,banks,fr=setup();a=jr(PUB/'ADMISSION.json');assert a['status']=='PASS' and a['input_binding_sha256']==digest(binding)
    inputs=store();summaries=[];rows=[];histories=[];cells=[]
    for fold in range(1,6):
        d=bank(fold,banks);ss=scale(fold);r,state,q,out,error=replay_control(fr,fold,core,fac,records,d,inputs,ss)
        part=patient_rows(d,d['val'],out,state['threshold'],ARMS[0],fold);rows+=part
        summaries.append(dict(arm=ARMS[0],fold=fold,epochs=0,prior_training_epochs=q['summary']['epochs'],
            selected_epoch=state['epoch'],threshold=state['threshold'],patients=len(part),parameters=10378,
            checkpoint_sha256=r['checkpoint_sha256'],initial_state_sha256=q['binding']['initial'],
            execution_kind='REUSED_EXACT_C2',**{k:float(np.mean([z[k] for z in part])) for k in METRICS}))
        cells.append(dict(r,arm=ARMS[0]))
        for arm in NEW:
            summary,part,history=fit((fold,d,ss),arm,binding,inputs,core,fac,records)
            summaries.append(summary);rows+=part;histories+=history;root=ROOT/'models_private'/arm/f'fold{fold}'
            cells.append(dict(arm=arm,fold=fold,checkpoint=str(root/'BEST_PRIVATE.pt'),checkpoint_sha256=sha(root/'BEST_PRIVATE.pt'),
                validation=str(root/'VALIDATION_PRIVATE.pkl'),validation_sha256=sha(root/'VALIDATION_PRIVATE.pkl')))
            pd.DataFrame(summaries).to_csv(PUB/'DEVELOPMENT_CELLS.csv',index=False);pd.DataFrame(histories).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
            print('CHANNEL_RELATION_CELL_COMPLETE',arm,fold,flush=True)
    save(ROOT/'DEVELOPMENT_PATIENTS_PRIVATE.pkl',rows)
    freeze(ROOT/'MODEL_FREEZE_PRIVATE.json',dict(input=binding,cells=cells));selection=choose(pd.DataFrame(rows))
    freeze(ROOT/'CANDIDATE_PRIVATE.json',dict(input=binding,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),selection=selection))
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL10_NEW_PLUS5_REUSED_MODELS_FROZEN',actual_new_fits=10,reused_controls=5,
        epochs=len(histories),new_Outer_passes=0,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),
        candidate_sha256=sha(ROOT/'CANDIDATE_PRIVATE.json'),goal_established=False))
    print('CHANNEL_RELATION_ALL_TRAIN_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':main()
