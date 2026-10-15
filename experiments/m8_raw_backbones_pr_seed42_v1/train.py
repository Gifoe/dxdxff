"""Train paired raw-only models on original FIT/VAL; no Outer loader."""
import gc
import time
from support import *
from engine import forward,rng,restore


def validation(model,d,store,chunk,pos):
    model.eval();va=d['val'];score=np.empty(len(va),np.float32);loss=[]
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][va])):
            k=np.flatnonzero(d['patient'][va]==pid);ii=va[k]
            raw,mask=raw_patient(store,pid,d['channel'][ii]);ll,h=forward(model,raw,mask,chunk)
            score[k]=1-torch.sigmoid(ll).cpu().numpy()
            y=torch.tensor(d['y'][ii],dtype=torch.float32,device='cuda')
            loss.append(float(torch.nn.functional.binary_cross_entropy_with_logits(ll,y,pos_weight=pos)))
    sel=EV.threshold(1-d['y'][va],score,d['patient'][va])
    rows=[]
    for pid in sorted(set(d['patient'][va])):
        k=np.flatnonzero(d['patient'][va]==pid);ii=va[k]
        rows.append(EV.classification(1-d['y'][ii],score[k],score[k]>=sel['threshold']))
    assert np.isclose(np.mean([v['macro_f1'] for v in rows]),sel['validation_macro_f1'],rtol=0,atol=1e-12)
    return dict(**sel,loss=float(np.mean(loss)),auroc=EV.nanmean([v['auroc'] for v in rows]),auprc=EV.nanmean([v['auprc'] for v in rows])),score


def run(a):
    lock,core,rec,banks,binding=setup(a.runtime)
    smoke=jr(a.runtime/'public/SMOKE_TEST_AUDIT.json');assert smoke['status']=='PASS' and smoke['binding_sha256']==digest(binding)
    pub=a.runtime/'public';store=Store(BASELINE);selections=[];initials=[]
    for version in lock['versions']:
        for fold,d in enumerate(banks,1):
            model=build(a.model,version=='PR',fold,lock,core);initial=statehash(model.state_dict());initials.append(dict(version=version,fold=fold,sha256=initial))
            if version=='PR':assert initial==next(r['sha256'] for r in initials if r['version']=='Original' and r['fold']==fold)
            cell=a.runtime/a.model/version/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True)
            bind=dict(input=binding,model=a.model,version=version,fold=fold,initial=initial)
            opt=torch.optim.AdamW(model.parameters(),lr=lock['lr'][a.model],weight_decay=lock['weight_decay'],fused=True)
            core.seed_all(42+1009*fold);best=None;key=None;history=[];stale=0;start=1
            last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json'
            if last.exists():
                q=torch.load(last,map_location='cuda',weights_only=False);assert q['binding']==bind
                model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);restore(q['rng']);best=q['best'];key=q['key'];history=q['history'];stale=q['stale'];start=q['epoch']+1
                if done.exists():
                    cc=jr(done);assert cc['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and cc['last_sha256']==sha(last)
                    assert cc['binding_sha256']==digest(bind)
                    start=lock['max_epochs']+1
                elif q['epoch']>=lock['min_epochs'] and stale>=lock['patience']:start=lock['max_epochs']+1
            tr=d['train'];groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
            pos=torch.tensor(float((1-d['y'][tr]).sum())/max(float(d['y'][tr].sum()),1),device='cuda')
            chunk=lock['chunk'][a.model]
            for epoch in range(start,lock['max_epochs']+1):
                t=time.perf_counter();model.train();losses=[]
                for n,j in enumerate(np.random.default_rng(42+1009*fold+epoch).permutation(len(groups)),1):
                    ii=groups[int(j)];pid=str(d['patient'][ii[0]]);raw,mask=raw_patient(store,pid,d['channel'][ii]);y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
                    opt.zero_grad(set_to_none=True);loss,_=forward(model,raw,mask,chunk,True,y,pos)
                    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
                    torch.nn.utils.clip_grad_norm_(model.parameters(),lock['gradient_clip_norm']);opt.step();model.constraints();losses.append(loss)
                    if n%10==0:write(pub/'RUN_STATUS.json',dict(status='TRAINING',model=a.model,version=version,fold=fold,epoch=epoch,patients_complete=n,patients_total=len(groups),new_Outer_access=False))
                selected,score=validation(model,d,store,chunk,pos)
                candidate=(selected['validation_macro_f1'],selected['validation_positive_f1'],-epoch)
                history.append(dict(model=a.model,version=version,fold=fold,epoch=epoch,train_loss=float(np.mean(losses)),seconds=time.perf_counter()-t,**selected))
                if key is None or candidate>key:
                    key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,selection=selected)
                else:stale+=1
                atomic_torch(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),best=best,key=key,stale=stale,history=history,epoch=epoch))
                pd.DataFrame(history).to_csv(cell/'HISTORY_PRIVATE.csv',index=False)
                print('RAW_EPOCH',a.model,version,fold,epoch,'seconds',round(history[-1]['seconds'],2),'best',best['epoch'],flush=True)
                if epoch>=lock['min_epochs'] and stale>=lock['patience']:break
            assert best is not None
            if not done.exists():
                atomic_torch(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best))
                write(done,dict(binding_sha256=digest(bind),best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last),epoch=best['epoch'],threshold=best['selection']['threshold']))
            model.load_state_dict(best['model']);selected,score=validation(model,d,store,chunk,pos)
            assert selected['threshold']==best['selection']['threshold']
            save(cell/'VALIDATION_PRIVATE.pkl',dict(score_ez=score,indices=d['archive_indices'][d['val']],threshold=selected['threshold'],checkpoint_sha256=sha(cell/'BEST_PRIVATE.pt')))
            selections.append(dict(model=a.model,version=version,fold=fold,checkpoint=str(cell/'BEST_PRIVATE.pt'),checkpoint_sha256=sha(cell/'BEST_PRIVATE.pt'),initial_sha256=initial,threshold=selected['threshold'],epoch=best['epoch'],epochs_run=len(history),parameters=sum(p.numel() for p in model.parameters()),trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            print('RAW_TRAIN_CELL_COMPLETE',a.model,version,fold,flush=True)
            del model,opt;gc.collect();torch.cuda.empty_cache()
    assert len(selections)==10
    write(a.runtime/a.model/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=selections))
    pd.DataFrame([{k:v for k,v in q.items() if k!='checkpoint'} for q in selections]).to_csv(pub/f'{a.model}_MODEL_SELECTION.csv',index=False)
    write(pub/f'{a.model}_INITIALIZATION_AUDIT.json',dict(status='PASS',paired_initial_states_identical=True,initials=initials))
    print('RAW_BACKBONE_TRAIN_COMPLETE',a.model,flush=True)


if __name__=='__main__':run(arguments())
