"""Bounded warm end-to-end Raw fine-tuning; ownFIT-only racing."""
import argparse
import gc
import time
from common import *
from model import Model,ARMS,TRAINED,objective,optimizer
from admit import evaluate


def aggregate(rows):return {m:float(np.mean([r[m] for r in rows])) for m in METRICS}


def penalized(rows):
    means=[np.mean([r['macro_f1'] for r in rows if r['center']==s]) for s in sorted(set(r['center'] for r in rows))]
    return aggregate(rows)['macro_f1']-.1*(float(np.std(means,ddof=1)) if len(means)>1 else 0.)


def native(ctx,binding):
    d=ctx['data'];rr=ctx['roles'];cal=rr['cal'];logit=ctx['stored']['raw_logit_nez'];theta=ctx['reference_threshold']
    assert calibrate(d,cal,logit[cal])['threshold']==theta
    rows=held_rows(ctx,logit[rr['held']],theta,'NATIVE')
    return dict(fold=ctx['fold'],screen=ctx['screen'],arm='NATIVE',epochs=0,selected_epoch=0,parameters=0,
                threshold=theta,patients=len(rows),penalized_macro=penalized(rows),**aggregate(rows)),rows


def fit(ctx,arm,binding,store):
    assert arm in TRAINED
    d=ctx['data'];rr=ctx['roles'];fold=ctx['fold'];screen=ctx['screen'];tr=np.r_[rr['train'],rr['choice']];cal=rr['cal'];held=rr['held']
    cell=ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm;cell.mkdir(parents=True,exist_ok=True)
    model=Model(arm,fold,ctx['reference_state']).cuda();seed=42+1009*fold;seed_all(seed)
    bind=dict(input=binding,fold=fold,screen=screen,arm=arm,reference_checkpoint_sha256=ctx['stored']['checkpoint_sha256'],
              role_representation_sha256=ctx['representation_sha256'],initial_sha256=state_hash(model.state_dict()))
    done=cell/'COMPLETE_PRIVATE.json';hp=cell/'HELD_PRIVATE.pkl';bp=cell/'BEST_PRIVATE.pt';last=cell/'LAST_PRIVATE.pt'
    if done.exists():
        q=jr(done);assert q['binding_sha256']==digest(bind) and q['score_sha256']==sha(hp) and q['best_sha256']==sha(bp)
        result=read(hp);assert result['binding']==bind
        return result['summary'],result['rows'],result['history']
    opt=optimizer(model);pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda')
    selected=calibrate(d,cal,evaluate(model,d,cal,store));best=dict(model=state_cpu(model),epoch=0,selection=selected)
    key=(selected['macro_f1'],selected['ez_f1'],0);stale=0;start=1;history=[]
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind
        model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);restore(q['rng'])
        best,key,stale,start,history=q['best'],q['key'],q['stale'],q['epoch']+1,q['history']
        if q['epoch']>=5 and stale>=5:start=21
    groups=[tr[d['patient'][tr]==p] for p in sorted(set(d['patient'][tr]))]
    for epoch in range(start,21):
        tick=time.perf_counter();losses=[];model.train()
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];raw,mask,times=store.get(d['patient'][ii[0]],d['channel'][ii])
            y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32);opt.zero_grad(set_to_none=True)
            logit=model(raw,mask,times);loss=objective(logit,y,pos);assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
            del raw,mask,times,y,logit,loss
        selected=calibrate(d,cal,evaluate(model,d,cal,store));candidate=(selected['macro_f1'],selected['ez_f1'],-epoch)
        if candidate>key:key=candidate;stale=0;best=dict(model=state_cpu(model),epoch=epoch,selection=selected)
        else:stale+=1
        history.append(dict(epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**selected))
        atomic_torch(last,dict(binding=bind,model=state_cpu(model),optimizer=opt.state_dict(),rng=rng(),best=best,key=key,
                               stale=stale,epoch=epoch,history=history))
        write(PUB/'RUN_STATUS.json',dict(status='TRAINING_INTERNAL',fold=fold,screen=screen,arm=arm,epoch=epoch,
                                       selected_epoch=best['epoch'],last_epoch_seconds=history[-1]['seconds'],new_Outer_access=False))
        print('DISPERSION_EPOCH',fold,screen,arm,epoch,'seconds',round(history[-1]['seconds'],2),'best',best['epoch'],flush=True)
        if epoch>=5 and stale>=5:break
    model.load_state_dict(best['model']);atomic_torch(bp,dict(binding=bind,**best))
    cal_logit=evaluate(model,d,cal,store);assert calibrate(d,cal,cal_logit)==best['selection']
    logit=evaluate(model,d,held,store);rows=held_rows(ctx,logit,best['selection']['threshold'],arm)
    summary=dict(fold=fold,screen=screen,arm=arm,epochs=len(history),selected_epoch=best['epoch'],parameters=sum(p.numel() for p in model.parameters()),
                 threshold=best['selection']['threshold'],patients=len(rows),penalized_macro=penalized(rows),
                 checkpoint_sha256=sha(bp),**aggregate(rows))
    save(hp,dict(binding=bind,archive_indices=d['archive_indices'][held],logit_nez=logit,cal_logit_nez=cal_logit,
                 cal_archive_indices=d['archive_indices'][cal],summary=summary,rows=rows,history=history))
    write(done,dict(binding_sha256=digest(bind),best_sha256=sha(bp),score_sha256=sha(hp),last_sha256=sha(last)))
    del model,opt;gc.collect();torch.cuda.empty_cache()
    print('DISPERSION_CELL_COMPLETE',fold,screen,arm,flush=True)
    return summary,rows,history


def choose(results):
    maximum=max(r['penalized_macro'] for r in results)
    return min([r for r in results if r['penalized_macro']>=maximum-.005],key=lambda r:(r['parameters'],-r['penalized_macro'],ARMS.index(r['arm'])))


def run():
    binding=setup();gate=jr(PUB/'ADMISSION.json');assert gate['status']=='PASS' and gate['input_binding_sha256']==digest(binding)
    cc={(q['fold'],q['screen']):q for q in contexts()};store=RawStore();summaries=[];rows=[];histories=[];races=[];seals=[]
    def record(ctx,arm):
        if arm=='NATIVE':summary,rr=native(ctx,binding);hh=[]
        else:
            summary,rr,hh=fit(ctx,arm,binding,store)
            p=ROOT/'models_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm/'HELD_PRIVATE.pkl'
            seals.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,sha256=sha(p)))
        summaries.append(summary);rows.extend(rr)
        histories.extend([dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,**h) for h in hh])
        pd.DataFrame(summaries).to_csv(PUB/'INTERNAL_CELL_RESULTS.csv',index=False)
        pd.DataFrame(histories).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
        return summary
    for fold in range(1,6):
        ctx=cc[(fold,1)];first=[record(ctx,arm) for arm in ARMS];base=first[0];candidate=choose(first[1:])
        advanced=candidate['macro_f1']>=base['macro_f1']+.01 and candidate['auroc']>=base['auroc']-.01
        decision=dict(fold=fold,chosen=candidate['arm'],advanced=advanced,Macro_gain=candidate['macro_f1']-base['macro_f1'],
                      AUROC_gain=candidate['auroc']-base['auroc'],first_path_selection_only=True)
        freeze(ROOT/'race_private'/f'fold{fold}.json',decision);races.append(decision)
        if advanced:
            for screen in [2,3]:
                for arm in dict.fromkeys(['NATIVE','MEAN_TUNE',candidate['arm']]):record(cc[(fold,screen)],arm)
        write(PUB/'RACING_DECISIONS.json',dict(status='RACING_INTERNAL',decisions=races))
    actual=sum(r['arm']!='NATIVE' for r in summaries);assert 15<=actual<=35
    save(ROOT/'INTERNAL_PATIENT_ROWS_PRIVATE.pkl',rows)
    freeze(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json',dict(input=binding,cells=seals,races=races))
    write(PUB/'RUN_STATUS.json',dict(status='ALL_REGISTERED_INTERNAL_CELLS_COMPLETE',actual_trainable_fits=actual,
          logical_cells=len(summaries),epochs=sum(r['epochs'] for r in summaries),new_Outer_access=False,
          seal_sha256=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),goal_established=False))
    print('DISPERSION_ALL_INTERNAL_TRAIN_COMPLETE',actual,flush=True)


if __name__=='__main__':run()
