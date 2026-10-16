"""Bounded all-three-arm information comparison; no Outer prediction loader."""
import gc
import time
from common import *
from model import Model,objective
from admit import evaluate


def means(rows):return {m:float(np.mean([r[m] for r in rows])) for m in METRICS}
def parameters(arm):return 12529 if arm=='NATIVE' else 13105 if arm=='FROZEN_PRE' else 14577
def penalized(rows):
    values=[np.mean([r['macro_f1'] for r in rows if r['center']==s]) for s in sorted(set(r['center'] for r in rows))]
    return means(rows)['macro_f1']-.1*(float(np.std(values,ddof=1)) if len(values)>1 else 0.)
def choose(results):return min(results,key=lambda r:(-r['penalized_macro'],r['parameters'],ARMS.index(r['arm'])))


def baseline(ctx,arm):
    x,theta=(ctx['native_logits'],ctx['native_threshold']) if arm=='NATIVE' else (ctx['pre_logits'],ctx['pre_threshold'])
    rows=held_rows(ctx,x,theta,arm)
    return dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,parameters=parameters(arm),epochs=0,selected_epoch=0,
        threshold=theta,patients=len(rows),penalized_macro=penalized(rows),**means(rows)),rows


def fit(ctx,arm,binding,store):
    d=ctx['data'];rr=ctx['roles'];fold,screen=ctx['fold'],ctx['screen'];tr=np.r_[rr['train'],rr['choice']];cal,held=rr['cal'],rr['held']
    cell=ROOT/'models_private'/f'fold{fold}_screen{screen}'/arm;cell.mkdir(parents=True,exist_ok=True)
    model=Model(arm,fold).cuda();seed=42+1009*fold;seed_all(seed)
    bind=dict(input=binding,fold=fold,screen=screen,arm=arm,initial_state_sha256=state_hash(model.state_dict()),
        role_representation_sha256=ctx['representation_sha256'],initialization='RANDOM_NO_WARMSTART')
    hp,bp,last,done=cell/'HELD_PRIVATE.pkl',cell/'BEST_PRIVATE.pt',cell/'LAST_PRIVATE.pt',cell/'COMPLETE_PRIVATE.json'
    if done.exists():
        q=jr(done);assert q['binding_sha256']==digest(bind) and q['held_sha256']==sha(hp) and q['best_sha256']==sha(bp)
        result=read(hp);assert result['binding']==bind;return result['summary'],result['rows'],result['history']
    opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
    pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda');best=key=None;history=[];stale=0;start=1
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind
        model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);restore(q['rng'])
        best,key,history,stale,start=q['best'],q['key'],q['history'],q['stale'],q['epoch']+1
        if q['epoch']>=15 and stale>=15:start=101
    groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))];assert set(np.concatenate(groups))==set(tr)
    for epoch in range(start,101):
        tick=time.perf_counter();model.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];raw,mask,times=store.get(d['patient'][ii[0]],d['channel'][ii]);y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);logit=model(raw,mask,times);loss=objective(logit,y,pos);assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
            del raw,mask,times,y,logit,loss
        selected=calibrate(d,cal,evaluate(model,d,cal,store));candidate=(selected['macro_f1'],selected['ez_f1'],-epoch)
        if key is None or candidate>key:key=candidate;stale=0;best=dict(model=state_cpu(model),epoch=epoch,selection=selected)
        else:stale+=1
        history.append(dict(epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**selected))
        atomic_torch(last,dict(binding=bind,model=state_cpu(model),optimizer=opt.state_dict(),rng=rng(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        write(PUB/'RUN_STATUS.json',dict(status='TRAINING_JOINT_RAW_INTERNAL',fold=fold,screen=screen,arm=arm,epoch=epoch,
            selected_epoch=best['epoch'],last_epoch_seconds=history[-1]['seconds'],original_Development_Outer_used=False))
        print('JOINT_RAW_EPOCH',fold,screen,arm,epoch,'best',best['epoch'],'seconds',round(history[-1]['seconds'],3),flush=True)
        if epoch>=15 and stale>=15:break
    assert best is not None and best['epoch']>=1
    model.load_state_dict(best['model']);atomic_torch(bp,dict(binding=bind,**best));cl=evaluate(model,d,cal,store)
    assert calibrate(d,cal,cl)==best['selection'];logit=evaluate(model,d,held,store);rows=held_rows(ctx,logit,best['selection']['threshold'],arm)
    summary=dict(fold=fold,screen=screen,arm=arm,parameters=parameters(arm),epochs=len(history),selected_epoch=best['epoch'],threshold=best['selection']['threshold'],
        patients=len(rows),penalized_macro=penalized(rows),checkpoint_sha256=sha(bp),**means(rows))
    save(hp,dict(binding=bind,archive_indices=d['archive_indices'][held],cal_archive_indices=d['archive_indices'][cal],logit_nez=logit,
        cal_logit_nez=cl,summary=summary,rows=rows,history=history))
    write(done,dict(binding_sha256=digest(bind),held_sha256=sha(hp),best_sha256=sha(bp),last_sha256=sha(last)))
    del model,opt;gc.collect();torch.cuda.empty_cache();print('JOINT_RAW_CELL_COMPLETE',fold,screen,arm,flush=True)
    return summary,rows,history


def main():
    binding=setup();admission=jr(PUB/'ADMISSION.json');assert admission['status']=='PASS' and admission['input_binding_sha256']==digest(binding)
    cc={(r['fold'],r['screen']):r for r in contexts()};store=RawStore();summaries=[];rows=[];histories=[];seals=[];races=[]
    def record(ctx,arm):
        if arm in REFERENCES:summary,part=baseline(ctx,arm);history=[]
        else:
            summary,part,history=fit(ctx,arm,binding,store)
            path=ROOT/'models_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm/'HELD_PRIVATE.pkl'
            seals.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,sha256=sha(path)))
        summaries.append(summary);rows.extend(part);histories.extend([dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,**r) for r in history])
        pd.DataFrame(summaries).to_csv(PUB/'INTERNAL_CELL_RESULTS.csv',index=False);pd.DataFrame(histories).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
        return summary
    for fold in range(1,6):
        first=[record(cc[(fold,1)],arm) for arm in ARMS];candidate=choose(first[2:]);base=first[:2]
        advanced=candidate['macro_f1']>=max(r['macro_f1'] for r in base)+.01 and candidate['auroc']>=max(r['auroc'] for r in base)-.01
        race=dict(fold=fold,chosen=candidate['arm'],advanced=advanced,promoted=TRAINED if advanced else [],
            Macro_gain=candidate['macro_f1']-max(r['macro_f1'] for r in base),AUROC_gain=candidate['auroc']-max(r['auroc'] for r in base))
        freeze(ROOT/'races_private'/f'fold{fold}.json',race);races.append(race)
        if advanced:
            for screen in [2,3]:
                for arm in ARMS:record(cc[(fold,screen)],arm)
        write(PUB/'RACING_DECISIONS.json',dict(status='JOINT_RAW_INTERNAL_RACING',decisions=races))
    assert 15<=len(seals)<=45;save(ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl',rows)
    freeze(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json',dict(input=binding,cells=seals,races=races))
    write(PUB/'RUN_STATUS.json',dict(status='ALL_REGISTERED_JOINT_RAW_INTERNAL_COMPLETE',actual_neural_fits=len(seals),logical_cells=len(summaries),
        epochs=sum(r['epochs'] for r in summaries),score_seal_sha256=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),new_Outer_passes=0,goal_established=False))
    print('JOINT_RAW_ALL_INTERNAL_COMPLETE',len(seals),flush=True)


if __name__=='__main__':main()
