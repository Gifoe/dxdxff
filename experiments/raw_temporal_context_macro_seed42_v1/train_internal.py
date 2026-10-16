"""All prescribed first-path fine-tunes and ownFIT-only bounded advancement."""
import gc
import time
from common import *
from model import Model,objective,optimizer
from admit import evaluate


def means(rows):return {m:float(np.mean([r[m] for r in rows])) for m in METRICS}
def penalized(rows):
    v=[np.mean([r['macro_f1'] for r in rows if r['center']==s]) for s in sorted(set(r['center'] for r in rows))]
    return means(rows)['macro_f1']-.1*(float(np.std(v,ddof=1)) if len(v)>1 else 0.)


def choose(results):return min(results,key=lambda r:(-r['penalized_macro'],r['parameters'],ARMS.index(r['arm'])))


def baseline(ctx,arm):
    assert arm in ARMS[:2];held=ctx['roles']['held']
    if arm=='NATIVE':logit=ctx['stored']['raw_logit_nez'][held];theta=ctx['reference_threshold'];parameters=12529
    else:logit=ctx['pre_result']['logit_nez'];theta=ctx['pre_result']['summary']['threshold'];parameters=13105
    rows=held_rows(ctx,logit,theta,arm)
    return dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,parameters=parameters,epochs=0,selected_epoch=0,threshold=theta,
                patients=len(rows),penalized_macro=penalized(rows),**means(rows)),rows


def fit(ctx,arm,binding,store):
    d=ctx['data'];rr=ctx['roles'];tr=np.r_[rr['train'],rr['choice']];cal,held=rr['cal'],rr['held'];f,s=ctx['fold'],ctx['screen']
    cell=ROOT/'models_private'/f'fold{f}_screen{s}'/arm;cell.mkdir(parents=True,exist_ok=True)
    m=Model(arm,f,ctx['reference_state'],ctx['matrix']).cuda();seed=42+1009*f;seed_all(seed)
    bind=dict(input=binding,fold=f,screen=s,arm=arm,source_body=ctx['stored']['checkpoint_sha256'],source_matrix=ctx['pre_matrix_sha256'],
              source_representation=ctx['representation_sha256'],initial_state_sha256=state_hash(m.state_dict()))
    hp,bp,last,done=cell/'HELD_PRIVATE.pkl',cell/'BEST_PRIVATE.pt',cell/'LAST_PRIVATE.pt',cell/'COMPLETE_PRIVATE.json'
    if done.exists():
        q=jr(done);assert q['binding_sha256']==digest(bind) and q['held_sha256']==sha(hp) and q['best_sha256']==sha(bp)
        result=read(hp);assert result['binding']==bind;return result['summary'],result['rows'],result['history']
    opt=optimizer(m);pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda')
    selected=calibrate(d,cal,evaluate(m,d,cal,store));assert selected['threshold']==ctx['pre_result']['summary']['threshold']
    best=dict(model=state_cpu(m),epoch=0,selection=selected);key=(selected['macro_f1'],selected['ez_f1'],0)
    stale=0;start=1;history=[]
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind
        m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);restore(q['rng'])
        best,key,stale,start,history=q['best'],q['key'],q['stale'],q['epoch']+1,q['history']
        if q['epoch']>=8 and stale>=8:start=41
    groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))]
    assert set(np.concatenate(groups))==set(tr)
    for epoch in range(start,41):
        tick=time.perf_counter();m.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];raw,mask,times=store.get(d['patient'][ii[0]],d['channel'][ii]);y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);score=m(raw,mask,times);loss=objective(score,y,pos);assert torch.isfinite(loss)
            loss.backward();assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
            del raw,mask,times,y,score,loss
        selected=calibrate(d,cal,evaluate(m,d,cal,store));candidate=(selected['macro_f1'],selected['ez_f1'],-epoch)
        if candidate>key:key=candidate;stale=0;best=dict(model=state_cpu(m),epoch=epoch,selection=selected)
        else:stale+=1
        history.append(dict(epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**selected))
        atomic_torch(last,dict(binding=bind,model=state_cpu(m),optimizer=opt.state_dict(),rng=rng(),best=best,key=key,stale=stale,epoch=epoch,history=history))
        write(PUB/'RUN_STATUS.json',dict(status='TRAINING_INTERNAL',fold=f,screen=s,arm=arm,epoch=epoch,selected_epoch=best['epoch'],last_epoch_seconds=history[-1]['seconds'],new_Outer_passes=0))
        print('RAW_TEMPORAL_EPOCH',f,s,arm,epoch,'best',best['epoch'],'seconds',round(history[-1]['seconds'],3),flush=True)
        if epoch>=8 and stale>=8:break
    m.load_state_dict(best['model']);atomic_torch(bp,dict(binding=bind,**best));cl=evaluate(m,d,cal,store)
    assert calibrate(d,cal,cl)==best['selection'];logit=evaluate(m,d,held,store);rows=held_rows(ctx,logit,best['selection']['threshold'],arm)
    summary=dict(fold=f,screen=s,arm=arm,parameters=sum(p.numel() for p in m.parameters()),epochs=len(history),selected_epoch=best['epoch'],
        threshold=best['selection']['threshold'],patients=len(rows),penalized_macro=penalized(rows),checkpoint_sha256=sha(bp),**means(rows))
    save(hp,dict(binding=bind,archive_indices=d['archive_indices'][held],cal_archive_indices=d['archive_indices'][cal],logit_nez=logit,
                 cal_logit_nez=cl,summary=summary,rows=rows,history=history))
    write(done,dict(binding_sha256=digest(bind),best_sha256=sha(bp),held_sha256=sha(hp),last_sha256=sha(last)))
    del m,opt;gc.collect();torch.cuda.empty_cache();print('RAW_TEMPORAL_CELL_COMPLETE',f,s,arm,flush=True)
    return summary,rows,history


def main():
    binding=setup();gate=jr(PUB/'ADMISSION.json');assert gate['status']=='PASS' and gate['input_binding_sha256']==digest(binding)
    cc={(q['fold'],q['screen']):q for q in contexts()};store=RawStore();summaries=[];rows=[];histories=[];cells=[];races=[]
    def record(ctx,arm):
        if arm in ARMS[:2]:summary,part=baseline(ctx,arm);hh=[]
        else:
            summary,part,hh=fit(ctx,arm,binding,store)
            p=ROOT/'models_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm/'HELD_PRIVATE.pkl'
            cells.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,sha256=sha(p)))
        summaries.append(summary);rows.extend(part);histories.extend([dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,**h) for h in hh])
        pd.DataFrame(summaries).to_csv(PUB/'INTERNAL_CELL_RESULTS.csv',index=False);pd.DataFrame(histories).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
        return summary
    for fold in range(1,6):
        first=[record(cc[(fold,1)],arm) for arm in ARMS];candidate=choose(first[2:]);base=first[:2]
        advanced=candidate['macro_f1']>=max(r['macro_f1'] for r in base)+.01 and candidate['auroc']>=max(r['auroc'] for r in base)-.01
        race=dict(fold=fold,chosen=candidate['arm'],advanced=advanced,Macro_gain=candidate['macro_f1']-max(r['macro_f1'] for r in base),
                  AUROC_gain=candidate['auroc']-max(r['auroc'] for r in base),first_path_ownFIT_only=True)
        freeze(ROOT/'races_private'/f'fold{fold}.json',race);races.append(race)
        if advanced:
            for screen in [2,3]:
                for arm in ARMS:record(cc[(fold,screen)],arm)
        write(PUB/'RACING_DECISIONS.json',dict(status='INTERNAL_RACING',decisions=races))
    assert 15<=len(cells)<=45;save(ROOT/'INTERNAL_PATIENT_METRICS_PRIVATE.pkl',rows)
    freeze(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json',dict(input=binding,cells=cells,races=races))
    write(PUB/'RUN_STATUS.json',dict(status='ALL_REGISTERED_TEMPORAL_INTERNAL_COMPLETE',actual_neural_fits=len(cells),logical_cells=len(summaries),
        epochs=sum(r['epochs'] for r in summaries),score_seal_sha256=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),new_Outer_passes=0,goal_established=False))
    print('RAW_TEMPORAL_ALL_INTERNAL_COMPLETE',len(cells),flush=True)


if __name__=='__main__':main()
