"""Full-patient Macro-F1 optimization of576parameters only; no body/head updates."""
import copy
import gc
import time
from common import *
from model import Model,ARMS,TRAINED,objective


def load_context(ctx,binding):
    d=ctx['data'];seal=jr(ROOT/'PHASE_VIEWS_SEAL_PRIVATE.json');assert seal['input']==binding
    result={}
    for pid in sorted(set(d['patient'])):
        ix=np.flatnonzero(d['patient']==pid);path=path_for(ctx,pid)
        assert sha(path)==seal['parts'][str(path.relative_to(ROOT))]
        q=read(path);assert q['binding']['input']==binding and q['binding']['archive_indices']==d['archive_indices'][ix].tolist()
        assert q['binding']['checkpoint_sha256']==ctx['stored']['checkpoint_sha256']
        assert q['complete_patient_channels'] and q['no_labels_or_manual_features_used']
        result[str(pid)]=payload(q)
    return result


def predict(model,ctx,ix,caches):
    model.eval();d=ctx['data'];out=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            loc=np.flatnonzero(d['patient'][ix]==pid)
            assert len(loc)==int((d['patient']==pid).sum())
            out[loc]=model.forward_stats(caches[str(pid)]).cpu().numpy()
    assert np.isfinite(out).all();return out


def means(rows):return {m:float(np.mean([r[m] for r in rows])) for m in METRICS}


def native(ctx):
    ix=ctx['roles']['held'];theta=ctx['reference_threshold'];logit=ctx['stored']['raw_logit_nez'][ix]
    rows=held_rows(ctx,logit,theta,'NATIVE')
    return dict(fold=ctx['fold'],screen=ctx['screen'],arm='NATIVE',epochs=0,selected_epoch=0,parameters=0,
                threshold=theta,patients=len(rows),**means(rows)),rows


def fit(ctx,arm,binding,caches):
    d=ctx['data'];rr=ctx['roles'];fold=ctx['fold'];screen=ctx['screen'];tr=np.r_[rr['train'],rr['choice']];cal=rr['cal'];held=rr['held']
    cell=ROOT/'matrices_private'/f'fold{fold}_screen{screen}'/arm;cell.mkdir(parents=True,exist_ok=True)
    model=Model(arm,fold,ctx['reference_state']).cuda()
    bind=dict(input=binding,fold=fold,screen=screen,arm=arm,phase_seal_sha256=sha(ROOT/'PHASE_VIEWS_SEAL_PRIVATE.json'),
              body_checkpoint_sha256=ctx['stored']['checkpoint_sha256'],role_representation_sha256=ctx['representation_sha256'])
    hp=cell/'HELD_PRIVATE.pkl';bp=cell/'BEST_PRIVATE.pt';last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json'
    if done.exists():
        q=jr(done);assert q['binding_sha256']==digest(bind) and q['best_sha256']==sha(bp) and q['held_sha256']==sha(hp)
        saved=read(hp);assert saved['binding']==bind
        return saved['summary'],saved['rows'],saved['history']
    original_body=state_hash({k:v for k,v in model.state_dict().items() if k!='matrix'})
    assert sum(p.numel() for p in model.parameters() if p.requires_grad)==576
    opt=torch.optim.AdamW([model.matrix],lr=.003,weight_decay=.01)
    pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()),device='cuda')
    groups=[np.flatnonzero(d['patient']==p) for p in sorted(set(d['patient'][tr]))]
    assert set(np.concatenate(groups))==set(tr)
    truth=[torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32) for ii in groups]
    selected=calibrate(d,cal,predict(model,ctx,cal,caches))
    assert selected['threshold']==ctx['reference_threshold']
    best=dict(matrix=model.matrix.detach().cpu().clone(),epoch=0,selection=selected)
    key=(selected['macro_f1'],selected['ez_f1'],0);stale=0;start=1;history=[]
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind
        with torch.no_grad():model.matrix.copy_(q['matrix'])
        opt.load_state_dict(q['optimizer']);best,key,stale,start,history=q['best'],q['key'],q['stale'],q['epoch']+1,q['history']
        if q['epoch']>=30 and stale>=30:start=301
    for epoch in range(start,301):
        tick=time.perf_counter();model.train();opt.zero_grad(set_to_none=True);losses=[]
        for ii,y in zip(groups,truth):
            score=model.forward_stats(caches[str(d['patient'][ii[0]])]);losses.append(objective(score,y,pos))
        loss=torch.stack(losses).mean();assert torch.isfinite(loss);loss.backward()
        assert model.matrix.grad is not None and torch.isfinite(model.matrix.grad).all()
        assert all(p.grad is None for p in model.cnn.parameters()) and all(p.grad is None for p in model.head.parameters())
        torch.nn.utils.clip_grad_norm_([model.matrix],1.);opt.step()
        selected=calibrate(d,cal,predict(model,ctx,cal,caches));candidate=(selected['macro_f1'],selected['ez_f1'],-epoch)
        if candidate>key:key=candidate;stale=0;best=dict(matrix=model.matrix.detach().cpu().clone(),epoch=epoch,selection=selected)
        else:stale+=1
        history.append(dict(epoch=epoch,seconds=time.perf_counter()-tick,loss=float(loss.detach().cpu()),**selected))
        atomic_torch(last,dict(binding=bind,matrix=model.matrix.detach().cpu(),optimizer=opt.state_dict(),best=best,key=key,
                               stale=stale,epoch=epoch,history=history))
        if epoch%25==0:print('PRECOMPAND_EPOCH',fold,screen,arm,epoch,'best',best['epoch'],flush=True)
        if epoch>=30 and stale>=30:break
    assert original_body==state_hash({k:v for k,v in model.state_dict().items() if k!='matrix'})
    with torch.no_grad():model.matrix.copy_(best['matrix'])
    atomic_torch(bp,dict(binding=bind,**best));cal_logit=predict(model,ctx,cal,caches)
    assert calibrate(d,cal,cal_logit)==best['selection']
    logit=predict(model,ctx,held,caches);rows=held_rows(ctx,logit,best['selection']['threshold'],arm)
    summary=dict(fold=fold,screen=screen,arm=arm,epochs=len(history),selected_epoch=best['epoch'],parameters=576,
                 total_model_parameters=13105,threshold=best['selection']['threshold'],patients=len(rows),**means(rows))
    save(hp,dict(binding=bind,logit_nez=logit,archive_indices=d['archive_indices'][held],cal_logit_nez=cal_logit,
                 cal_archive_indices=d['archive_indices'][cal],summary=summary,rows=rows,history=history))
    write(done,dict(binding_sha256=digest(bind),best_sha256=sha(bp),held_sha256=sha(hp),body_head_unchanged=True))
    del model,opt;gc.collect();torch.cuda.empty_cache()
    print('PRECOMPAND_MATRIX_COMPLETE',fold,screen,arm,len(history),'selected',best['epoch'],flush=True)
    return summary,rows,history


def main():
    binding=setup();gate=jr(PUB/'ADMISSION.json');assert gate['status']=='PASS' and gate['input_binding_sha256']==digest(binding)
    summaries=[];rows=[];history=[];seals=[]
    for ctx in contexts():
        caches=load_context(ctx,binding)
        for arm in ARMS:
            if arm=='NATIVE':summary,part=native(ctx);hh=[]
            else:
                summary,part,hh=fit(ctx,arm,binding,caches)
                path=ROOT/'matrices_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm/'HELD_PRIVATE.pkl'
                seals.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,sha256=sha(path)))
            summaries.append(summary);rows.extend(part);history.extend([dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,**h) for h in hh])
            pd.DataFrame(summaries).to_csv(PUB/'INTERNAL_CELL_RESULTS.csv',index=False)
            pd.DataFrame(history).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
            write(PUB/'RUN_STATUS.json',dict(status='FITTING_MATRICES',logical_cells=len(summaries),total=45,body_head_updates=0,
                                            original_Development_Outer_used=False))
        del caches;gc.collect();torch.cuda.empty_cache()
    assert len(seals)==30 and len(summaries)==45
    save(ROOT/'INTERNAL_PATIENT_ROWS_PRIVATE.pkl',rows)
    freeze(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json',dict(input=binding,cells=seals))
    write(PUB/'RUN_STATUS.json',dict(status='ALL30_MATRIX_FITS_COMPLETE',actual_matrix_fits=30,logical_cells=45,
          epochs=sum(r['epochs'] for r in summaries),seal_sha256=sha(ROOT/'INTERNAL_RESULTS_SEAL_PRIVATE.json'),
          original_Development_Outer_used=False,goal_established=False))
    print('PRECOMPAND_ALL30_MATRIX_FITS_COMPLETE',flush=True)


if __name__=='__main__':main()
