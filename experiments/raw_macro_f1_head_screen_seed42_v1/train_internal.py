"""Bounded Macro-F1 readout fitting on genuinely cross-fitted Raw features."""
import argparse
import copy
import gc
import time
from common import *
from head import Head,ARMS,RECIPES,groups,objective


def predict(model,z,patient):
    model.eval()
    with torch.inference_mode():return model(torch.as_tensor(z),groups(patient)).numpy()


def select(d,ix,logit):
    return core.select_threshold(d['y'][ix],probability(logit),d['patient'][ix])


def admit():
    binding=setup();reports=[]
    for ctx in contexts():
        z,cp=inputs(ctx);d=ctx['data'];rr=ctx['roles'];raw=z['1111'];fold=ctx['fold']
        models=[Head(a,cp['model'],fold) for a in ['REFIT','PATIENT','LOCAL']]
        scores=[predict(m,raw,d['patient']) for m in models]
        assert np.array_equal(scores[0],scores[1]) and np.array_equal(scores[0],scores[2])
        error=float(np.max(np.abs(scores[0]-ctx['stored']['raw_logit_nez'])))
        assert error<=1e-6,(fold,ctx['screen'],error)
        cal=rr['cal'];grid=np.round(np.arange(0,1.0001,.005),3)
        a=probability(scores[0][cal]);b=probability(ctx['stored']['raw_logit_nez'][cal])
        assert np.array_equal(a[:,None]>=grid,b[:,None]>=grid)
        assert select(d,cal,scores[0][cal])['threshold']==cp['threshold']
        # Complete-patient context is independent of unrelated patients in the batch.
        pid=sorted(set(d['patient']))[0];ix=np.flatnonzero(d['patient']==pid)
        m=models[1];test=copy.deepcopy(m)
        with torch.no_grad():test.conditioner.weight.fill_(.001)
        full=predict(test,raw,d['patient']);one=predict(test,raw[ix],d['patient'][ix])
        assert np.max(np.abs(full[ix]-one))<=1e-6
        if not reports:
            tr=np.r_[rr['train'],rr['choice']];g=groups(d['patient'][tr]);y=torch.as_tensor(d['y'][tr],dtype=torch.float32)
            pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()))
            native=models[0];other=copy.deepcopy(native)
            score=native(torch.from_numpy(raw[tr]),g);loss=objective(score,y,g,pos);loss.backward()
            ss=other(torch.from_numpy(raw[tr]),g)
            explicit=torch.stack([objective(ss[g==j],y[g==j],torch.zeros(int((g==j).sum()),dtype=torch.long),pos) for j in range(int(g.max())+1)]).mean()
            explicit.backward();maximum=0.
            assert abs(float(loss.detach())-float(explicit.detach()))<=1e-6
            for p,q in zip(native.parameters(),other.parameters()):
                maximum=max(maximum,float((p.grad-q.grad).abs().max()))
                assert torch.allclose(p.grad,q.grad,atol=1e-6,rtol=1e-5)
            capacity={a:sum(p.numel() for p in Head(a,cp['model'],fold).parameters()) for a in ARMS}
            assert capacity['REFIT']==10369 and capacity['PATIENT']==capacity['LOCAL']==14097
        reports.append(dict(fold=fold,screen=ctx['screen'],native_CPU_logit_max_error=error,
                            all_calibration_grid_decisions_identical=True,zero_conditioner_identity=True,
                            complete_patient_context_isolation=True,role_and_source_hashes_verified=True))
        print('MACRO_HEAD_ADMISSION',fold,ctx['screen'],error,flush=True)
        del z,cp,models;gc.collect()
    write(PUB/'ADMISSION.json',dict(status='PASS',input_binding_sha256=digest(binding),paths=len(reports),reports=reports,
        parameters=capacity,vectorized_patient_loss_gradient_max_error=maximum,
        no_new_Raw_encoder_fits=True,only96_Raw_inputs=True,no_original_Development_or_Outer_used=True))
    print('MACRO_HEAD_ALL15_ADMISSION_PASS',flush=True)


def fit(ctx,arm,z,cp,binding):
    d,rr=ctx['data'],ctx['roles'];fold,screen=ctx['fold'],ctx['screen']
    cell=ROOT/'heads_private'/f'fold{fold}_screen{screen}'/arm;cell.mkdir(parents=True,exist_ok=True)
    model=Head(arm,cp['model'],fold);initial=state_hash(model.state_dict())
    bind=dict(input=binding,representation=ctx['representation_sha256'],encoder_checkpoint=ctx['stored']['checkpoint_sha256'],
              fold=fold,screen=screen,arm=arm,recipe=RECIPES[arm],initial=initial)
    scores_path=cell/'SCORES_PRIVATE.pkl';done=cell/'COMPLETE_PRIVATE.json';best_path=cell/'BEST_PRIVATE.pt'
    if done.exists():
        finish=jr(done);assert finish['binding_sha256']==digest(bind)
        assert finish['best_sha256']==sha(best_path) and finish['score_sha256']==sha(scores_path)
        return read(scores_path)
    tr=np.r_[rr['train'],rr['choice']];cal=rr['cal'];x=torch.from_numpy(z[tr]);g=groups(d['patient'][tr])
    y=torch.as_tensor(d['y'][tr],dtype=torch.float32);pos=torch.tensor(float((1-d['y'][tr]).sum())/float(d['y'][tr].sum()))
    initial_logit=predict(model,z[cal],d['patient'][cal]);selection=select(d,cal,initial_logit)
    best=dict(model=copy.deepcopy(model.state_dict()),epoch=0,selection=selection)
    key=(selection['macro_f1'],selection['ez_f1'],0);stale=0;history=[];start=1
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001);last=cell/'LAST_PRIVATE.pt'
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind
        model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer'])
        best,key,stale,history,start=q['best'],q['key'],q['stale'],q['history'],q['epoch']+1
        if q['epoch']>=10 and stale>=20:start=201
    if arm=='NATIVE':
        start=201
        best['selection']=select(d,cal,ctx['stored']['raw_logit_nez'][cal])
        assert best['selection']['threshold']==cp['threshold']
    for epoch in range(start,201):
        tick=time.perf_counter();model.train();opt.zero_grad(set_to_none=True)
        logit=model(x,g);loss=objective(logit,y,g,pos)
        assert torch.isfinite(loss);loss.backward()
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step()
        selected=select(d,cal,predict(model,z[cal],d['patient'][cal]));candidate=(selected['macro_f1'],selected['ez_f1'],-epoch)
        if candidate>key:key=candidate;stale=0;best=dict(model=copy.deepcopy(model.state_dict()),epoch=epoch,selection=selected)
        else:stale+=1
        history.append(dict(epoch=epoch,loss=float(loss.detach()),seconds=time.perf_counter()-tick,**selected))
        atomic_torch(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),best=best,key=key,
                               stale=stale,history=history,epoch=epoch))
        if epoch%25==0:print('MACRO_HEAD_EPOCH',fold,screen,arm,epoch,'best',best['epoch'],flush=True)
        if epoch>=10 and stale>=20:break
    atomic_torch(best_path,dict(binding=bind,**best));model.load_state_dict(best['model'])
    logit=ctx['stored']['raw_logit_nez'].copy() if arm=='NATIVE' else predict(model,z,d['patient'])
    assert select(d,cal,logit[cal])==best['selection']
    result=dict(binding=bind,archive_indices=d['archive_indices'],logit_nez=logit,
                threshold=best['selection']['threshold'],epoch=best['epoch'],epochs=len(history),
                parameters=sum(p.numel() for p in model.parameters()),history=history,
                best_sha256=sha(best_path))
    save(scores_path,result);write(done,dict(binding_sha256=digest(bind),best_sha256=sha(best_path),score_sha256=sha(scores_path)))
    print('MACRO_HEAD_CELL_COMPLETE',fold,screen,arm,'epochs',len(history),'selected',best['epoch'],flush=True)
    return result


def train():
    binding=setup();gate=jr(PUB/'ADMISSION.json');assert gate['status']=='PASS' and gate['input_binding_sha256']==digest(binding)
    summaries=[];patient_results=[];seals=[]
    for ctx in contexts():
        z,cp=inputs(ctx);d=ctx['data'];held=ctx['roles']['held']
        for arm in ARMS:
            result=fit(ctx,arm,z[RECIPES[arm]],cp,binding)
            rows=patient_rows(d,held,result['logit_nez'][held],result['threshold'])
            patient_results.extend([dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,**r) for r in rows])
            summaries.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,patients=len(rows),
                epochs=result['epochs'],selected_epoch=result['epoch'],threshold=result['threshold'],
                trainable_parameters=0 if arm=='NATIVE' else result['parameters'],
                **{m:float(np.mean([r[m] for r in rows])) for m in METRICS}))
            path=ROOT/'heads_private'/f"fold{ctx['fold']}_screen{ctx['screen']}"/arm/'SCORES_PRIVATE.pkl'
            seals.append(dict(fold=ctx['fold'],screen=ctx['screen'],arm=arm,sha256=sha(path)))
            pd.DataFrame(summaries).to_csv(PUB/'INTERNAL_CELL_RESULTS.csv',index=False)
            write(PUB/'INTERNAL_STATUS.json',dict(status='FITTING_HEADS',logical_cells=len(summaries),total_cells=120,
                                                no_Outer_or_original_Development_used=True))
        del z,cp;gc.collect()
    assert len(summaries)==120 and sum(r['arm']!='NATIVE' for r in summaries)==105
    save(ROOT/'INTERNAL_PATIENT_ROWS_PRIVATE.pkl',patient_results)
    freeze(ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json',dict(input=binding,cells=seals))
    write(PUB/'INTERNAL_STATUS.json',dict(status='ALL120_INTERNAL_HEAD_CELLS_COMPLETE',logical_cells=120,
        actual_head_fits=105,epochs=sum(r['epochs'] for r in summaries),
        seal_sha256=sha(ROOT/'INTERNAL_SCORE_SEAL_PRIVATE.json'),new_Outer_passes=0))
    print('MACRO_HEAD_INTERNAL_TRAIN_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['admit','train']);a=p.parse_args()
    admit() if a.action=='admit' else train()
