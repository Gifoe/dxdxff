"""25 real scratch fits with original M8 patient objective and validation rules."""
from common import *
import argparse
import time

def admission(prot,core,fac,records,d,inputs,ss):
    seed=42+1009;tr=d['train'];ii=tr[d['patient'][tr]==sorted(set(d['patient'][tr]))[0]]
    raw,mask,times,delta,perms=inputs.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
    x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    source=OLD.P3.make(core,fac,records,OLD.BASELINE/'D1/fold1/TRAIN/BEST_PRIVATE.pt',seed,'M8_MACRO_AWARE');source.eval()
    models={arm:make(core,fac,records,1,arm) for arm in prot['models']}
    assert len({core.state_hash(m.cnn.state_dict()) for m in models.values()})==1
    assert len({core.state_hash(m.head.state_dict()) for m in models.values()})==1
    rows=[]
    for arm,m in models.items():
        m.eval();prefix=core.state_hash(m.cnn.state_dict());assert prefix==core.state_hash(source.cnn.state_dict())
        with torch.inference_mode():
            q=m(x,raw,mask,times,delta,ss)
            md=delta*m.keep7
            expected=source.features(raw,mask,times,md,ss,mode='full' if arm in ['S3','S4'] else 'identity')[0]
            assert torch.equal(q['raw_summary'],expected)
            changed=x.clone();dd=delta.clone()
            if arm=='S0':changed+=123;dd+=321
            elif arm in ['S2','S4']:changed[:,8:10]+=123;dd[...,4]+=321
            if arm in ['S0','S2','S4']:assert torch.equal(m(changed,raw,mask,times,dd,ss)['logit'],q['logit'])
            padded=raw.clone();padded[~mask]=123
            assert torch.equal(q['logit'],m(x,padded,mask,times,delta,ss)['logit'])
        fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda'))
        opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001);norms=[]
        for step in range(3):
            m.train();opt.zero_grad(set_to_none=True);q=m(x,raw,mask,times,delta,ss)
            loss=OLD.P3.objective(q['logit'],y,fn,'M8_MACRO_AWARE');loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
            g={n:float(p.grad.norm().cpu()) for n,p in m.named_parameters()}
            assert g['cnn.0.weight']>0 and g['head.0.weight']>0 and g['head.2.weight']>0
            if arm in ['S3','S4']:
                assert g['gate.network.2.weight']>0
                if step>=1:assert g['gate.network.0.weight']>0
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();norms.append(g)
        rows.append(dict(model=arm,parameters=sum(p.numel() for p in m.parameters()),original_CNN_initialization=True,
            exact_original_raw_summary=True,no_D1_module=True,masked_inputs_independent=arm in ['S0','S2','S4'],padding_ignored=True,smoke_gradients=norms))
    write(PUB/'ADMISSION.json',dict(status='PASS',all5_initial_CNN_and_heads_pairwise_identical=True,all5_real_FIT_gradient_smokes=rows,
        discarded_steps_per_model=3,completed_formal_epochs_discarded=0))

def fit(m,d,inputs,ss,cell,seed,binding,core,arm,fold):
    cell.mkdir(parents=True,exist_ok=True);initial=core.state_hash(m.state_dict())
    bind=dict(input=binding,arm=arm,fold=fold,seed=seed,initial=initial,scale=ss.cpu().tolist())
    opt=torch.optim.AdamW(m.parameters(),lr=.0003,weight_decay=.001);core.seed_all(seed)
    tr,va=d['train'],d['val'];pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1)
    fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'))
    best=key=None;stale=0;history=[];start=1;last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json'
    if last.exists():
        q=torch.load(last,map_location='cuda',weights_only=False);assert q['binding']==bind
        m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng'])
        best=q['best'];key=q['key'];stale=q['stale'];history=q['history'];start=q['epoch']+1
        if done.exists():
            cc=jr(done);assert cc['binding']==bind and cc['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and cc['last_sha256']==sha(last)
            m.load_state_dict(best['model']);return best,history,initial
        if q['epoch']>=6 and stale>=6:start=31
    groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
    for epoch in range(start,31):
        t=time.perf_counter();m.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];raw,mask,times,delta,perms=inputs.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
            x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);q=m(x,raw,mask,times,delta,ss);loss=OLD.P3.objective(q['logit'],y,fn,'M8_MACRO_AWARE')
            assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opt.step();losses.append(float(loss.detach().cpu()))
            del raw,mask,times,delta,perms,x,y,q,loss
        logit=predict(m,d,va,inputs,ss);sel=core.select_threshold(d['y'][va],torch.sigmoid(torch.from_numpy(logit)).numpy(),d['patient'][va])
        candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append(dict(model=arm,fold=fold,epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-t,**sel))
        if key is None or candidate>key:key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        core.torch_write(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,stale=stale,history=history,epoch=epoch))
        write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING',model=arm,fold=fold,epoch=epoch,best_epoch=best['epoch']))
        print('SINGLE_HEAD_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:break
    core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best));write(done,dict(binding=bind,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last)))
    m.load_state_dict(best['model']);return best,history,initial

def main():
    prot,core,fac,records,rec,banks,binding=setup()
    if (ROOT/'MODEL_FREEZE_PRIVATE.pkl').exists():
        assert read(ROOT/'MODEL_FREEZE_PRIVATE.pkl')['binding']==binding
        print('SINGLE_HEAD_ALL25_ALREADY_FROZEN_NO_REFIT',flush=True);return
    inputs=store()
    if not (PUB/'ADMISSION.json').exists():admission(prot,core,fac,records,bank(1,banks),inputs,scale(1))
    selections=[];histories=[];frames=[];init=[]
    for arm in prot['models']:
        for fold in range(1,6):
            d=bank(fold,banks);ss=scale(fold);m=make(core,fac,records,fold,arm)
            init.append(dict(model=arm,fold=fold,cnn=core.state_hash(m.cnn.state_dict()),head=core.state_hash(m.head.state_dict())))
            cell=ROOT/arm/f'fold{fold}';best,h,initial=fit(m,d,inputs,ss,cell,42+1009*fold,binding,core,arm,fold)
            cp=cell/'BEST_PRIVATE.pt';logit=predict(m,d,d['val'],inputs,ss)
            q=dict(logit=logit,checkpoint_sha256=sha(cp),indices=d['archive_indices'][d['val']])
            vp=cell/'VALIDATION_PRIVATE.pkl'
            if vp.exists():old=read(vp);assert old['checkpoint_sha256']==q['checkpoint_sha256'] and np.array_equal(old['logit'],logit)
            else:save(vp,q)
            pn=torch.sigmoid(torch.from_numpy(logit)).numpy();va=d['val']
            frames.append(pd.DataFrame(dict(scope='development',model=arm,version='seed42',fold=fold,patient=d['patient'][va],channel=d['channel'][va],center=d['center'][va],
                y=(d['y'][va]==0).astype(int),probability=1-pn.astype(float),prediction=pn<best['threshold'])))
            selections.append(dict(model=arm,fold=fold,checkpoint=str(cp),checkpoint_sha256=sha(cp),epoch=best['epoch'],threshold=best['threshold'],epochs_run=len(h),
                bank_sha256=sha(OLD.BASELINE/f'D1/fold{fold}/BANK_PRIVATE.pt'),validation=str(vp),validation_sha256=sha(vp),initial=initial))
            histories+=h
            save(ROOT/'PROGRESS_PRIVATE.pkl',dict(binding=binding,selections=selections))
            pd.DataFrame(histories).to_csv(PUB/'TRAINING_HISTORY.csv',index=False)
            print('SINGLE_HEAD_CELL_COMPLETE',arm,fold,len(selections),'of25',flush=True)
            del m,d;gc.collect();torch.cuda.empty_cache()
    for fold in range(1,6):
        rows=[r for r in init if r['fold']==fold];assert len({r['cnn'] for r in rows})==len({r['head'] for r in rows})==1
    oldselections=read(OLD.RUN/'MODEL_SELECTION_PRIVATE.pkl')['selections']
    for fold in range(1,6):
        d=bank(fold,banks);va=d['val']
        for name,variant,head in [('REF_D1','FULL_RETRAIN','D1'),('REF_M8','FULL_RETRAIN','M8'),('REF_M8_NO_VAR','DROP_4','M8')]:
            r=next(q for q in oldselections if q['variant']==variant and q['fold']==fold);q=read(r['validation_path'])
            assert q[head+'_checkpoint_sha256']==r[head+'_checkpoint_sha256'] and np.array_equal(q['validation_archive_indices'],d['archive_indices'][va])
            pn=torch.sigmoid(torch.from_numpy(q[head+'_logit'])).numpy()
            frames.append(pd.DataFrame(dict(scope='development',model=name,version='seed42',fold=fold,patient=d['patient'][va],channel=d['channel'][va],center=d['center'][va],
                y=(d['y'][va]==0).astype(int),probability=1-pn.astype(float),prediction=pn<r[head+'_threshold'])))
    save(ROOT/'DEVELOPMENT_PREDICTIONS_PRIVATE.pkl',pd.concat(frames,ignore_index=True))
    save(ROOT/'MODEL_FREEZE_PRIVATE.pkl',dict(binding=binding,selections=selections,initialization=init))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['checkpoint','validation']} for r in selections]).to_csv(PUB/'MODEL_SELECTION.csv',index=False)
    protected_unchanged(binding)
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL25_SINGLE_HEAD_MODELS_FROZEN',models=25,epochs=len(histories),no_new_Outer=True))

if __name__=='__main__':main()
