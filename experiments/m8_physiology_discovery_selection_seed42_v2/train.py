"""At most80 matched actual fits after immutable own-FIT feature selection."""
import gc
import time
from common import *
from model_io import P3,FeatureHead,Inputs,bank,masked,make,template,restore_cpu_rng

RECIPES=['SELECTED','BEST_SWAP','RANDOM_A','RANDOM_B','OLD7']


def predict(model,d,ix,inputs,scale):
    model.eval();result=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            jj=np.flatnonzero(d['patient'][ix]==pid);ii=ix[jj]
            raw,mask,times,delta,perms=inputs.get(str(pid),d['channel'][ii].astype(str).tolist(),'cuda')
            result[jj]=model(torch.as_tensor(d['x'][ii],device='cuda'),raw,mask,times,delta,scale)[0].cpu().numpy()
    assert np.isfinite(result).all();return result


def fit_raw(m,d,inputs,scale,cell,seed,binding,core,family,fold):
    cell.mkdir(parents=True,exist_ok=True)
    initial=core.state_hash(m.state_dict());base_hash=core.state_hash(m.base.state_dict()) if family=='M8' else None
    bind=dict(input=binding,initial=initial,family=family,fold=fold,seed=seed,base_hash=base_hash,scale=scale.cpu().tolist())
    opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
    core.seed_all(seed);tr,va=d['train'],d['val'];pos=max(int((1-d['y'][tr]).sum()),1)/max(int(d['y'][tr].sum()),1)
    fn=torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos,device='cuda'));best=key=None;stale=0;history=[];start=1
    last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json'
    if last.exists():
        q=torch.load(last,map_location='cuda',weights_only=False);assert q['binding']==bind
        m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng'])
        best=q['best'];key=q['key'];stale=q['stale'];history=q['history'];start=q['epoch']+1
        if done.exists():
            cc=jr(done);assert cc['binding']==bind and cc['best_sha256']==sha(cell/'BEST_PRIVATE.pt') and cc['last_sha256']==sha(last)
            m.load_state_dict(best['model']);return best,history
        if q['epoch']>=6 and stale>=6:start=31
    groups=[tr[ix] for ix in core.groups(d['patient'][tr])]
    for epoch in range(start,31):
        tick=time.perf_counter();m.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];raw,mask,times,delta,perms=inputs.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
            x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True);q=m(x,raw,mask,times,delta,scale)
            loss=P3.objective(q[0],y,fn,'M8_MACRO_AWARE');assert torch.isfinite(loss);loss.backward()
            if family=='M8':assert all(p.grad is None for p in m.base.parameters())
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters())
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step();losses.append(float(loss.detach().cpu()))
            del raw,mask,times,delta,perms,x,y,q,loss
        logit=predict(m,d,va,inputs,scale);sel=core.select_threshold(d['y'][va],torch.sigmoid(torch.from_numpy(logit)).numpy(),d['patient'][va])
        if family=='M8':assert core.state_hash(m.base.state_dict())==base_hash and not m.base.training
        candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),seconds=time.perf_counter()-tick,**sel))
        if key is None or candidate>key:key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        core.torch_write(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,stale=stale,history=history,epoch=epoch))
        write(PUB/'NEURAL_STATUS.json',dict(status='TRAINING',family=family,fold=fold,epoch=epoch,best_epoch=best['epoch']))
        print('MATCHED_RAW_EPOCH',family,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=6 and stale>=6:break
    core.torch_write(cell/'BEST_PRIVATE.pt',dict(binding=bind,**best));write(done,dict(binding=bind,best_sha256=sha(cell/'BEST_PRIVATE.pt'),last_sha256=sha(last)))
    m.load_state_dict(best['model']);return best,history


def admission(core,fac,records,d,inputs,scale,groups):
    tr=d['train'];ii=tr[d['patient'][tr]==sorted(set(d['patient'][tr]))[0]]
    raw,mask,times,delta,perms=inputs.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
    x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.as_tensor(d['y'][ii],device='cuda',dtype=torch.float32)
    original=template(core,fac,records,1);reference=original.features(raw,mask,times,delta[...,:7],scale[:7],mode='identity')[0]
    reports=[];cnn=[]
    for family in ['M8','Concat']:
        m=make(core,fac,records,1,family,groups);m.eval();cnn.append(core.state_hash(m.cnn.state_dict()))
        with torch.inference_mode():
            summary=m.features(raw,mask,times,delta,scale,mode='identity')[0];assert torch.equal(summary,reference)
            result=m(x,raw,mask,times,delta,scale)[0]
            xx=x.clone();dd=delta.clone();inactive=[g for g in range(32) if g not in groups]
            xx[:,np.repeat(inactive,2)*2+np.tile([0,1],len(inactive))]+=123
            dd[...,inactive]+=321
            assert torch.equal(result,m(xx,raw,mask,times,dd,scale)[0])
            padded=raw.clone();padded[~mask]=123
            assert torch.equal(result,m(x,padded,mask,times,delta,scale)[0])
        base=core.state_hash(m.base.state_dict()) if family=='M8' else None
        opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=.0003,weight_decay=.001)
        fn=torch.nn.BCEWithLogitsLoss();grad=[]
        for step in range(3):
            m.train();opt.zero_grad(set_to_none=True);q=m(x,raw,mask,times,delta,scale)
            loss=P3.objective(q[0],y,fn,'M8_MACRO_AWARE');loss.backward()
            ns={n:float(p.grad.norm().cpu()) for n,p in m.named_parameters() if p.grad is not None}
            assert ns['head.2.weight']>0
            if family=='Concat' or step>=1:assert ns['cnn.0.weight']>0 and ns['head.0.weight']>0
            if family=='M8':
                assert core.state_hash(m.base.state_dict())==base and all(p.grad is None for p in m.base.parameters())
                if step>=1:assert ns['gate.network.2.weight']>0
                if step>=2:assert ns['gate.network.0.weight']>0
            torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step();grad.append(ns)
        reports.append(dict(family=family,parameters=sum(p.numel() for p in m.parameters()),original_Raw_identity_parity_exact=True,masked_inputs_independent=True,padding_ignored=True,gradient_steps=grad))
        del m
    assert len(set(cnn))==1
    write(PUB/'NEURAL_ADMISSION.json',dict(status='PASS',actual_complete_FIT=True,three_discarded_steps=True,formal_epochs_discarded=0,reports=reports,CNN_initializers_pairwise_exact=True))


def main():
    c,core,fac,trainer,records,rec,banks,deps=legal()
    original_restore=core.restore_rng
    core.restore_rng=lambda s:original_restore(restore_cpu_rng(s));trainer.restore_rng=core.restore_rng
    torch.set_num_threads(2);torch.set_float32_matmul_precision('highest');torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    screen=jr(PUB/'SCREEN_STATUS.json');assert screen['status']=='ALL5_FIT_ONLY_SELECTIONS_FROZEN'
    selected=jr(ROOT/'SELECTED_FEATURES_PRIVATE.json');assert sha(ROOT/'SELECTED_FEATURES_PRIVATE.json')==screen['feature_freeze_sha256']
    pool=read(ROOT/'POOL_PRIVATE.pkl');reg=read(ROOT/'POOL_REGISTRY_PRIVATE.pkl')
    assert sha(ROOT/'POOL_PRIVATE.pkl')==jr(PUB/'POOL_STATUS.json')['pool_sha256']
    binding=dict(source={n:sha(HERE/n) for n in ['PROTOCOL_LOCK.json','IMPLEMENTATION_LOCK.json','common.py','model.py','model_io.py','train.py']},dependencies=deps,
        selected_features_sha256=sha(ROOT/'SELECTED_FEATURES_PRIVATE.json'),pool_sha256=sha(ROOT/'POOL_PRIVATE.pkl'),pool_registry_sha256=sha(ROOT/'POOL_REGISTRY_PRIVATE.pkl'),
        original_raw_trainer_sha256=sha(Path(P3.__file__)),D1_trainer_sha256=sha(Path(trainer.__file__)),torch=torch.__version__,numpy=np.__version__)
    freeze(ROOT/'NEURAL_INPUT_FREEZE_PRIVATE.json',binding)
    if (ROOT/'MODEL_FREEZE_PRIVATE.pkl').exists():
        fr=read(ROOT/'MODEL_FREEZE_PRIVATE.pkl');assert fr['binding']==binding
        print('MATCHED_MODELS_ALREADY_FROZEN_NO_REFIT',flush=True);return
    inputs=Inputs(reg);selections=[];history=[];frames=[];actual={}
    for fold,original in enumerate(banks,1):
        d=bank(original,pool['summary'],core);seed=42+1009*fold;ss=inputs.scale(d['roles']['fit'])
        old=read(Path(r'D:\d1_small_module_exploration_seed42_v1_runtime')/f'scale_fold{fold}_PRIVATE.pkl')
        assert np.array_equal(ss[:7],old['scale'])
        legacy=torch.load(BASE/f'D1/fold{fold}/BANK_PRIVATE.pt',map_location='cpu',weights_only=False)
        assert np.array_equal(d['x'][:,:14],legacy['x'])
        core.torch_write(ROOT/f'fold{fold}/BANK_PRIVATE.pt',dict(d,binding=binding,scale=ss))
        if fold==1 and not (PUB/'NEURAL_ADMISSION.json').exists():admission(core,fac,records,d,inputs,torch.tensor(ss,device='cuda'),selected['folds']['1']['SELECTED'])
        for recipe in RECIPES:
            groups=selected['folds'][str(fold)][recipe];md=masked(d,groups);signature=digest(dict(fold=fold,groups=sorted(groups)))
            cell=ROOT/'models_private'/f'fold{fold}_{signature}'
            if signature in actual:
                for value in actual[signature]:selections.append(dict(value,recipe=recipe,alias_of=value['recipe']))
                continue
            old_factory=trainer.SourceA0
            try:
                trainer.SourceA0=lambda seed=seed:FeatureHead(seed)
                base,best,h=trainer.train(md['x'],md['y'],md['patient'],md['g'],md['train'],md['val'],cell/'D1',seed,digest(dict(binding=binding,signature=signature)),device='cuda',penalty=0.)
            finally:trainer.SourceA0=old_factory
            cp=cell/'D1/BEST_PRIVATE.pt';va=md['val']
            with torch.inference_mode():logit=base(torch.as_tensor(md['x'][va],device='cuda')).cpu().numpy()
            states=[('D1',cp,best,h,logit)]
            for family in ['M8','Concat']:
                m=make(core,fac,records,fold,family,groups,base.state_dict() if family=='M8' else None)
                best_raw,rh=fit_raw(m,md,inputs,torch.tensor(ss,device='cuda'),cell/family,seed,dict(binding=binding,signature=signature),core,family,fold)
                logit_raw=predict(m,md,va,inputs,torch.tensor(ss,device='cuda'));states.append((family,cell/family/'BEST_PRIVATE.pt',best_raw,rh,logit_raw));del m
            values=[]
            for family,checkpoint,b,h,logit in states:
                vp=cell/f'{family}_VALIDATION_PRIVATE.pkl'
                save(vp,dict(checkpoint_sha256=sha(checkpoint),logit=logit,archive_indices=md['archive_indices'][va],threshold=b['threshold']))
                row=dict(fold=fold,recipe=recipe,family=family,groups=groups,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),epoch=b['epoch'],threshold=b['threshold'],epochs_run=len(h),validation=str(vp),validation_sha256=sha(vp),signature=signature,alias_of=None)
                values.append(row);selections.append(row);history += [dict(fold=fold,recipe=recipe,family=family,**r) for r in h]
            actual[signature]=values
            pd.DataFrame([{k:v for k,v in r.items() if k not in ['checkpoint','validation']} for r in selections]).to_csv(PUB/'NEURAL_SELECTION.csv',index=False)
            pd.DataFrame(history).to_csv(PUB/'NEURAL_HISTORY.csv',index=False)
            save(ROOT/'NEURAL_PROGRESS_PRIVATE.pkl',dict(binding=binding,selections=selections))
            print('MATCHED_RECIPE_FOLD_COMPLETE',recipe,fold,len(selections),'of80 cells',flush=True)
            del base;gc.collect();torch.cuda.empty_cache()
        # Truly branch-free Raw-only, identical Concat head capacity/initialization.
        md=masked(d,[]);m=make(core,fac,records,fold,'Concat',[])
        cell=ROOT/'models_private'/f'RAW_ONLY_fold{fold}'
        b,h=fit_raw(m,md,inputs,torch.tensor(ss,device='cuda'),cell,seed,binding,core,'RawOnly',fold)
        logit=predict(m,md,md['val'],inputs,torch.tensor(ss,device='cuda'));cp=cell/'BEST_PRIVATE.pt';vp=cell/'VALIDATION_PRIVATE.pkl';save(vp,dict(checkpoint_sha256=sha(cp),logit=logit,archive_indices=md['archive_indices'][md['val']],threshold=b['threshold']))
        selections.append(dict(fold=fold,recipe='RAW_ONLY',family='Concat',groups=[],checkpoint=str(cp),checkpoint_sha256=sha(cp),epoch=b['epoch'],threshold=b['threshold'],epochs_run=len(h),validation=str(vp),validation_sha256=sha(vp),signature='RAW_ONLY',alias_of=None))
        history += [dict(fold=fold,recipe='RAW_ONLY',family='Concat',**r) for r in h]
        del m;gc.collect();torch.cuda.empty_cache()
    assert len(selections)==80
    save(ROOT/'MODEL_FREEZE_PRIVATE.pkl',dict(binding=binding,selections=selections))
    pd.DataFrame([{k:v for k,v in r.items() if k not in ['checkpoint','validation']} for r in selections]).to_csv(PUB/'NEURAL_SELECTION.csv',index=False)
    pd.DataFrame(history).to_csv(PUB/'NEURAL_HISTORY.csv',index=False)
    write(PUB/'NEURAL_STATUS.json',dict(status='ALL80_LOGICAL_MODELS_FROZEN',logical_cells=80,actual_models=sum(r['alias_of'] is None for r in selections),epochs=len(history),model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.pkl'),before_Outer=True,source=binding['source']))
    print('ALL80_LOGICAL_NEURAL_MODELS_FROZEN',flush=True)


if __name__=='__main__':main()
