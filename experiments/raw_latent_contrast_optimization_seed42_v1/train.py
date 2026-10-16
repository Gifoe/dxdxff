"""Bounded20-cell end-to-end Raw study, own-FIT recipes and Development selection."""
import argparse
import copy
import gc
import time
import pandas as pd
from scipy.special import expit
from common import *
from engine import cached,direct,rng,restore
from model import RawModel
from metrics import patient_rows,threshold

ARMS=['B0_ORIGINAL_POOL_MACRO_LOSS','B1_ORIGINAL_POOL_EZ_LOSS',
      'S1_SELECTED_POOL_EZ_LOSS','A1_SELECTED_POOL_ATTENTION_EZ_LOSS']
STAGE2_SOURCE=['PROTOCOL_LOCK.json','model.py','common.py','metrics.py','engine.py','train.py']


def stage2_setup():
    base=setup();status=jr(PUB/'STAGE1_STATUS.json')
    assert status['status']=='ALL16_FACTORIAL_SCREEN_AND_FOLD_LOCAL_RECIPES_COMPLETE'
    assert sha(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json')==status['recipe_freeze_sha256']
    pools=jr(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json');assert pools['binding']==base and pools['own_FIT_only']
    binding=dict(stage1=base,pools=sha(ROOT/'POOL_RECIPE_FREEZE_PRIVATE.json'),
        source={n:sha(HERE/n) for n in STAGE2_SOURCE},scikit_learn=__import__('sklearn').__version__)
    freeze(ROOT/'STAGE2_INPUT_FREEZE_PRIVATE.json',binding)
    _,banks=identity_and_banks(labels=True)
    return binding,pools['recipes'],banks


def config(arm,fold,recipes):
    recipe=(1,1,1,1) if arm.startswith('B') else tuple(map(int,recipes[str(fold)]))
    return dict(recipe=recipe,attention=arm.startswith('A'),loss='macro' if arm.startswith('B0') else 'ez_rank',
                threshold_kind='macro' if arm.startswith('B0') else 'ez')


def build(arm,fold,recipes):
    c=config(arm,fold,recipes)
    return RawModel(fold,c['recipe'],c['attention']).cuda(),c


def predict(model,d,ix,store):
    model.eval();logit=np.empty(len(ix),np.float32)
    with torch.inference_mode():
        for pid in sorted(set(d['patient'][ix])):
            loc=np.flatnonzero(d['patient'][ix]==pid);ii=ix[loc]
            raw,mask,times=store.get(pid,d['channel'][ii]);q,_=cached(model,raw,mask,times)
            logit[loc]=q.cpu().numpy()
    assert np.isfinite(logit).all();return logit


def select(d,logit,kind):
    va=d['val'];y=(d['y'][va]==0).astype(int);prob=expit(-logit.astype(float))
    theta=threshold(y,prob,d['patient'][va],kind)
    rows=patient_rows(y,prob,d['patient'][va],d['channel'][va],d['center'][va],theta['threshold'])
    return dict(threshold=theta['threshold'],**{m:float(np.nanmean([r[m] for r in rows])) for m in METRICS})


def admit():
    binding,recipes,banks=stage2_setup();store=RawStore();reports=[]
    for arm in ARMS:
        d=banks[0];ix=d['train'];pid=sorted(set(d['patient'][ix]))[0];ii=ix[d['patient'][ix]==pid]
        raw,mask,times=store.get(pid,d['channel'][ii]);y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
        pos=torch.tensor(float((1-d['y'][ix]).sum())/max(float(d['y'][ix].sum()),1),device='cuda')
        a,c=build(arm,1,recipes);b=copy.deepcopy(a);a.train();b.train();before=rng()
        loss,logit,z=direct(a,raw,mask,times,y,pos,c['loss']);loss.backward();after=rng()
        restore(before);cached_loss,cached_logit,cached_z=cached(b,raw,mask,times,y,pos,c['loss']);after_cached=rng()
        assert torch.equal(logit,cached_logit) and torch.equal(z,cached_z)
        assert abs(float(loss.detach().cpu())-cached_loss)==0
        maximum=0.;relative=0.
        for (na,pa),(nb,pb) in zip(a.named_parameters(),b.named_parameters()):
            assert na==nb and (pa.grad is None)==(pb.grad is None)
            if pa.grad is None:continue
            assert torch.isfinite(pa.grad).all() and torch.isfinite(pb.grad).all()
            error=float((pa.grad-pb.grad).abs().max().cpu());maximum=max(maximum,error)
            norm=float(pa.grad.abs().max().cpu());relative=max(relative,error/max(norm,1e-12))
            assert torch.allclose(pa.grad,pb.grad,atol=1e-6,rtol=1e-5),('Full-patient gradient gate',arm,na,error,norm)
        assert torch.equal(after['cpu'],after_cached['cpu']) and all(torch.equal(x,y) for x,y in zip(after['cuda'],after_cached['cuda']))
        reports.append(dict(arm=arm,patient_channels=len(ii),genuine_windows=int(mask.sum().cpu()),
            logit_max_error=0.,loss_error=0.,full_gradient_max_error=maximum,maximum_tensor_relative_error=relative,
            RNG_exact=True,parameters=sum(p.numel() for p in a.parameters()),physiology_inputs=0,D1_branches=0))
        del a,b,raw,mask,times,loss,logit,z,cached_logit,cached_z;gc.collect();torch.cuda.empty_cache()
    write(PUB/'END_TO_END_ADMISSION.json',dict(status='PASS',binding_sha256=digest(binding),reports=reports,
        all_complete_patient=True,head_pooling_Z_and_CNN_gradients_checked=True,formal_epochs_discarded=0,
        precision='FP32_TF32_OFF',gradient_gate_absolute_tolerance=1e-6,gradient_gate_relative_tolerance=1e-5))
    print('RAW_END_TO_END_FULL_PATIENT_ADMISSION_PASS',flush=True)


def run():
    binding,recipes,banks=stage2_setup();admission=jr(PUB/'END_TO_END_ADMISSION.json')
    assert admission['status']=='PASS' and admission['binding_sha256']==digest(binding)
    store=RawStore();selections=[];all_history=[]
    for fold,d in enumerate(banks,1):
        initial=[]
        for arm in ARMS:
            model,c=build(arm,fold,recipes);state=state_hash(model.state_dict())
            shared=state_hash({k:v for k,v in model.state_dict().items() if k.startswith(('cnn.','head.'))})
            initial.append(shared);assert len(set(initial))==1
            cell=ROOT/'models_private'/f'{arm}_fold{fold}';cell.mkdir(parents=True,exist_ok=True)
            bind=dict(input=binding,fold=fold,arm=arm,configuration=clean(c),initial_sha256=state)
            opt=torch.optim.AdamW(model.parameters(),lr=.0003,weight_decay=.001)
            seed=42+1009*fold;seed_all(seed);tr,va=d['train'],d['val'];start=1;best=key=None;history=[];stale=0
            pos=torch.tensor(float((1-d['y'][tr]).sum())/max(float(d['y'][tr].sum()),1),device='cuda')
            last=cell/'LAST_PRIVATE.pt';done=cell/'COMPLETE_PRIVATE.json';best_path=cell/'BEST_PRIVATE.pt'
            if last.exists():
                q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind
                model.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);restore(q['rng'])
                best=q['best'];key=q['key'];history=q['history'];stale=q['stale'];start=q['epoch']+1
                if done.exists():
                    dd=jr(done);assert dd['binding_sha256']==digest(bind) and dd['last_sha256']==sha(last) and dd['best_sha256']==sha(best_path)
                    start=31
                elif q['epoch']>=6 and stale>=6:start=31
            groups=[tr[d['patient'][tr]==p] for p in sorted(set(d['patient'][tr]))]
            for epoch in range(start,31):
                tick=time.perf_counter();losses=[];model.train()
                for number,j in enumerate(np.random.default_rng(seed+epoch).permutation(len(groups)),1):
                    ii=groups[int(j)];pid=d['patient'][ii[0]];raw,mask,times=store.get(pid,d['channel'][ii])
                    y=torch.tensor(d['y'][ii],dtype=torch.float32,device='cuda');opt.zero_grad(set_to_none=True)
                    value,_,_=cached(model,raw,mask,times,y,pos,c['loss'])
                    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
                    torch.nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();losses.append(value)
                    if number%10==0:write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING',arm=arm,fold=fold,
                        epoch=epoch,patients_complete=number,patients_total=len(groups),no_new_Outer_access=True))
                    del raw,mask,times,y
                logit=predict(model,d,va,store);selected=select(d,logit,c['threshold_kind'])
                candidate=(selected['macro_f1'],selected['f1'],-epoch) if arm.startswith('B0') else (selected['f1'],selected['macro_f1'],-epoch)
                history.append(dict(epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**selected))
                if key is None or candidate>key:
                    key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},epoch=epoch,selection=selected)
                else:stale+=1
                atomic_torch(last,dict(binding=bind,model=model.state_dict(),optimizer=opt.state_dict(),rng=rng(),
                    best=best,key=key,history=history,stale=stale,epoch=epoch))
                print('RAW_LATENT_EPOCH',arm,fold,epoch,'seconds',round(history[-1]['seconds'],2),'best',best['epoch'],flush=True)
                if epoch>=6 and stale>=6:break
            assert best is not None
            if not done.exists():
                atomic_torch(best_path,dict(binding=bind,**best))
                write(done,dict(binding_sha256=digest(bind),best_sha256=sha(best_path),last_sha256=sha(last)))
            model.load_state_dict(best['model']);logit=predict(model,d,va,store);selected=select(d,logit,c['threshold_kind'])
            assert selected==best['selection']
            # Both decision rules are validation-only sensitivities of the same frozen weights.
            ez=select(d,logit,'ez');macro=select(d,logit,'macro')
            vp=cell/'VALIDATION_PRIVATE.pkl';save(vp,dict(checkpoint_sha256=sha(best_path),
                archive_indices=d['archive_indices'][va],logit_nez=logit,primary=selected,ez=ez,macro=macro))
            selections.append(dict(arm=arm,fold=fold,checkpoint=str(best_path),checkpoint_sha256=sha(best_path),
                validation=str(vp),validation_sha256=sha(vp),configuration=clean(c),epoch=best['epoch'],epochs_run=len(history),
                threshold=selected['threshold'],ez_threshold=ez['threshold'],macro_threshold=macro['threshold'],
                selection=selected,parameters=sum(p.numel() for p in model.parameters()),shared_initial_sha256=shared))
            all_history.extend([dict(fold=fold,arm=arm,**r) for r in history])
            pd.DataFrame(all_history).to_csv(PUB/'END_TO_END_TRAIN_HISTORY.csv',index=False)
            write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING',complete_cells=len(selections),total_cells=20,
                epochs=sum(s['epochs_run'] for s in selections),no_new_Outer_access=True))
            print('RAW_LATENT_TRAIN_CELL_COMPLETE',arm,fold,len(selections),'of20',flush=True)
            del model,opt;gc.collect();torch.cuda.empty_cache()
    assert len(selections)==20
    deployment={}
    for fold in range(1,6):
        candidates=[r for r in selections if r['fold']==fold]
        chosen=max(candidates,key=lambda r:(r['selection']['f1'],r['selection']['macro_f1'],-r['parameters'],-ARMS.index(r['arm'])))
        deployment[str(fold)]=chosen['arm']
    freeze(ROOT/'MODEL_FREEZE_PRIVATE.json',dict(binding=binding,selections=selections,deployment=deployment))
    public=[]
    for row in selections:
        public.append({k:v for k,v in row.items() if k not in ['checkpoint','validation','configuration','selection']}|
            dict(recipe=''.join(map(str,row['configuration']['recipe'])),**{'validation_'+k:v for k,v in row['selection'].items()}))
    pd.DataFrame(public).to_csv(PUB/'END_TO_END_SELECTION.csv',index=False)
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL20_RAW_MODELS_AND_DEPLOYMENT_FROZEN',complete_cells=20,
        total_cells=20,epochs=sum(s['epochs_run'] for s in selections),deployment=deployment,
        model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),no_new_Outer_access=True))
    print('ALL_RAW_LATENT_MODELS_FROZEN',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['admit','train']);a=p.parse_args()
    admit() if a.action=='admit' else run()
