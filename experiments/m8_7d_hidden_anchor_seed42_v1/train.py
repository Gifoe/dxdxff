"""Five actual shared warm fits, fifteen actual Raw joint fits; no new Outer."""
import gc
import time
from common import *
from admit import selected_state


def initialize(core,fac,records,fold,arm,warm=None):
    m=make(core,fac,records,fold,arm if arm in NEW else 'H0_SCRATCH')
    random=state_hash(m.state_dict());warm_sha=None
    if warm is not None:
        warm_sha=sha(warm);q=torch.load(warm,map_location='cpu',weights_only=False)
        assert q['binding']['arm']=='P_WARM_ONLY' and q['binding']['fold']==fold
        assert q['binding']['input']==jr(ROOT/'INPUT_FREEZE_PRIVATE.json') and q['binding']['random_initializer']==random and q['epoch']>=1
        assert state_hash(selected_state(m,('cnn.','gate.','raw_projection.')))==state_hash({k:v for k,v in q['model'].items() if k.startswith(('cnn.','gate.','raw_projection.'))})
        assert torch.count_nonzero(q['model']['raw_projection.weight'])==0;m.load_state_dict(q['model'])
    m.stage('phys' if arm=='P_WARM_ONLY' else 'fixed' if arm=='H2_WARM_FIXED' else 'joint')
    return m,random,warm_sha


def evaluate(m,d,ix,inputs,ss,phase):
    if phase=='joint':return predict(m,d,ix,inputs,ss)
    m.eval()
    with torch.inference_mode():return m.physiology_only(torch.as_tensor(d['x'][ix],device='cuda')).cpu().numpy()


def fit(ctx,arm,binding,inputs,core,fac,records,warm=None):
    fold,d,ss=ctx;phase='phys' if arm=='P_WARM_ONLY' else 'joint';maxepoch,minimum,patience=(100,15,15) if phase=='phys' else (30,6,6)
    m,random,warm_sha=initialize(core,fac,records,fold,arm,warm);initial=state_hash(m.state_dict());seed=42+1009*fold
    cell=ROOT/'models_private'/arm/f'fold{fold}';cell.mkdir(parents=True,exist_ok=True)
    bind=dict(input=binding,arm=arm,fold=fold,phase=phase,initial=initial,random_initializer=random,warm_checkpoint_sha256=warm_sha)
    last,bp,hp,done=[cell/n for n in ['LAST_PRIVATE.pt','BEST_PRIVATE.pt','VALIDATION_PRIVATE.pkl','COMPLETE_PRIVATE.json']]
    if done.exists():
        q=jr(done);assert q['binding']==digest(bind) and sha(bp)==q['best_sha256'] and sha(hp)==q['validation_sha256']
        q=read(hp);assert q['binding']==bind;return q['summary'],q['rows'],q['history']
    seed_all(seed);params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=.0003,weight_decay=.001)
    prefixes=('cnn.','gate.','raw_projection.') if phase=='phys' else ('phys_first.','readout.') if arm=='H2_WARM_FIXED' else ()
    unchanged=state_hash(selected_state(m,prefixes));tr,va=d['train'],d['val'];pos=torch.tensor(float((1-d['y'][tr]).sum()/d['y'][tr].sum()),device='cuda')
    best=key=None;history=[];stale=0;start=1
    if last.exists():
        q=torch.load(last,map_location='cpu',weights_only=False);assert q['binding']==bind;m.load_state_dict(q['model']);opt.load_state_dict(q['optimizer']);core.restore_rng(q['rng'])
        best,key,history,stale,start=q['best'],q['key'],q['history'],q['stale'],q['epoch']+1
        if q['epoch']>=minimum and stale>=patience:start=maxepoch+1
    groups=[tr[d['patient'][tr]==pid] for pid in sorted(set(d['patient'][tr]))]
    for epoch in range(start,maxepoch+1):
        tick=time.perf_counter();m.train();losses=[]
        for j in np.random.default_rng(seed+epoch).permutation(len(groups)):
            ii=groups[int(j)];x=torch.as_tensor(d['x'][ii],device='cuda');y=torch.tensor(d['y'][ii],device='cuda',dtype=torch.float32)
            opt.zero_grad(set_to_none=True)
            if phase=='phys':logit=m.physiology_only(x)
            else:
                raw,mask,times,delta,_=inputs.get(str(d['patient'][ii[0]]),d['channel'][ii].astype(str).tolist(),'cuda')
                logit=m(x,raw,mask,times,delta,ss)['logit'];del raw,mask,times,delta
            loss=MODEL.objective(logit,y,pos);assert torch.isfinite(loss);loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
            assert all(p.grad is None for p in m.parameters() if not p.requires_grad)
            torch.nn.utils.clip_grad_norm_(params,1.);opt.step();losses.append(float(loss.detach().cpu()));del x,y,logit,loss
        assert state_hash(selected_state(m,prefixes))==unchanged
        sel=core.select_threshold(d['y'][va],probability(evaluate(m,d,va,inputs,ss,phase)),d['patient'][va]);candidate=(sel['macro_f1'],sel['ez_f1'],-epoch)
        if key is None or candidate>key:
            key=candidate;stale=0;best=dict(model={k:v.detach().cpu().clone() for k,v in m.state_dict().items()},epoch=epoch,**sel)
        else:stale+=1
        history.append(dict(arm=arm,fold=fold,phase=phase,epoch=epoch,seconds=time.perf_counter()-tick,loss=float(np.mean(losses)),**sel))
        atomic_torch(last,dict(binding=bind,model=m.state_dict(),optimizer=opt.state_dict(),rng=core.rng_state(),best=best,key=key,history=history,stale=stale,epoch=epoch))
        write(PUB/'TRAIN_STATUS.json',dict(status='TRAINING_HIDDEN_ANCHOR',arm=arm,fold=fold,phase=phase,epoch=epoch,best_epoch=best['epoch'],new_Outer_passes=0))
        print('HIDDEN_ANCHOR_EPOCH',arm,fold,epoch,'best',best['epoch'],flush=True)
        if epoch>=minimum and stale>=patience:break
    assert best['epoch']>=1;m.load_state_dict(best['model']);assert state_hash(selected_state(m,prefixes))==unchanged
    atomic_torch(bp,dict(binding=bind,**best));out=evaluate(m,d,va,inputs,ss,phase)
    sel=core.select_threshold(d['y'][va],probability(out),d['patient'][va]);assert all(abs(float(v)-float(best[k]))<1e-12 for k,v in sel.items())
    rows=patient_rows(d,va,out,best['threshold'],arm,fold)
    summary=dict(arm=arm,fold=fold,phase=phase,epochs=len(history),selected_epoch=best['epoch'],threshold=best['threshold'],patients=len(rows),parameters=sum(p.numel() for p in m.parameters()),
        trainable=sum(p.numel() for p in params),checkpoint_sha256=sha(bp),initial_state_sha256=initial,execution_kind='NEW_FIT',**{k:float(np.mean([r[k] for r in rows])) for k in METRICS})
    save(hp,dict(binding=bind,archive_indices=d['archive_indices'][va],logit=out,rows=rows,summary=summary,history=history,unchanged_prefixes=list(prefixes),unchanged_state_sha256=unchanged))
    write(done,dict(binding=digest(bind),best_sha256=sha(bp),validation_sha256=sha(hp)))
    del m,opt;gc.collect();torch.cuda.empty_cache();return summary,rows,history


def main():
    binding,core,fac,records,rec,banks,fr=setup();a=jr(PUB/'ADMISSION.json');assert a['status']=='PASS' and a['input_binding_sha256']==digest(binding)
    inputs=store();summaries=[];rows=[];histories=[];cells=[]
    def record(ctx,arm,warm=None):
        fold=ctx[0];summary,part,history=fit(ctx,arm,binding,inputs,core,fac,records,warm);summaries.append(summary);rows.extend(part);histories.extend(history)
        root=ROOT/'models_private'/arm/f'fold{fold}';cells.append(dict(arm=arm,fold=fold,checkpoint=str(root/'BEST_PRIVATE.pt'),checkpoint_sha256=sha(root/'BEST_PRIVATE.pt'),validation=str(root/'VALIDATION_PRIVATE.pkl'),validation_sha256=sha(root/'VALIDATION_PRIVATE.pkl')))
        pd.DataFrame(summaries).to_csv(PUB/'DEVELOPMENT_CELLS.csv',index=False);pd.DataFrame(histories).to_csv(PUB/'TRAIN_HISTORY.csv',index=False)
        print('HIDDEN_ANCHOR_CELL_COMPLETE',arm,fold,flush=True)
    for fold in range(1,6):
        d=bank(fold,banks);ss=scale(fold);ctx=(fold,d,ss);r,state,q,out,error=reference(fr,fold,core,fac,records,d,inputs,ss)
        part=patient_rows(d,d['val'],out,state['threshold'],'C2_REFERENCE',fold);rows+=part
        summaries.append(dict(arm='C2_REFERENCE',fold=fold,phase='reused',epochs=0,prior_training_epochs=q['summary']['epochs'],selected_epoch=state['epoch'],threshold=state['threshold'],patients=len(part),parameters=10378,trainable=10378,
            checkpoint_sha256=r['checkpoint_sha256'],initial_state_sha256=q['binding']['initial'],execution_kind='REUSED_EXACT_C2',**{k:float(np.mean([z[k] for z in part])) for k in METRICS}))
        cells.append(dict(r,arm='C2_REFERENCE'))
        record(ctx,'P_WARM_ONLY');warm=ROOT/'models_private/P_WARM_ONLY'/f'fold{fold}'/'BEST_PRIVATE.pt'
        for arm in NEW:record(ctx,arm,None if arm=='H0_SCRATCH' else warm)
    assert len(cells)==25
    save(ROOT/'DEVELOPMENT_PATIENTS_PRIVATE.pkl',rows);freeze(ROOT/'MODEL_FREEZE_PRIVATE.json',dict(input=binding,cells=cells))
    selection=choose(pd.DataFrame(rows));freeze(ROOT/'CANDIDATE_PRIVATE.json',dict(input=binding,model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),selection=selection))
    warm=sum(r['phase']=='phys' for r in histories);joint=len(histories)-warm
    write(PUB/'TRAIN_STATUS.json',dict(status='ALL20_NEW_PLUS5_REUSED_HIDDEN_ANCHOR_FROZEN',actual_new_fits=20,warm_fits=5,joint_fits=15,reused_controls=5,
        warm_epochs=warm,joint_epochs=joint,epochs=len(histories),model_freeze_sha256=sha(ROOT/'MODEL_FREEZE_PRIVATE.json'),candidate_sha256=sha(ROOT/'CANDIDATE_PRIVATE.json'),new_Outer_passes=0,goal_established=False))
    print('HIDDEN_ANCHOR_ALL_TRAIN_VALIDATION_COMPLETE',flush=True)


if __name__=='__main__':main()
